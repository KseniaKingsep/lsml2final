from celery import Celery
from celery.result import AsyncResult
from flask import Flask, redirect, render_template, request, url_for
from quant_gan import TCN, TemporalBlock, Generator, Discriminator
from train import train_quantgan
from utils import *
import json
import numpy as np
import mlflow
import mlflow.pytorch
import torch
import yfinance as yf

app = Flask(__name__) 
celery_app = Celery('server', backend='redis://redis', broker='redis://redis')  # both database and broker are redis
MLFLOW_SERVER_URL = 'http://mlflow:5000/'
mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
client = mlflow.tracking.MlflowClient(MLFLOW_SERVER_URL)

@celery_app.task(bind=True)
def build_model(self, name):
    data = yf.download(name, '2010-01-01', '2021-09-01')
    data_log = np.log(data['Close'] / data['Close'].shift(1))[1:].values
    if len(data_log) < 1000:
        self.update_state(state='FAILED', meta={'No enough data'})
        return None
    model = train_quantgan(data_log, num_epochs = 4, celery_task = self)
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, "model", registered_model_name = f'gan_{name}')
    return name

@app.route('/')  # handler for /
def hello():
    return render_template("hello.html")

@app.route('/checker', methods=['POST'])
def check_handler():
    symbol = request.form['symbol']
    print(symbol)
    models = client.search_registered_models("name='gan_{}'".format(symbol))
    if len(models) == 0:
        return redirect(url_for("generate_handler", symbol=symbol))
    return redirect(url_for("get_handler", symbol=symbol))

@app.route('/generate/<symbol>', methods=["GET"])
def generate_handler(symbol):
    task = build_model.delay(symbol)
    return redirect(url_for("status_handler", task_id=task.id))

@app.route('/status/<task_id>', methods=["GET"])
def status_handler(task_id):
    task = AsyncResult(task_id, app=celery_app)
    if task.ready():
        return redirect(url_for("get_handler", symbol=task.result))
    print(task.state, task.info)
    info = task.info
    if info is None:
        info = {"done": 0, "total": "Unknown"}
    return render_template("status.html", done=info["done"], total=info["total"])

def generate_fakes(model, n=1, cumsum=True):
    fakes = []
    for i in range(n):
        noise = torch.randn(1, 2000, 3).cpu()
        fake = model(noise).detach().cpu().reshape(2000).numpy()
        fake = inverse(fake * model.preprocess_params["data_max"], model.preprocess_params["params"]) + model.preprocess_params["data_mean"]
        fakes.append(fake)
    if n > 1:
        if not cumsum:
            return pd.DataFrame(fakes).T
        return pd.DataFrame(fakes).T.cumsum()
    if not cumsum:
        return fake
    return fake.cumsum()

@app.route('/gan/<symbol>', methods=["GET"])
def get_handler(symbol):
    models = client.search_registered_models("name='gan_{}'".format(symbol))
    model = mlflow.pytorch.load_model(model_uri=f'models:/gan_{symbol}/{models[0].latest_versions[0].version}')
    fake = generate_fakes(model).tolist()
    return json.dumps(fake)

if __name__ == '__main__':
    app.run("0.0.0.0", 8000) 
