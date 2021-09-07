# Financial Time Series Generation project

## Project purpose

This project aims to generate synthetic time series for financial securities using QuantGAN implementation from [Wiese et al., Quant GANs: Deep Generation of Financial Time Series, 2019](https://arxiv.org/abs/1907.06673). 
We use the Yahoo Finance data. The user can select the specific stock via web interface and get the generated data. The generated data can be used for experiments or other models (not in the scope this project).  

For easiness please select symbols from [S&P 500 list](https://en.wikipedia.org/wiki/List_of_S\%26P_500_companies), for example AAPL (Apple), ACN (Accenture), ETSY (Etsy), FB (Facebook), SLB (Schlumberger), V (Visa), NVDA (Nvidia). The app can work for others too, but those from S&P are in the Yahoo Finance database for sure. 


## Quick start

To run the whole application, execute the following commands from the project directory: 
 
 ```
 docker-compose up
 ```
 
 Then enter in your web browser:  `http://localhost:8000/`

Prerequisites for manually creation of environments can be found in `requirements.txt`. 

## Architecture
The app consists of the following microservices:
-	Flask: an http server; connected to Celery via port 8000
-	Celery: asynchronous task processing; connected to Flask (port 8000) and MLFlow (port 5000)
-	Redis: stores Celery data
-	MLFlow: track model training 
-	MySQL DB: stores mlflow models, communicates with MLFlow via port 3306
-	Jupyter notebook: play with models


After the user enters the symbol, the app checks whehter there is a registered MLFlow model for this symbol. 
If the model exists, we generate new sample from the existing model (from MySQL database) and the new page with the results is open. Otherwise, celery creates a task to create a new model. The new page is open, the user can see the progress and wait a bit till the model is trained. We train model for 4 epochs in order not to consume too much time waiting. The results are saved to MLFlow (models are stored in MySQL database).

## Dataset 
We use the Yahoo finance historical market data (downloaded via Python `yfinance` library, [documentation](https://pypi.org/project/yfinance/)). The downloaded time series include daily stock price data. We can generate one time series at a time. 
We download data for the period 01.01.2010 between 01.09.2021. We use the following data: date and the close price for the stock. 

## Model
The project features my implementation of the QuantGAN using PyTorch. The Quant GAN uses Temporal Convolutional Network. The model itself can be found in the `quant_gan.py` file. 

### Loss
We use standard generative adversarial networks loss. 
Minimax loss: 
$$E_x[\log(D(x))] + E_z[\log(1 - D(G(z)))]$$
 
The generator tries to minimize the function while the discriminator tries to maximize it.

### Metrics
It is hard to assess the quality of the GANs, but we can use Earth Mover Distance (Wasserstein-1 metric) that compares two 1-D distributions. 

