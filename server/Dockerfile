FROM ubuntu:latest
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3.8 python3-pip python3-dev tmux git
RUN pip3 -q install pip --upgrade
COPY ../requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm requirements.txt
WORKDIR src/
COPY common/ .
COPY server/ .
