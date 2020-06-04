ARG PARENT=hub.docker.prod.walmart.com/tensorflow/tensorflow:1.15.0-gpu-py3

FROM ${PARENT}
MAINTAINER Binwei Yang <byang@walmartlabs.com>

WORKDIR /app

COPY requirements.txt /app

RUN pip install -i https://repository.walmart.com/repository/pypi-proxy/simple/ -r requirements.txt
RUN pip install -i https://repository.walmart.com/repository/pypi-proxy/simple/ streamlit

COPY *.py /app/
COPY *.sh /app/
COPY Makefile /app

ENV PYTHONPATH /app
