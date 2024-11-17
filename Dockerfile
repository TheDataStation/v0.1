FROM python:3.9
#FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/python:0-3.9

ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/ds

RUN apt-get clean && apt-get update && apt-get install -y fuse && apt-get install libfuse-dev && rm -rf /var/lib/apt/lists/*

RUN mkdir /usr/src/ds/functions
RUN mkdir /usr/src/ds/pickled
RUN mkdir /usr/src/ds/connectors
RUN mkdir -p /mnt/data_mount

COPY ds_dev_utils/docker/image/requirements.txt .
RUN pip install -r requirements.txt

COPY ds_dev_utils/docker/image .

#RUN brew install openblas
#RUN pip install dowhy
#RUN pip install --no-cache-dir -r requirements.txt

COPY common common
COPY crypto crypto
COPY escrowapi escrowapi
COPY dsapplicationregistration dsapplicationregistration