FROM python:3.10-slim-buster

WORKDIR /app

RUN apt-get update && \
    apt-get install build-essential curl unzip file git ruby-full locales --no-install-recommends -y && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install -r requirements.txt