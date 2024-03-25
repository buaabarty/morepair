FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    maven \
    openjdk-11-jdk \
    zstd \
    python3.10 \
    python3-pip

RUN pip3 install tokenizers==0.15.0
RUN pip3 install torch==2.0.1+cu117 transformers==4.36.2 wandb==0.16.0 peft==0.6.1 trl==0.7.4 numpy==1.24.2

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
