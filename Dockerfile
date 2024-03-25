FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# for Chinese users
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    maven \
    openjdk-8-jdk \
    zstd \
    python3.10 \
    python3-pip

RUN apt install -y libssl-dev

RUN pip3 install tokenizers==0.15.0

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

ADD evalrepair-java /evalrepair-java
RUN cd /evalrepair-java && mvn test
RUN rm -r /evalrepair-java