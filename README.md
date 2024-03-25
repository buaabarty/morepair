# Multi-Objective Fine-Tuning for Enhanced Program Repair with LLMs

## MORepair
MORepair ...

#### Catalogue of Repository
```
evalrepair-
```

## I) Dataset

## II) Requirements

### A) Clone this Anonymous Repository
```
git clone https://github.com/fedebotu/clone-anonymous-github.git && cd clone-anonymous-github
python3 src/download.py --url https://anonymous.4open.science/r/morepair-1024
cd morepair-1024
```

### B) Unarchive Datasets
```
zstd -d evalrepair-java.zst -o evalrepair-java.tar && tar -xvf evalrepair-java.tar
zstd -d evalrepair-cpp-res.zst -o evalrepair-cpp-res.tar && tar -xvf evalrepair-cpp-res.tar
zstd -d evalrepair-java-res.zst -o evalrepair-java-res.tar && tar -xvf evalrepair-java-res.tar
```

## III) Experiment

### 2a) Recommend: Environment Preparation (Docker)
Docker version: 20.10.17

```
docker build -t morepair .
docker run -it -v `pwd`/:/opt/morepair morepair
cd /opt/morepair
```

### 2b) Optional: Environment Preparation (Local)
Recommend System: Ubuntu 20.04

Java version: **11.0.21**

Python version: 3.10.11

CUDA Version: **12.0**

1. install libboost, maven, zstd, openssl, and javac11.0.21
```
apt install libboost-all-dev maven openjdk-11-jdk zstd libssl-dev
```

2. install tiktoken difflib
```
pip3 install tokenizers==0.15.0
```

(Optional) If you want to train the model, you need to install the following python packages.
```
pip3 install torch==2.0.1+cu117 transformers==4.36.2 wandb==0.16.0 peft==0.6.1 trl==0.7.4 numpy==1.24.2
```

## Evaluation
Evaluation should be run in the docker container described above.

### RQ-1) Effectiveness of Multi-objective Fine-tuning for Program Repair

```
bash rq1.sh
# or run the following command to rejudge all the results
bash rq1.sh rejudge
```

### RQ-2) Impact of Size or Type for Fine-tuning LLMs on Code Repair Performance

```
bash rq2.sh
# or run the following command to rejudge all the results
bash rq2.sh rejudge
```

### RQ-3) Evaluating the Impact of Guidance Sources and Comparing MOREPAIR against State-of-the-Art Fine-tuning Methods

```
bash rq3.sh
# or run the following command to rejudge all the results
bash rq3.sh rejudge
```

### Optional: Fine-tune

```
mkdir output_model || true  # create model output directory
python3 MOTrain.py 1 # run the trainer, with the gamma value of 1
```

You can make MOTrain.py do single-objective fine-tuning by setting its first parameter to zero.

The fine-tuned model is saved in the output_model directory and can be tested for inference by executing the following command.

```
python3 test.py
```

### Optional: Inference

```
mkdir output_model || true  # create model output directory
python3 MOTrain.py 1 # run the trainer, with the gamma value of 1
```

You can make MOTrain.py do single-objective fine-tuning by setting its first parameter to zero.

The fine-tuned model is saved in the output_model directory and can be tested for inference by executing the following command.

```
python3 test.py
```
