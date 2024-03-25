# Multi-Objectives program REPAIR framework

### How to clone this repository
```
git clone https://github.com/fedebotu/clone-anonymous-github.git && cd clone-anonymous-github
python3 src/download.py --url https://anonymous.4open.science/r/morepair-1024
cd morepair-1024
```

### Environment Preparation (Local)
Recommend System: Ubuntu 20.04

Java version: **11.0.21**

Python version: 3.10.11

CUDA Version: **12.0**

1. install libboost, maven, zstd, and javac11.0.21
```
apt install libboost-all-dev maven openjdk-11-jdk zstd
```

2. install tiktoken difflib
```
pip3 install tokenizers==0.15.0
```

(Optional) If you want to train the model, you need to install the following python packages.
```
pip3 install torch==2.0.1+cu117 transformers==4.36.2 wandb==0.16.0 peft==0.6.1 trl==0.7.4 numpy==1.24.2
```

3. unzip data files
```
zstd -d evalrepair-java.zst -o evalrepair-java.tar && tar -xvf evalrepair-java.tar
zstd -d evalrepair-cpp-res.zst -o evalrepair-cpp-res.tar && tar -xvf evalrepair-cpp-res.tar
zstd -d evalrepair-java-res.zst -o evalrepair-java-res.tar && tar -xvf evalrepair-java-res.tar
```

4. prepare the EvalRepair-Java

### Environment Preparation (Docker)
Recommend System: Ubuntu 20.04

Docker version: 20.10.17

```
```



### Model Training (Optional)

```
mkdir output_model || true  # create model output directory
python3 MOTrain.py 1 # run the trainer, with the gamma value of 1
```

You can make MOTrain.py do single-objective fine-tuning by setting its first parameter to zero.

The fine-tuned model is saved in the output_model directory and can be tested for inference by executing the following command.

```
python3 test.py
```


### RQ-1: How effective is fine-tuning with two objectives for program repair?
```
python3 rq1.py
```

### RQ-2: How does model size or type impact repair performance of MORepair?

### RQ-3: How does MORepair compare against MORepair with human guidance and state-of-the-art fine-tuning methods?

