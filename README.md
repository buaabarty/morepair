# Multi-Objective Fine-Tuning for Enhanced Program Repair with LLMs

## MORepair
MORepair, a novel **M**ulti-**O**bjective fine-tuning framework designed specifically for LLM-based program **Repair**. MORepair steers LLMs towards a precise understanding the reasoning logic behind the repair process, thereby enabling them to generate high-quality patches.


#### Quick Start
```
git clone https://github.com/fedebotu/clone-anonymous-github.git && cd clone-anonymous-github
python3 src/download.py --url https://anonymous.4open.science/r/morepair-1024
cd morepair-1024
apt install docker.io zstd
zstd -d evalrepair-java.zst -o evalrepair-java.tar && tar -xvf evalrepair-java.tar
zstd -d evalrepair-cpp-res.zst -o evalrepair-cpp-res.tar && tar -xvf evalrepair-cpp-res.tar
cat evalrepair-java-res.zst.part-* > evalrepair-java-res.zst && zstd -d evalrepair-java-res.zst -o evalrepair-java-res.tar && tar -xvf evalrepair-java-res.tar
docker build -t morepair .
docker run -it -v `pwd`/:/opt/morepair morepair
cd /opt/morepair
```

After all, run the following command within docker:
```
bash rq1.sh
bash rq2.sh
bash rq3.sh
```

## I) Requirements

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
cat evalrepair-java-res.zst.part-* > evalrepair-java-res.zst && zstd -d evalrepair-java-res.zst -o evalrepair-java-res.tar && tar -xvf evalrepair-java-res.tar
```

### C) Environment Preparation
You should build the evaluation environment through docker, and then evaluate experimental results within the docker image.

- Docker version: 20.10.17

```
docker build -t morepair .
docker run -it -v `pwd`/:/opt/morepair morepair
cd /opt/morepair
```

## II) Experiments
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

## III) Fine-tune & Inference

### Environment Preparation for Fine-tuning
- Recommend System: Ubuntu 20.04
- Python version: 3.10.11
- CUDA Version: **12.0**

Install the following python library.

```
pip3 install torch==2.0.1+cu117 transformers==4.36.2 wandb==0.16.0 peft==0.6.1 trl==0.7.4 numpy==1.24.2
```

### Regenerate Experimental Results

It will take a long time. Suggest running this command in the background. You can fine-tune a specific model and re-generate the results by executing the following commands.

```
# you should run this command at first, only once
python3 fetch_data.py

# download evalrepair-java dataset
zstd -d evalrepair-java.zst -o evalrepair-java.tar && tar -xvf evalrepair-java.tar

# fine-tune and inference
bash finetune_and_inference.sh CodeLlama-13b-Instruct-hf llama_llm codellama13b-stdft 0
```

Parameters are configured as follows:

| Model (`$3`) | Base Model (`$1`) | Dataset (`$2`) | Lambda (`$4`) |
|-|-|-|-|
| codellama13b-stdft | CodeLlama-13b-Instruct-hf | llama_llm | 0 |
| codellama13b-morepair | CodeLlama-13b-Instruct-hf | llama_llm | 1 |
| codellama13b-cot | CodeLlama-13b-Instruct-hf | llama_cot | 0 |
| codellama13b-human | CodeLlama-13b-Instruct-hf | llama_human | 1 |
|||||
| codellama7b-stdft | CodeLlama-7b-Instruct-hf | llama_llm | 0 |
| codellama7b-morepair | CodeLlama-7b-Instruct-hf | llama_llm | 1 |
| codellama7b-cot | CodeLlama-7b-Instruct-hf | llama_cot | 0 |
| codellama7b-human | CodeLlama-7b-Instruct-hf | llama_human | 1 |
|||||
| starchat-stdft | starchat-alpha | starchat_llm | 0 |
| starchat-morepair | starchat-alpha | starchat_llm | 1 |
| starchat-cot | starchat-alpha | starchat_cot | 0 |
| starchat-human | starchat-alpha | starchat_human | 1 |
|||||
| mistral-stdft | Mistral-7B-Instruct-v0.1 | llama_llm | 0 |
| mistral-morepair | Mistral-7B-Instruct-v0.1 | llama_llm | 1 |
| mistral-cot | Mistral-7B-Instruct-v0.1 | llama_cot | 0 |
| mistral-human | Mistral-7B-Instruct-v0.1 | llama_human | 1 |

If you want to reproduce the results of RepairLLaMA, you can run the following command.

```
bash repairllama.sh
```

After re-inference, you can follow the steps in the `rq1.sh`, `rq2.sh`, and `rq3.sh` files to generate the evaluation results.

## Tutorial to Train Your Own Multi-Objective Fine-Tuning Model
1. Prepare the dataset

The dataset designated for fine-tuning needs to be formatted similarly to `data/trainset/*.json`, comprising a list of dictionaries. Each dictionary should have a solitary key, `text`, whose value is the amalgamation of three parts: the input, the output for the first objective, and the output for the second objective. These parts are separated by an end-of-sentence (EOS) token, such as `</s>` for Llama, and the entire value should also terminate with an EOS token. When employing a model that has undergone instruction-based fine-tuning as the base model, it is recommended to present the input data in the established instruction format. This approach aids in achieving superior fine-tuning outcomes.

2. Fine-tuning

To conduct multi-objective fine-tuning, the `MOTrain.py` script requires the provision of three parameters. These are the name of the base model, the file location of the dataset, and the directory for saving the fine-tuned model. Please note, that the script is designed to store the trained model within a specified subdirectory of the `./models` directory.

3. Inference

When deploying the fine-tuned model for inference tasks, it is straightforward to utilize the model files found in the `./codellama_merged` subdirectory, which is nested within the relevant `./models` subdirectory. These model files can be freely copied for any intended use. For guidance on the inference process, you can refer to the methodology outlined in the `inference_cpp.py` script. It is advisable to employ 8-bit quantization to optimize the use of computational resources during inference.