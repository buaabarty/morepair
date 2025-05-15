# MORepair: Teaching LLMs to Repair Code via Multi-Objective Fine-tuning

<p align="center">
  <a href="https://doi.org/10.1145/3735129"><img src="https://img.shields.io/badge/DOI-10.1145/3735129-blue.svg" alt="Paper DOI"></a>
  <a href="https://colab.research.google.com/drive/1vlabdN5Oucm-5kVtMHuEw-kvqDOtB5hg"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.0.1+-orange.svg" alt="PyTorch 2.0.1+">
</p>

## Cite Our Work

If you use MORepair in your research, please cite our paper:

```bibtex
@article{10.1145/3735129,
author = {Yang, Boyang and Tian, Haoye and Ren, Jiadong and Zhang, Hongyu and Klein, Jacques and Bissyande, Tegawende and Le Goues, Claire and Jin, Shunfu},
title = {MORepair: Teaching LLMs to Repair Code via Multi-Objective Fine-Tuning},
year = {2025},
publisher = {Association for Computing Machinery},
issn = {1049-331X},
url = {https://doi.org/10.1145/3735129},
doi = {10.1145/3735129},
journal = {ACM Trans. Softw. Eng. Methodol.},
}
```

## Colab Demo

Explore MORepair with our Colab Notebook:
[MORepair Demo](https://colab.research.google.com/drive/1vlabdN5Oucm-5kVtMHuEw-kvqDOtB5hg)

## 📚 Datasets

MORepair is evaluated on five carefully curated datasets, covering different programming languages and repair scenarios:

### 🎯 Training Dataset
| Dataset | Description | Size | Language | Obtain |
|---------|-------------|------|----------|---------|
| TutorLLMCode | High-quality C++ code repair dataset with human and LLM-generated rationales | 1.2K | C++ | [Website](https://tutorcode.org/docs/) |

### 🏆 Evaluation Benchmarks
| Dataset | Description | Size | Language | Obtain |
|---------|-------------|------|----------|---------|
| EvalRepair-Java | Real-world Java program repair benchmark derived from HumanEval | 163 | Java | [Hugging Face](https://huggingface.co/datasets/barty/EvalRepair-Java) |
| EvalRepair-C++ | Real-world C++ program repair benchmark derived from HumanEval | 164 | C++ | [Hugging Face](https://huggingface.co/datasets/barty/EvalRepair-Cpp) |
| D4J-Repair | single-function subset of Defects4J | 371 | Java | [Hugging Face](https://huggingface.co/datasets/barty/D4J-Repair) |
| SWE-Repair | single-function subset of SWE-Bench | 204 | Multi | [Hugging Face](https://huggingface.co/datasets/barty/SWE-Repair) |

> 💡 **Note**: All datasets are preprocessed and ready to use. For detailed dataset statistics and usage instructions, please refer to our [paper](https://doi.org/10.1145/3735129).

## MORepair Framework Overview

MORepair is a novel **M**ulti-**O**bjective fine-tuning framework designed specifically for LLM-based program **Repair**. It steers LLMs toward a precise understanding of the reasoning logic behind the repair process, thereby enabling them to generate high-quality patches.

## Key Features

*   🚀 **Multi-Objective Fine-Tuning:** A novel approach for significantly enhanced code repair capabilities.
*   🧠 **Improved Reasoning Logic:** Guides LLMs to deeply understand the "why" behind code fixes, not just the "what."
*   🛠️ **High-Quality Patch Generation:** Empowers LLMs to produce more accurate and reliable code patches.
*   📄 **Instruction-Following Enhancement:** Particularly effective with instruction-tuned base models.
*   🐳 **Dockerized & Reproducible:** Easy setup with Docker ensures consistent environments for research and development.
*   🧩 **Extensible & Adaptable:** Designed to be flexible for various models and custom datasets.

## Quick Start & Environment Setup

Get up and running with MORepair using Docker.

1.  **Prerequisites:**
    *   `docker.io`
    *   `zstd` (for decompressing datasets, if you plan to use the provided ones)

2.  **Build the Docker Image:**
    Clone this repository, then navigate to its root directory and run:
    ```bash
    docker build -t morepair .
    ```

3.  **Run the Docker Container:**
    ```bash
    # Mount your local MORepair repository (replace /path/to/your/local/morepair with the actual path)
    docker run -it -v /path/to/your/local/morepair:/opt/morepair morepair
    cd /opt/morepair
    ```
    *Tip: On Linux/macOS, use `$(pwd)` for the current path: `docker run -it -v $(pwd):/opt/morepair morepair`*

## Using MORepair

Follow these steps to leverage MORepair for your program repair tasks.

### 1. Prepare Your Custom Dataset

Your fine-tuning dataset should be a JSON file containing a list of dictionaries. Each dictionary must have a single key, `text`, whose value is a string formed by concatenating:
    1.  The input (e.g., buggy code with instructions)
    2.  The output for the first objective (e.g., the rationale or thought process)
    3.  The output for the second objective (e.g., the corrected code)

These three parts must be separated by an End-Of-Sentence (EOS) token (e.g., `</s>` for Llama). The entire `text` value must also end with an EOS token. For instruction-tuned base models, formatting the input as per the model's instruction template is highly recommended for optimal performance.

Refer to `data/trainset/llama_llm.json` for an example and [TutorLLMCode.md](TutorLLMCode.md) for details on the TutorLLMCode dataset structure.

**(Optional) Download Preprocessed TutorLLMCode Datasets:**
Within the Docker container, run:
```bash
python3 fetch_data.py
```
This script downloads `llama_human.json` (human-generated rationales) and `llama_llm.json` (GPT-4 generated rationales) for single-file C++ buggy programs.

### 2. Perform Multi-Objective Fine-tuning

Use the `MOTrain.py` script for fine-tuning. Key arguments include:
    *   `--base_model_name_or_path`: Name or path of your base LLM.
    *   `--dataset_path`: Path to your prepared JSON dataset.
    *   `--output_model_dir_name`: Subdirectory name under `./models` to save the fine-tuned model.

**Example (inside Docker):**
```bash
python3 MOTrain.py \\
    --base_model_name_or_path CodeLlama-7b-Instruct-hf \\
    --dataset_path data/trainset/llama_llm.json \\
    --output_model_dir_name my_custom_codellama7b
```

### 3. Model Inference

The fine-tuned model (LoRA adapters and potentially merged model) will be saved in `./models/<your_output_model_dir_name>`. For inference, use the model files from the `./codellama_merged` (or similarly named) subdirectory.

The `inference_cpp.py` script provides an example of an inference pipeline. Using 8-bit quantization is recommended to optimize resource usage.

## (Optional) Reproducing Paper Results

For those interested in replicating the results from our paper:

1.  **Full Dataset Download & Setup:**
    Refer to our paper or earlier `README.md` versions for detailed instructions on acquiring and preparing datasets like the full EvalRepair series, Defects4J, and SWE-bench (often involving `*.zst` decompression).
    Example (inside Docker):
    ```bash
    # zstd -d evalrepair-java.zst -o evalrepair-java.tar && tar -xvf evalrepair-java.tar
    ```

2.  **Execution Scripts:**
    Use the provided `rqN.sh` scripts (e.g., `rq1.sh`, `rq2.sh`) within the Docker container. These scripts typically include re-judging options.

3.  **Specific Model Fine-tuning & Inference:**
    The `finetune_and_inference.sh` script can be used.
    Example (inside Docker):
    ```bash
    # python3 fetch_data.py # If TutorLLMCode is not yet downloaded
    # bash finetune_and_inference.sh CodeLlama-13b-Instruct-hf llama_llm codellama13b-stdft 0
    ```
    Consult the original `README.md`'s parameter table for model, dataset, and lambda configurations.
