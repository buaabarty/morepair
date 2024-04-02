import json
from pathlib import Path
import sys
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path
import os

import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("CodeLlama-7b-hf", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    model,
    'repairllama/repairllama-lora',
    torch_dtype=torch.float16,
)
model.config.pad_token = tokenizer.pad_token = tokenizer.unk_token
model.to(device)

def merge_fixed_code(original_code, fixed_code_fragment):
    lines = original_code.split('\n')
    merged_code = []
    for line in lines:
        if '<FILL_ME>' in line:
            merged_code.append(fixed_code_fragment)
            continue
        else:
            merged_code.append(line)
    return '\n'.join(merged_code)

def calc(code):
    try:
        inputs = tokenizer(code, return_tensors="pt")
        inputs_len = inputs["input_ids"].shape[1]
        inputs_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            num_beams=10,
            early_stopping=True,
        )

        outputs = model.generate(
            input_ids=inputs_ids,
            max_new_tokens=256,
            num_return_sequences=10,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
        )

        output_ids = outputs[:, inputs_len:]
        output_patch = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        ret = []
        for each in output_patch:
            fixed = merge_fixed_code(code, each.split('</s>')[0])
            print(fixed)
            ret.append(fixed)
        return ret
    except:
        return ['' for i in range(10)]

base_dir = 'evalrepair-java/origin/'
fix_dir = 'evalrepair-java-res/repairllama/fixed'

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.java'), reverse=False):
    full_path = str(file_path)
    file_name = os.path.basename(full_path)
    print(full_path, flush=True)
    content = open('evalrepair-java/diff/' + file_name, 'r', encoding='utf-8').read()
    print(content)
    ret = calc(content)

    for e in range(10):
        fix_name = os.path.join(fix_dir + str(e) + '/', file_name)
        print(fix_name, flush=True)

        with open(fix_name, 'w', encoding='utf-8') as file:
            print(ret[e], file=file)
        with open(fix_name + '.log', 'w', encoding='utf-8') as file:
            print('[/INST]\n```java\n' + ret[e] + '\n```\n', file=file)