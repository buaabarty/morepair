import os
import sys
import re
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer

def extract_first_java_code(s: str) -> str:
    matches = re.findall(r'```java(.*?)```', s, re.DOTALL)
    return matches[0].strip() if matches else ""

BOF = '[INST]'
EOF = '[/INST]'

if sys.argv[1].startswith('codellama') or sys.argv[1].startswith('mistral'):
    BOF = '[INST]'
    EOF = '[/INST]'
elif sys.argv[1].startswith('starchat'):
    BOF = '<|system|>\n<|end|>\n<|user|>'
    EOF = '<|end|>\n<|assistant|>'
else:
    print('parameter error ...', flush=True)
    sys.exit(0)

if sys.argv[1] == 'codellama13b-baseline':
    model_path = 'CodeLlama-13B-Instruct-GPTQ'
elif sys.argv[1] == 'codellama7b-baseline':
    model_path = 'CodeLlama-7B-Instruct-GPTQ'
elif sys.argv[1] == 'starchat-baseline':
    model_path = 'starchat-alpha'
elif sys.argv[1] == 'mistral-baseline':
    model_path = 'Mistral-7B-Instruct-v0.1'
else:
    model_path = './models/' + sys.argv[1] + '/codellama_merged'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", load_in_8bit=True)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print('load model success ..', flush=True)

def cal(code, filename):
    prompt = BOF + " This is an incorrect code(" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```java\n"
    print(prompt, flush=True)
    cnt = len(tokenizer.tokenize(prompt))
    max_d = 500 - cnt
    while True:
        output = pipe(prompt, min_length=cnt+64, max_length=cnt+max_d, temperature=1.0, do_sample=True)
        full_text = output[0]['generated_text']
        print(full_text)
        ret = extract_first_java_code(full_text.split('[/INST]')[1])
        print('code:', ret, flush=True)
        if ret.strip() != '':
            break
        max_d = min(3000 - cnt, max_d + 500)
    return [full_text, ret]

base_dir = 'evalrepair-java/origin/'
fix_dir = 'evalrepair-java-res/' + sys.argv[1] + '/fixed'

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.java'), reverse=True):
    full_path = str(file_path)
    print(full_path, flush=True)

    with open(full_path, 'r') as file:
        content = file.read()

    print(content)

    for e in range(10):
        file_name = os.path.basename(full_path)
        fix_name = os.path.join(fix_dir + str(e) + '/', file_name)
        print(fix_name, flush=True)
        full, res = cal(content, file_name)
        if full == None:
            continue
        with open(fix_name, 'w') as file:
            print(res, file=file)
        with open(fix_name + '.log', 'w') as file:
            print(full, file=file)