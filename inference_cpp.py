import os
import sys
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer
import re

def extract_first_cpp_code(s: str) -> str:
    # 寻找所有匹配的代码块
    matches = re.findall(r'```c\+\+(.*?)```', s, re.DOTALL)
    # 返回最后一个匹配的代码块，如果没有找到匹配的代码块则返回空字符串
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

def remove_comments_and_return_first_part(code):
    comments = re.findall(r'/\*(.*?)\*/', code, flags=re.DOTALL)
    first_comment = comments[0].strip() if comments else ''
    input_content_match = re.search(r'INPUT.*', first_comment, flags=re.DOTALL)
    input_content = input_content_match.group(0) if input_content_match else ''
    code_without_comments = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return input_content, code_without_comments

def cal(code, filename):
    prompt_suffix = open('data/evalrepair-c++/prompt/' + filename, 'r', encoding='utf-8').read()
    _, prompt_suffix = remove_comments_and_return_first_part(prompt_suffix)
    prompt = BOF + " This is an incorrect code (" + filename + "):\n```c++\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```c++\n" + prompt_suffix
    print(prompt, flush=True)
    cnt = len(tokenizer.tokenize(prompt))
    max_d = 500
    while True:
        output = pipe(prompt, min_length=cnt+64, max_length=cnt+max_d, temperature=1.0, do_sample=True)
        full_text = output[0]['generated_text']
        print(full_text)
        ret = extract_first_cpp_code(full_text.split(EOF)[1])
        print('code:', ret, flush=True)
        if ret.strip() != '':
            break
        max_d = min(3000 - cnt, max_d + 500)
    return [full_text, ret]



base_dir = 'data/evalrepair-c++/buggy/'
fix_dir = 'evalrepair-cpp-res/' + sys.argv[1] + '/fixed'

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.cpp'), reverse=True):
    # 获取文件的完整路径
    full_path = str(file_path)
    print(full_path, flush=True)

    # 读取文件内容
    with open(full_path, 'r', encoding='utf-8') as file:
        content = file.read()

    print(content)

    for e in range(10):
        file_name = os.path.basename(full_path)
        fix_name = os.path.join(fix_dir + str(e) + '/', file_name)
        print(fix_name, flush=True)
        full, res = cal(content, file_name)
        if full == None:
            continue
        with open(fix_name, 'w', encoding='utf-8') as file:
            print(res, file=file)
        with open(fix_name + '.log', 'w', encoding='utf-8') as file:
            print(full, file=file)