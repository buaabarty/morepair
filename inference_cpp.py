import os
import sys
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import re
import os

def extract_first_cpp_code(s: str) -> str:
    # 寻找所有匹配的代码块
    matches = re.findall(r'```c\+\+(.*?)```', s, re.DOTALL)
    # 返回最后一个匹配的代码块，如果没有找到匹配的代码块则返回空字符串
    return matches[0].strip() if matches else ""

BOF = '[INST]'
EOF = '[/INST]'

if sys.argv[3][0] in ['8', '9'] or len(sys.argv[3]) > 4 and sys.argv[3][1] in ['8', '9']:
    BOF = '[INST]'
    EOF = '[/INST]'
elif sys.argv[3][0] == '7':
    BOF = '<|system|>\n<|end|>\n<|user|>'
    EOF = '<|end|>\n<|assistant|>'
elif len(sys.argv[3]) > 4 and sys.argv[3][1] == '2':
    BOF = '###Instruction\n'
    EOF = '###Response\n'
elif len(sys.argv[3]) > 4 and sys.argv[3][1] == '6':
    BOF = 'GPT4 Correct User: '
    EOF = '<|end_of_turn|>GPT4 Correct Assistant: '
else:
    BOF = 'You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n### Instruction:\n'
    EOF = '### Response:'

if sys.argv[3] == '16001':
    model_path = '/root/autodl-tmp/openchat-3.5-0106'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:"+sys.argv[4], load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
elif sys.argv[3] in ['12001']:
    model_path = '/root/autodl-tmp/stablecode-instruct-alpha-3b'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:"+sys.argv[4], load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
elif sys.argv[3] in ['19001', '9001']:
    model_path = '/root/autodl-tmp/Mistral-7B-Instruct-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:"+sys.argv[4], load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
elif sys.argv[3] == '3001':
    model_path = '/root/autodl-tmp/deepseek-coder-7b-base-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:"+sys.argv[4], load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
elif sys.argv[3] in ['14001', '4001']:
    model_path = '/root/autodl-tmp/deepseek-coder-7b-instruct-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:"+sys.argv[4], load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
elif sys.argv[3] == '7001':
    tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/starchat-alpha")
    model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/starchat-alpha", device_map="cuda:"+sys.argv[4], load_in_8bit=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
elif sys.argv[3] == '8001':
    use_triton = False
    model_name = '/root/autodl-tmp/CodeLlama-7B-Instruct-GPTQ'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:" + sys.argv[4],
        use_triton=use_triton,load_in_8bit=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
elif sys.argv[3] == '8101':
    use_triton = False
    model_name = '/root/autodl-tmp/CodeLlama-13B-Instruct-GPTQ'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:" + sys.argv[4],
        use_triton=use_triton,load_in_8bit=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
else:
    model_path = '/root/autodl-tmp/codellama_finetune/' + sys.argv[3][1:] + '/codellama_merged'
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:"+sys.argv[4], load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

if sys.argv[3][0] not in ['9', '8', '7', '4', '1']:
    sys.exit(0)

print('load model success ..', flush=True)

def remove_comments_and_return_first_part(code):
    comments = re.findall(r'/\*(.*?)\*/', code, flags=re.DOTALL)
    first_comment = comments[0].strip() if comments else ''
    input_content_match = re.search(r'INPUT.*', first_comment, flags=re.DOTALL)
    input_content = input_content_match.group(0) if input_content_match else ''
    code_without_comments = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return input_content, code_without_comments


def cal(code, filename):
    prompt_suffix = open('/root/autodl-tmp/apr/humaneval-cpp/prompt/' + filename, 'r', encoding='utf-8').read()
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
        #if max_d > 500 - cnt:
        #    return [None, None]
        max_d = min(3000 - cnt, max_d + 500)
    return [full_text, ret]



base_dir = '/root/autodl-tmp/apr/humaneval-cpp/buggy/'
fix_dir = '/root/autodl-tmp/humaneval-cpp/' + sys.argv[3] + '/fixed'

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.cpp'), reverse=True):
    # 获取文件的完整路径
    full_path = str(file_path)
    print(full_path, flush=True)

    # 读取文件内容
    with open(full_path, 'r', encoding='utf-8') as file:
        content = file.read()

    print(content)

    for e in range(int(sys.argv[-1])):
        cnt += 1
        if cnt % int(sys.argv[1]) != int(sys.argv[2]):
            continue
        # 获取文件名
        file_name = os.path.basename(full_path)
        fix_name = os.path.join(fix_dir + str(e) + '/', file_name)
        print(fix_name, flush=True)
        if os.path.exists(fix_name) and os.path.exists(fix_name + '.log'):
            print('result exists ...')
            continue
        #cnt += 1
        #if cnt % int(sys.argv[1]) != int(sys.argv[2]):
        #    continue
        full, res = cal(content, file_name)
        if full == None:
            continue
        with open(fix_name, 'w', encoding='utf-8') as file:
            print(res, file=file)
        with open(fix_name + '.log', 'w', encoding='utf-8') as file:
            print(full, file=file)