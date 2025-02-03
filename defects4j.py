import os
import sys
import torch
from pathlib import Path
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

BOF = '[INST]'
EOF = '[/INST]'

if sys.argv[3][0] in ['8', '9'] or len(sys.argv[3]) > 4 and sys.argv[3][1] in ['8', '9']:
    BOF = '[INST]'
    EOF = '[/INST]'
elif sys.argv[3][0] == '7' or len(sys.argv[3]) > 4 and sys.argv[3][1] == '7':
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

def cal(bug_id, code, title, description, filename):
    prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```java\n"
    print(prompt, flush=True)
    cnt = len(tokenizer.tokenize(prompt))
    max_d = cnt
    while True:
        output = pipe(prompt, min_length=cnt+64, max_length=cnt+max_d, temperature=1.0, do_sample=True)
        full_text = output[0]['generated_text']
        print(full_text)
        ret = extract_first_java_code(full_text.split('[/INST]')[1])
        print('code:', ret, flush=True)
        if ret.strip() != '':
            break
    return [full_text, ret]

if sys.argv[3][0] not in ['8', '3', '1']:
    sys.exit(0)

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
elif sys.argv[3] in ['18001', '8001']:
    use_triton = True
    model_name = '/root/autodl-tmp/CodeLlama-7B-Instruct-GPTQ'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:" + sys.argv[4],
        use_triton=use_triton)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
elif sys.argv[3] in ['18101', '8101']:
    use_triton = True
    model_name = '/root/autodl-tmp/CodeLlama-13B-Instruct-GPTQ'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name,
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:" + sys.argv[4],
        quantize_config=None,
        disable_exllama=False,
        inject_fused_attention=True,  # 使用融合注意力机制
        inject_fused_mlp=True,  # 使用融合 MLP
        use_triton=use_triton)#,load_in_8bit=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
else:
    model_path = '/root/autodl-tmp/codellama_finetune/' + sys.argv[3][1:] + '/codellama_merged'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:"+sys.argv[4],
        load_in_8bit=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print('load model success ..', flush=True)

base_dir = '/root/autodl-tmp/apr/defects4j/dataset/'
fix_dir = '/root/autodl-tmp/apr/defects4j/results/' + sys.argv[3] + '/fixed'

cnt = 0

for file_path in sorted(Path(base_dir).rglob('*.json'), reverse=True):
    cnt += 1
    if cnt % int(sys.argv[1]) != int(sys.argv[2]):
        continue
        
    # 获取文件的完整路径
    full_path = str(file_path)
    print(full_path, flush=True)
    
    # 读取文件内容
    with open(full_path, 'r') as file:
        content = file.read()

    json_data = json.loads(content)
    result_data = json_data

    for e in range(int(sys.argv[-1])):
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
        full, res = cal(file_name.split('.')[0], json_data['buggy'], json_data['issue_title'], json_data['issue_description'], json_data['loc'])
        if full == None:
            continue
        result_data['fix'] = res
        with open(fix_name, 'w') as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)
        with open(fix_name + '.log', 'w') as file:
            print(full, file=file)
