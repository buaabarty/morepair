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
    prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```python\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code without modifying any code indentations?\n" + EOF + "\n```python\n"
    print(prompt, flush=True)
    cnt = len(tokenizer.tokenize(prompt))
    max_d = cnt
    retry_cnt = 0
    while True:
        output = pipe(prompt, min_length=cnt+64, max_length=cnt+max_d, do_sample=False, top_p=0)
        full_text = output[0]['generated_text']
        print(full_text)
        ret = extract_first_python_code(full_text.split(EOF)[1])
        print('code:', ret, flush=True)
        if retry_cnt > 0:
            break
        retry_cnt += 1
        #    return [None, None]
        max_d = min(1500 - cnt, max_d + 100)
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

cnt = 0
# 从 dataset.jsonl 读取输入
with open('dataset.jsonl', 'r') as f:
    for line in f:
        cnt += 1
        if cnt % int(sys.argv[1]) != int(sys.argv[2]):
            continue

        json_data = json.loads(line.strip())
        instance_id = json_data['instance_id']
        print(f'Processing instance: {instance_id}', flush=True)


        for e in range(int(sys.argv[-1])):
            instance_dir = f'/root/autodl-tmp/apr/swebench/results/{sys.argv[3]}_{e}'
            os.makedirs(instance_dir, exist_ok=True)
            output_file = f'/root/autodl-tmp/apr/swebench/results/{sys.argv[3]}_{e}.jsonl'
            log_file = f'{instance_dir}/{instance_id}.log'
            # 检查当前文件中是否已处理过该实例
            existing_ids = set()
            if os.path.exists(log_file):
                print(f'Skipping existing instance {instance_id} in attempt {e}', flush=True)
                continue

            full, res = cal(instance_id, json_data['buggy_code'],
                          json_data['problem_statement'].split('\n')[0],
                          json_data['problem_statement'],
                          json_data['file_path'])
            if full == None:
                continue

            # 保存完整的LLM输出到日志文件
            with open(log_file, 'w') as f:
                f.write(full)

            # 构建要追加的数据
            result_data = {
                'instance_id': instance_id,
                'file_path': json_data['file_path'],
                'buggy_code': json_data['buggy_code'],
                'problem_statement': json_data['problem_statement'],
                'fixed_code': res,
                'model_output': full
            }

            # 立即追加到jsonl文件
            with open(output_file, 'a') as f:
                f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
