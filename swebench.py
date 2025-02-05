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

def cal(bug_id, code, title, description, filename):
    prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```python\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code without modifying any code indentations?\n" + EOF + "\n```python\n"
    print(prompt, flush=True)
    cnt = len(tokenizer.tokenize(prompt))
    if cnt > 1000:
        return [None, None]
    max_d = cnt
    retry_cnt = 0
    while True:
        output = pipe(prompt, min_length=cnt+64, max_length=cnt+max_d, do_sample=False, top_p=0)
        full_text = output[0]['generated_text']
        print(full_text)
        ret = extract_first_python_code(full_text.split(EOF)[1])
        print('code:', ret, flush=True)
        #if max_d > 500 - cnt:
        if retry_cnt > 0:
            break
        retry_cnt += 1
        #    return [None, None]
        max_d = min(1500 - cnt, max_d + 100)
    return [full_text, ret]

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
