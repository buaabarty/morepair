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
    prompt = BOF + "\n# " + title + '\n' + description + '\n' + "This is an incorrect code (" + filename + "):\n```java\n" + code + "\n```\nYou are a software engineer. Can you repair the incorrect code?\n" + EOF + "\n```java\n"
    print(prompt, flush=True)
    cnt = len(tokenizer.tokenize(prompt))
    if cnt >= 1000:
        print('prompt too long', bug_id, flush=True)
        # 删除 dataset/bug_id.json
        if os.path.exists('/root/autodl-tmp/apr/defects4j/dataset/' + bug_id + '.json'):
            os.remove('/root/autodl-tmp/apr/defects4j/dataset/' + bug_id + '.json')
        return [None, None]
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
