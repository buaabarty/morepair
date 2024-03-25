import requests
import os
import json
def get_humaneval_cpp(name):
    url = f"http://170.187.174.223:9999/humanevalcpp/{name}"
    headers = {
        "API-KEY": "tutorcode_api_key_temp_52312"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(response.status_code)
        return {'status': 'failed'}

def get_tutorcodeplus(name, type):
    url = f"http://170.187.174.223:9999/tutorcodeplus/{name}/{type}"
    headers = {
        "API-KEY": "tutorcode_api_key_temp_52312"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(response.status_code)
        return {'status': 'failed'}

def create_file(base_path, category, name, content):
    dir_path = os.path.join(base_path, category)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{name}.cpp")
    with open(file_path, 'w') as file:
        file.write(content)

def save_content(file_path, content):
    with open(file_path, 'w') as file:
        file.write(json.dumps(content))

if __name__ == "__main__":
    base_path = 'data/evalrepair-c++'
    
    with open('config/cpp.list', 'r') as file:
        for line in file.readlines():
            name = line.strip()
            item = get_humaneval_cpp(name)
            
            if 'status' not in item or item['status'] != 'failed':
                create_file(base_path, 'buggy', name, item['buggy'])
                create_file(base_path, 'correct', name, item['correct'])
                create_file(base_path, 'prompt', name, item['prompt'])
                create_file(base_path, 'diff', name, item['diff'])
            else:
                print(f"Failed to get data for {name}")
    sys.exit(0)
    os.makedirs('data/trainset', exist_ok=True)
    save_content('data/trainset/llama_llm.json', get_tutorcodeplus("llama", "llm"))
    save_content('data/trainset/llama_human.json', get_tutorcodeplus("llama", "human"))
    save_content('data/trainset/llama_cot.json', get_tutorcodeplus("llama", "cot"))
    save_content('data/trainset/starchat_llm.json', get_tutorcodeplus("starchat", "llm"))
    save_content('data/trainset/starchat_human.json', get_tutorcodeplus("starchat", "human"))
    save_content('data/trainset/starchat_cot.json', get_tutorcodeplus("starchat", "cot"))
