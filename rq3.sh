rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
if [ "$2" = "train" ]; then
    mkdir models || true
    python3 MOTrain.py CodeLlama-13b-Instruct-hf data/trainset/llama_cot.json models/codellama13b-cot 0
    python3 MOTrain.py CodeLlama-13b-Instruct-hf data/trainset/llama_human.json models/codellama13b-cot 1
    python3 MOTrain.py CodeLlama-7b-Instruct-hf data/trainset/llama_cot.json models/codellama7b-cot 0
    python3 MOTrain.py CodeLlama-7b-Instruct-hf data/trainset/llama_human.json models/codellama7b-cot 1
    python3 MOTrain.py starchat-alpha data/trainset/llama_cot.json models/starchat-cot 0
    python3 MOTrain.py starchat-alpha data/trainset/llama_human.json models/starchat-cot 1
    python3 MOTrain.py Mistral-7B-Instruct-v0.1 data/trainset/llama_cot.json models/mistral-cot 0
    python3 MOTrain.py Mistral-7B-Instruct-v0.1 data/trainset/llama_human.json models/mistral-cot 1
    
    python3 inference_cpp.py codellama13b-cot
    python3 inference_cpp.py codellama13b-human
    python3 inference_java.py codellama13b-cot
    python3 inference_java.py codellama13b-human
    python3 inference_cpp.py codellama7b-cot
    python3 inference_cpp.py codellama7b-human
    python3 inference_java.py codellama7b-cot
    python3 inference_java.py codellama7b-human
    python3 inference_cpp.py starchat-cot
    python3 inference_cpp.py starchat-human
    python3 inference_java.py starchat-cot
    python3 inference_java.py starchat-human
    python3 inference_cpp.py mistral-cot
    python3 inference_cpp.py mistral-human
    python3 inference_java.py mistral-cot
    python3 inference_java.py mistral-human
    
    git clone git@github.com:ASSERT-KTH/repairllama.git
    python3 inference_repairllama_cpp.py
fi
python3 calc_cpp.py codellama13b-cot $1 > logs/cpp-codellama13b-cot.log
python3 calc_cpp.py codellama13b-human $1 > logs/cpp-codellama13b-human.log
python3 calc_java.py codellama13b-cot $1 > logs/java-codellama13b-cot.log
python3 calc_java.py codellama13b-human $1 > logs/java-codellama13b-human.log

python3 calc_cpp.py codellama7b-cot $1 > logs/cpp-codellama7b-cot.log
python3 calc_cpp.py codellama7b-human $1 > logs/cpp-codellama7b-human.log
python3 calc_java.py codellama7b-cot $1 > logs/java-codellama7b-cot.log
python3 calc_java.py codellama7b-human $1 > logs/java-codellama7b-human.log

python3 calc_cpp.py starchat-cot $1 > logs/cpp-starchat-cot.log
python3 calc_cpp.py starchat-human $1 > logs/cpp-starchat-human.log
python3 calc_java.py starchat-cot $1 > logs/java-starchat-cot.log
python3 calc_java.py starchat-human $1 > logs/java-starchat-human.log

python3 calc_cpp.py mistral-cot $1 > logs/cpp-mistral-cot.log
python3 calc_cpp.py mistral-human $1 > logs/cpp-mistral-human.log
python3 calc_java.py mistral-cot $1 > logs/java-mistral-cot.log
python3 calc_java.py mistral-human $1 > logs/java-mistral-human.log

python3 calc_cpp.py repairllama $1 > logs/cpp-repairllama.log
python3 calc_java.py repairllama $1 > logs/java-repairllama.log

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 1 "$log_file"
done
