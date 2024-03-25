rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
if [ "$2" = "train" ]; then
    mkdir models || true
    python3 MOTrain.py CodeLlama-7b-Instruct-hf data/trainset/llama_llm.json models/codellama7b-stdft 0
    python3 MOTrain.py CodeLlama-7b-Instruct-hf data/trainset/llama_llm.json models/codellama7b-morepair 1
    python3 MOTrain.py starchat-alpha data/trainset/starchat_llm.json models/starchat-stdft 0
    python3 MOTrain.py starchat-alpha data/trainset/starchat_llm.json models/starchat-morepair 1
    python3 MOTrain.py Mistral-7B-Instruct-v0.1 data/trainset/llama_llm.json models/mistral-stdft 0
    python3 MOTrain.py Mistral-7B-Instruct-v0.1 data/trainset/llama_llm.json models/mistral-morepair 1

    python3 inference_cpp.py codellama7b-baseline
    python3 inference_cpp.py codellama7b-stdft
    python3 inference_cpp.py codellama7b-morepair
    python3 inference_java.py codellama7b-baseline
    python3 inference_java.py codellama7b-stdft
    python3 inference_java.py codellama7b-morepair

    python3 inference_cpp.py starchat-baseline
    python3 inference_cpp.py starchat-stdft
    python3 inference_cpp.py starchat-morepair
    python3 inference_java.py starchat-baseline
    python3 inference_java.py starchat-stdft
    python3 inference_java.py starchat-morepair

    python3 inference_cpp.py mistral-baseline
    python3 inference_cpp.py mistral-stdft
    python3 inference_cpp.py mistral-morepair
    python3 inference_java.py mistral-baseline
    python3 inference_java.py mistral-stdft
    python3 inference_java.py mistral-morepair
fi
python3 calc_cpp.py codellama7b-baseline $1 > logs/cpp-codellama7b-baseline.log
python3 calc_cpp.py codellama7b-stdft $1 > logs/cpp-codellama7b-stdft.log
python3 calc_cpp.py codellama7b-morepair $1 > logs/cpp-codellama7b-morepair.log
python3 calc_java.py codellama7b-baseline $1 > logs/java-codellama7b-baseline.log
python3 calc_java.py codellama7b-stdft $1 > logs/java-codellama7b-stdft.log
python3 calc_java.py codellama7b-morepair $1 > logs/java-codellama7b-morepair.log

python3 calc_cpp.py starchat-baseline $1 > logs/cpp-starchat-baseline.log
python3 calc_cpp.py starchat-stdft $1 > logs/cpp-starchat-stdft.log
python3 calc_cpp.py starchat-morepair $1 > logs/cpp-starchat-morepair.log
python3 calc_java.py starchat-baseline $1 > logs/java-starchat-baseline.log
python3 calc_java.py starchat-stdft $1 > logs/java-starchat-stdft.log
python3 calc_java.py starchat-morepair $1 > logs/java-starchat-morepair.log

python3 calc_cpp.py mistral-baseline $1 > logs/cpp-mistral-baseline.log
python3 calc_cpp.py mistral-stdft $1 > logs/cpp-mistral-stdft.log
python3 calc_cpp.py mistral-morepair $1 > logs/cpp-mistral-morepair.log
python3 calc_java.py mistral-baseline $1 > logs/java-mistral-baseline.log
python3 calc_java.py mistral-stdft $1 > logs/java-mistral-stdft.log
python3 calc_java.py mistral-morepair $1 > logs/java-mistral-morepair.log

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 1 "$log_file"
done
