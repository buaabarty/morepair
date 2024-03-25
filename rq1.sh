rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
if [ "$2" = "train" ]; then
    mkdir models || true
    python3 MOTrain.py CodeLlama-13b-Instruct-hf data/trainset/llama_llm.json models/codellama13b-stdft 0
    python3 MOTrain.py CodeLlama-13b-Instruct-hf data/trainset/llama_llm.json models/codellama13b-morepair 1
    python3 inference_cpp.py codellama13b-baseline
    python3 inference_cpp.py codellama13b-stdft
    python3 inference_cpp.py codellama13b-morepair
    python3 inference_java.py codellama13b-baseline
    python3 inference_java.py codellama13b-stdft
    python3 inference_java.py codellama13b-morepair
fi
python3 calc_cpp.py codellama13b-baseline $1 > logs/cpp-codellama13b-baseline.log
python3 calc_cpp.py codellama13b-stdft $1 > logs/cpp-codellama13b-stdft.log
python3 calc_cpp.py codellama13b-morepair $1 > logs/cpp-codellama13b-morepair.log
python3 calc_java.py codellama13b-baseline $1 > logs/java-codellama13b-baseline.log
python3 calc_java.py codellama13b-stdft $1 > logs/java-codellama13b-stdft.log
python3 calc_java.py codellama13b-morepair $1 > logs/java-codellama13b-morepair.log

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 1 "$log_file"
done
