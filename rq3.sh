rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
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
