rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
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
