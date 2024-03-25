rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
mkdir models || true
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
