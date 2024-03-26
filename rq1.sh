rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
echo "evaluating codellama13b-baseline on EvalRepair-C++"
python3 calc_cpp.py codellama13b-baseline $1 > logs/cpp-codellama13b-baseline.log
echo "evaluating codellama13b-stdft on EvalRepair-C++"
python3 calc_cpp.py codellama13b-stdft $1 > logs/cpp-codellama13b-stdft.log
echo "evaluating codellama13b-morepair on EvalRepair-C++"
python3 calc_cpp.py codellama13b-morepair $1 > logs/cpp-codellama13b-morepair.log
echo "evaluating codellama13b-baseline on EvalRepair-Java"
python3 calc_java.py codellama13b-baseline $1 > logs/java-codellama13b-baseline.log
echo "evaluating codellama13b-stdft on EvalRepair-Java"
python3 calc_java.py codellama13b-stdft $1 > logs/java-codellama13b-stdft.log
echo "evaluating codellama13b-morepair on EvalRepair-Java"
python3 calc_java.py codellama13b-morepair $1 > logs/java-codellama13b-morepair.log

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 1 "$log_file"
done
