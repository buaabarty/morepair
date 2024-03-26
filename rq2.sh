rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
echo "evaluating codellama7b-baseline on EvalRepair-C++"
python3 calc_cpp.py codellama7b-baseline $1 > logs/cpp-codellama7b-baseline.log
echo "evaluating codellama7b-stdft on EvalRepair-C++"
python3 calc_cpp.py codellama7b-stdft $1 > logs/cpp-codellama7b-stdft.log
echo "evaluating codellama7b-morepair on EvalRepair-C++"
python3 calc_cpp.py codellama7b-morepair $1 > logs/cpp-codellama7b-morepair.log
echo "evaluating codellama7b-baseline on EvalRepair-Java"
python3 calc_java.py codellama7b-baseline $1 > logs/java-codellama7b-baseline.log
echo "evaluating codellama7b-stdft on EvalRepair-Java"
python3 calc_java.py codellama7b-stdft $1 > logs/java-codellama7b-stdft.log
echo "evaluating codellama7b-morepair on EvalRepair-Java"
python3 calc_java.py codellama7b-morepair $1 > logs/java-codellama7b-morepair.log

echo "evaluating starchat-baseline on EvalRepair-C++"
python3 calc_cpp.py starchat-baseline $1 > logs/cpp-starchat-baseline.log
echo "evaluating starchat-stdft on EvalRepair-C++"
python3 calc_cpp.py starchat-stdft $1 > logs/cpp-starchat-stdft.log
echo "evaluating starchat-morepair on EvalRepair-C++"
python3 calc_cpp.py starchat-morepair $1 > logs/cpp-starchat-morepair.log
echo "evaluating starchat-baseline on EvalRepair-Java"
python3 calc_java.py starchat-baseline $1 > logs/java-starchat-baseline.log
echo "evaluating starchat-stdft on EvalRepair-Java"
python3 calc_java.py starchat-stdft $1 > logs/java-starchat-stdft.log
echo "evaluating starchat-morepair on EvalRepair-Java"
python3 calc_java.py starchat-morepair $1 > logs/java-starchat-morepair.log

echo "evaluating mistral-baseline on EvalRepair-C++"
python3 calc_cpp.py mistral-baseline $1 > logs/cpp-mistral-baseline.log
echo "evaluating mistral-stdft on EvalRepair-C++"
python3 calc_cpp.py mistral-stdft $1 > logs/cpp-mistral-stdft.log
echo "evaluating mistral-morepair on EvalRepair-C++"
python3 calc_cpp.py mistral-morepair $1 > logs/cpp-mistral-morepair.log
echo "evaluating mistral-baseline on EvalRepair-Java"
python3 calc_java.py mistral-baseline $1 > logs/java-mistral-baseline.log
echo "evaluating mistral-stdft on EvalRepair-Java"
python3 calc_java.py mistral-stdft $1 > logs/java-mistral-stdft.log
echo "evaluating mistral-morepair on EvalRepair-Java"
python3 calc_java.py mistral-morepair $1 > logs/java-mistral-morepair.log

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 1 "$log_file"
done
