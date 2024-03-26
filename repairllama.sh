rm tmp/* || mkdir tmp
rm logs/* || mkdir logs

git clone git@github.com:ASSERT-KTH/repairllama.git
python3 inference_repairllama_cpp.py
python3 inference_repairllama_java.py

python3 calc_cpp.py repairllama rejudge > logs/cpp-repairllama.log
python3 calc_java.py repairllama rejudge > logs/java-repairllama.log

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 1 "$log_file"
done
