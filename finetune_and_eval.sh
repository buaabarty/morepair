rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
mkdir models || true
if [[ $3 == *"baseline" ]]; then
    python3 MOTrain.py $1 data/trainset/$2.json models/$3 $4
fi
python3 inference_cpp.py $3
python3 inference_java.py $3
python3 calc_cpp.py $3 rejudge > logs/cpp-$3.log
python3 calc_java.py $3 rejudge > logs/java-$3.log

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 1 "$log_file"
done
