rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
rm -rf models/* || mkdir models
if [[ $3 != *"baseline" ]]; then
    python3 MOTrain.py $1 data/trainset/$2.json $3 $4
fi
python3 inference_cpp.py $3
python3 inference_java.py $3
