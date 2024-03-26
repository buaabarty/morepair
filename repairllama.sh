rm tmp/* || mkdir tmp
rm logs/* || mkdir logs

git clone git@github.com:ASSERT-KTH/repairllama.git
python3 inference_repairllama_cpp.py
python3 inference_repairllama_java.py
