rm tmp/* || mkdir tmp
rm logs/* || mkdir logs
echo "evaluating codellama13b-baseline on defects4j"
python3 test_d4j.py -m baseline -n 10 > logs/d4j-codellama13b-baseline.log
echo "evaluating codellama13b-stdft on defects4j"
python3 test_d4j.py -m stdft -n 10 > logs/d4j-codellama13b-stdft.log
echo "evaluating codellama13b-morepair on defects4j"
python3 test_d4j.py -m morepair -n 10 > logs/d4j-codellama13b-morepair.log

cd swebench_result
echo "evaluating codellama13b-baseline on swebench"
python3 -m swebench.harness.run_evaluation --dataset SWE-Bench --predictions_path patches/baseline.jsonl --max_worker 16 --run_id baseline --timeout 900 > logs/swe-codellama13b-baseline.log
echo "evaluating codellama13b-stdft on swebench"
python3 -m swebench.harness.run_evaluation --dataset SWE-Bench --predictions_path patches/stdft.jsonl --max_worker 16 --run_id stdft --timeout 900 > logs/swe-codellama13b-stdft.log
echo "evaluating codellama13b-morepair on swebench"
python3 -m swebench.harness.run_evaluation --dataset SWE-Bench --predictions_path patches/morepair.jsonl --max_worker 16 --run_id morepair --timeout 900 > logs/swe-codellama13b-morepair.log
cd ..

for log_file in logs/*.log; do
    echo -n "$(basename "$log_file") : "
    tail -n 5 "$log_file"
done
