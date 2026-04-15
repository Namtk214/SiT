#!/bin/bash
# Re-run only the failed tasks (3,4,5,6,7)
# Task 9 also needs to run fresh

cd /home/thanhnamngo26/SiT

echo ">>> TASK 3 (re-run)" 
python3 -u experiments/task3_layer_timestep_map.py 2>&1 | tee experiments/results/task3/task3_log.txt

echo ">>> TASK 4 (re-run)"
python3 -u experiments/task4_spatial_metrics.py 2>&1 | tee experiments/results/task4/task4_log.txt

echo ">>> TASK 5 (re-run)"
python3 -u experiments/task5_probes.py 2>&1 | tee experiments/results/task5/task5_log.txt

echo ">>> TASK 6 (re-run)"
python3 -u experiments/task6_ablation.py 2>&1 | tee experiments/results/task6/task6_log.txt

echo ">>> TASK 7 (re-run)"
python3 -u experiments/task7_cosine_maps.py 2>&1 | tee experiments/results/task7/task7_log.txt

echo ">>> TASK 9"
python3 -u experiments/task9_stress_test.py 2>&1 | tee experiments/results/task9/task9_log.txt

echo ">>> TASK 10"
python3 -u experiments/task10_wavelet.py 2>&1 | tee experiments/results/task10/task10_log.txt

echo "ALL FAILED TASKS RE-RUN COMPLETE"
