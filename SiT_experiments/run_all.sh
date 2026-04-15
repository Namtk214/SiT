#!/bin/bash
# Master runner for all experiment tasks.
# Task 0 must complete first, then remaining tasks run sequentially.
# Usage: bash experiments/run_all.sh

set -e

cd /home/thanhnamngo26/SiT

echo "=================================================="
echo "SiT Experimental Protocol - Full Execution"
echo "=================================================="

# Check Task 0 completion
if [ ! -f experiments/results/task0/task0_metadata.csv ]; then
    echo "ERROR: Task 0 not complete. Run task0_extract.py first."
    exit 1
fi

echo "Task 0 data found. Running remaining tasks..."

# Task 1
echo ""
echo ">>> TASK 1: Basic Statistics"
python3 -u experiments/task1_statistics.py 2>&1 | tee experiments/results/task1/task1_log.txt

# Task 2
echo ""
echo ">>> TASK 2: Similarity Heatmaps"
python3 -u experiments/task2_similarity.py 2>&1 | tee experiments/results/task2/task2_log.txt

# Task 3
echo ""
echo ">>> TASK 3: Layer-Timestep Map"
python3 -u experiments/task3_layer_timestep_map.py 2>&1 | tee experiments/results/task3/task3_log.txt

# Task 4
echo ""
echo ">>> TASK 4: Spatial Structure Metrics"
python3 -u experiments/task4_spatial_metrics.py 2>&1 | tee experiments/results/task4/task4_log.txt

# Task 5
echo ""
echo ">>> TASK 5: Layer-ID / Timestep-ID Probes"
python3 -u experiments/task5_probes.py 2>&1 | tee experiments/results/task5/task5_log.txt

# Task 6
echo ""
echo ">>> TASK 6: Causal Ablation"
python3 -u experiments/task6_ablation.py 2>&1 | tee experiments/results/task6/task6_log.txt

# Task 7
echo ""
echo ">>> TASK 7: Patch-wise Cosine Maps"
python3 -u experiments/task7_cosine_maps.py 2>&1 | tee experiments/results/task7/task7_log.txt

# Task 8
echo ""
echo ">>> TASK 8: PCA / t-SNE / UMAP Visualization"
python3 -u experiments/task8_visualization.py 2>&1 | tee experiments/results/task8/task8_log.txt

# Task 9 (requires re-running model, so runs last)
echo ""
echo ">>> TASK 9: Spatial Stress Test"
python3 -u experiments/task9_stress_test.py 2>&1 | tee experiments/results/task9/task9_log.txt

# Task 10/11
echo ""
echo ">>> TASK 10/11: Wavelet Decomposition"
python3 -u experiments/task10_wavelet.py 2>&1 | tee experiments/results/task10/task10_log.txt

echo ""
echo "=================================================="
echo "ALL TASKS COMPLETE"
echo "=================================================="
