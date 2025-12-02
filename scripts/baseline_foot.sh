#!/bin/bash
# Foot Baseline 测试 (3/6/9 views)
# 用法: ./scripts/baseline_foot.sh <GPU>

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new
cd /home/qyhu/Documents/r2_ours/r2_gaussian

GPU=${1:-0}
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_DIR="data/369"
OUTPUT_BASE="output/baseline_${TIMESTAMP}"
ORGAN="foot"
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000 --gaussiansN 1"

echo "=== Foot Baseline (GPU $GPU) ==="
mkdir -p "$OUTPUT_BASE"

for views in 3 6 9; do
    DATA_FILE="${DATA_DIR}/${ORGAN}_50_${views}views.pickle"
    OUTPUT_DIR="${OUTPUT_BASE}/${ORGAN}_${views}views"

    echo "[${ORGAN}] ${views} views 开始..."
    mkdir -p "$OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s "$DATA_FILE" \
        -m "$OUTPUT_DIR" \
        $COMMON_FLAGS \
        2>&1 | tee "${OUTPUT_DIR}/training.log"

    echo "[${ORGAN}] ${views} views 完成"
done

echo "=== Foot 完成 ==="
