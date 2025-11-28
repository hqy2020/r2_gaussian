#!/bin/bash
# Foot 3 视角 Baseline 对比实验
# 比较三个数据集: 369, 369_new, density-369
# 用法: ./scripts/train_foot3_baseline_compare.sh <GPU>

set -e

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

GPU=${1:-0}
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# 公共参数
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000 --gaussiansN 1"

echo "=============================================="
echo "Foot 3 视角 Baseline 对比实验"
echo "时间戳: $TIMESTAMP"
echo "GPU: $GPU"
echo "=============================================="

# 实验 1: 369 (原始数据集)
echo ""
echo "[1/3] 训练 369 数据集..."
DATA_369="data/369/foot_50_3views.pickle"
OUTPUT_369="output/baseline_compare/${TIMESTAMP}_foot_3views_369"
mkdir -p "$OUTPUT_369"

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -s "$DATA_369" \
    -m "$OUTPUT_369" \
    $COMMON_FLAGS \
    2>&1 | tee "${OUTPUT_369}/training.log"

echo "完成: 369 数据集"

# 实验 2: 369_new (新数据集)
echo ""
echo "[2/3] 训练 369_new 数据集..."
DATA_369_NEW="data/369_new/foot_50_3views.pickle"
OUTPUT_369_NEW="output/baseline_compare/${TIMESTAMP}_foot_3views_369_new"
mkdir -p "$OUTPUT_369_NEW"

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -s "$DATA_369_NEW" \
    -m "$OUTPUT_369_NEW" \
    $COMMON_FLAGS \
    2>&1 | tee "${OUTPUT_369_NEW}/training.log"

echo "完成: 369_new 数据集"

# 实验 3: density-369 (密度初始化数据集)
echo ""
echo "[3/3] 训练 density-369 数据集..."
DATA_DENSITY="data/density-369/foot_50_3views.pickle"
OUTPUT_DENSITY="output/baseline_compare/${TIMESTAMP}_foot_3views_density_369"
mkdir -p "$OUTPUT_DENSITY"

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -s "$DATA_DENSITY" \
    -m "$OUTPUT_DENSITY" \
    $COMMON_FLAGS \
    2>&1 | tee "${OUTPUT_DENSITY}/training.log"

echo "完成: density-369 数据集"

echo ""
echo "=============================================="
echo "所有实验完成!"
echo "结果目录: output/baseline_compare/${TIMESTAMP}_*"
echo "=============================================="

# 打印结果汇总
echo ""
echo "结果汇总:"
for dir in "$OUTPUT_369" "$OUTPUT_369_NEW" "$OUTPUT_DENSITY"; do
    if [ -f "${dir}/results.json" ]; then
        echo "$(basename $dir):"
        python -c "import json; d=json.load(open('${dir}/results.json')); print(f\"  PSNR: {d.get('psnr', 'N/A'):.4f}, SSIM: {d.get('ssim', 'N/A'):.4f}\")" 2>/dev/null || echo "  结果文件解析失败"
    fi
done
