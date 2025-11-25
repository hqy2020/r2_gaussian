#!/bin/bash
# FSGS v3 深度监督实验 - 仅启用伪视角深度监督
#
# 目的: 单独测试深度监督的效果，作为消融实验
# 使用与最佳 FSGS 实验相同的参数，仅添加深度监督

# 设置 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 时间戳
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_PATH="data/369/foot_50_3views.pickle"

echo "==========================================="
echo "FSGS v3 深度监督实验"
echo "时间戳: $TIMESTAMP"
echo "==========================================="

# 创建输出目录
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_fsgs_v3_depth"
mkdir -p "$OUTPUT_DIR"

python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_DIR" \
    --eval \
    --iterations 30000 \
    --test_iterations 5000 10000 20000 30000 \
    --save_iterations 30000 \
    --gaussiansN 1 \
    \
    --enable_fsgs_proximity \
    --proximity_threshold 6.0 \
    --proximity_k_neighbors 3 \
    --enable_medical_constraints \
    --proximity_organ_type foot \
    \
    --lambda_tv 0.05 \
    --lambda_dssim 0.25 \
    \
    --depth_pseudo_weight 0.05 \
    --start_sample_pseudo 2000 \
    --end_sample_pseudo 15000 \
    --sample_pseudo_interval 5 \
    \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "==========================================="
echo "FSGS v3 深度监督实验完成"
echo "输出目录: $OUTPUT_DIR"
echo "==========================================="
