#!/bin/bash
# FSGS v3 参数优化实验
# 策略：针对 bad cases (视角 42-44)，调整 proximity 和 TV 参数

# 设置 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 时间戳
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_PATH="data/369/foot_50_3views.pickle"

echo "=========================================="
echo "FSGS v3 参数优化实验"
echo "时间戳: $TIMESTAMP"
echo "=========================================="

# 实验配置说明：
# 1. 提高 lambda_tv (0.05 -> 0.08): 增强几何平滑性，减少过拟合
# 2. 降低 proximity_threshold (6.0 -> 5.0): 更严格的 proximity 约束，避免过度密集化
# 3. 减少 proximity_k_neighbors (3 -> 5): 更多邻居约束，增强稳定性
# 4. 提高 densify_grad_threshold: 减少过度密集化

python train.py \
    -s "$DATA_PATH" \
    -m "output/${TIMESTAMP}_foot_3views_fsgs_v3_param_opt" \
    --eval \
    --iterations 30000 \
    --test_iterations 5000 10000 20000 30000 \
    --save_iterations 30000 \
    --gaussiansN 1 \
    \
    --enable_fsgs_proximity \
    --proximity_threshold 5.0 \
    --proximity_k_neighbors 5 \
    --enable_medical_constraints \
    --proximity_organ_type foot \
    \
    --lambda_tv 0.08 \
    --lambda_dssim 0.25 \
    --densify_grad_threshold 3e-4 \
    --densify_until_iter 12000 \
    \
    2>&1 | tee "output/${TIMESTAMP}_foot_3views_fsgs_v3_param_opt/train.log"

echo "=========================================="
echo "FSGS v3 参数优化实验完成"
echo "输出目录: output/${TIMESTAMP}_foot_3views_fsgs_v3_param_opt"
echo "=========================================="
