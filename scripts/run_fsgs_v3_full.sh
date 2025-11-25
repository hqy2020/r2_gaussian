#!/bin/bash
# FSGS v3 完整实验 - 结合参数优化 + 伪视角深度监督
#
# 优化策略:
# 1. 参数优化 (Plan A): 调整 proximity, TV, densify 参数
# 2. 深度监督 (Plan B): 启用 MiDaS 伪视角深度监督
#
# 针对 bad cases (视角 42-44) 的改进:
# - 提高 lambda_tv: 增强几何平滑性
# - 降低 proximity_threshold: 更严格的约束
# - 启用深度监督: 增强几何一致性

# 设置 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 时间戳
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_PATH="data/369/foot_50_3views.pickle"

echo "==========================================="
echo "FSGS v3 完整实验 - 参数优化 + 深度监督"
echo "时间戳: $TIMESTAMP"
echo "==========================================="

# 创建输出目录
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_fsgs_v3_full"
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
    --depth_pseudo_weight 0.05 \
    --start_sample_pseudo 2000 \
    --end_sample_pseudo 15000 \
    --sample_pseudo_interval 5 \
    \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "==========================================="
echo "FSGS v3 完整实验完成"
echo "输出目录: $OUTPUT_DIR"
echo "==========================================="
