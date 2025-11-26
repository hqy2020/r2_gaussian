#!/bin/bash

###############################################################################
# CoR-GS v2 医学适配版 - Foot 3 views 训练脚本
#
# 生成日期: 2025-11-26
# 目标: PSNR > 28.6 dB (超越 baseline 28.487 dB)
# 数据集: Foot 3 views (稀疏视角医学 CT 重建)
#
# v2 核心改进（解决双模型同步问题）:
#   1. 双模型差异化初始化 (init_noise_std=0.02)
#   2. 更严格的 Co-pruning 阈值 (threshold=0.02)
#   3. 增大伪视角扰动 (pseudo_noise_std=0.05)
#   4. 更频繁的 Co-pruning (interval=3)
#
# 问题诊断 (v1 失败原因):
#   - 双模型完全同步 (loss_m0 == loss_m1)
#   - Co-pruning 几乎无效 (pruned 0)
#   - threshold=0.1 对于 CT 场景太宽松
#
# 使用方法:
#   bash scripts/train_corgs_v2_foot3.sh
###############################################################################

set -e  # 遇到错误立即退出

# 激活 conda 环境
echo "=========================================="
echo "[Setup] Activating conda environment: r2_gaussian_new"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 生成时间戳
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# ========== 数据集配置 ==========
DATA_PATH="data/369/foot_50_3views.pickle"
OUTPUT_PATH="output/${TIMESTAMP}_foot_3views_corgs_v2"

# ========== 训练配置 ==========
ITERATIONS=30000
DENSIFY_FROM_ITER=500
DENSIFY_UNTIL_ITER=15000
DENSIFICATION_INTERVAL=100

# ========== 双模型配置 ==========
GAUSSIANS_N=2  # 双模型

# ========== 🔥 CoR-GS v2 核心改进参数 ==========
# 1. 差异化初始化 - 让双模型从不同起点开始
CORGS_INIT_NOISE_STD=0.02  # 第二个模型的位置扰动 (约 ±0.4mm)

# 2. Co-pruning 参数 - 更严格的阈值让 pruning 生效
ENABLE_COPRUNE=true
COPRUNE_THRESHOLD=0.02     # 🔥 从 0.1 降到 0.02 (更严格，让 co-pruning 生效)
COPRUNE_INTERVAL=3         # 🔥 从 5 降到 3 (更频繁地修剪)
COPRUNE_START_ITER=1000    # 稍晚启动，让模型先分化
COPRUNE_END_ITER=15000
COPRUNE_MIN_POINTS=5000

# 3. Pseudo-view Co-regularization 参数 - 增大差异性
ENABLE_CORGS=true
CORGS_PSEUDO_START_ITER=2000
CORGS_PSEUDO_WEIGHT=1.0
CORGS_PSEUDO_NOISE_STD=0.05  # 🔥 从 0.02 增大到 0.05 (更大的伪视角扰动)
CORGS_POOL_SIZE=1000
CORGS_RAMP_ITERS=500

# 检查数据集是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "[Error] Dataset not found: $DATA_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 打印配置信息
echo ""
echo "=========================================="
echo " CoR-GS v2 医学适配版"
echo " (双模型差异化 + 严格 Co-pruning)"
echo "=========================================="
echo ""
echo "[v2 核心改进]"
echo "  1. 差异化初始化: init_noise_std=${CORGS_INIT_NOISE_STD}"
echo "  2. 严格 Co-pruning: threshold=${COPRUNE_THRESHOLD} (v1=0.1)"
echo "  3. 更频繁 Co-pruning: interval=${COPRUNE_INTERVAL} (v1=5)"
echo "  4. 更大伪视角扰动: pseudo_noise_std=${CORGS_PSEUDO_NOISE_STD} (v1=0.02)"
echo ""
echo "[Dataset]"
echo "  Path: $DATA_PATH"
echo "  Output: $OUTPUT_PATH"
echo ""
echo "[Training]"
echo "  Iterations: $ITERATIONS"
echo "  GaussiansN: $GAUSSIANS_N"
echo ""

# 启动训练
echo "=========================================="
echo "[Training] Starting CoR-GS v2..."
echo "=========================================="

python train.py \
    --source_path "$DATA_PATH" \
    --model_path "$OUTPUT_PATH" \
    --iterations $ITERATIONS \
    --densify_from_iter $DENSIFY_FROM_ITER \
    --densify_until_iter $DENSIFY_UNTIL_ITER \
    --densification_interval $DENSIFICATION_INTERVAL \
    --gaussiansN $GAUSSIANS_N \
    --enable_coprune \
    --coprune_threshold $COPRUNE_THRESHOLD \
    --coprune_interval $COPRUNE_INTERVAL \
    --coprune_start_iter $COPRUNE_START_ITER \
    --coprune_end_iter $COPRUNE_END_ITER \
    --coprune_min_points $COPRUNE_MIN_POINTS \
    --enable_corgs \
    --corgs_pseudo_start_iter $CORGS_PSEUDO_START_ITER \
    --corgs_pseudo_weight $CORGS_PSEUDO_WEIGHT \
    --corgs_pseudo_noise_std $CORGS_PSEUDO_NOISE_STD \
    --corgs_pool_size $CORGS_POOL_SIZE \
    --corgs_ramp_iters $CORGS_RAMP_ITERS \
    --corgs_init_noise_std $CORGS_INIT_NOISE_STD \
    2>&1 | tee "${OUTPUT_PATH}/train.log"

echo ""
echo "=========================================="
echo " Training Completed!"
echo "=========================================="
echo ""
echo "[Results Location]"
echo "  Output: $OUTPUT_PATH"
echo "  Log: ${OUTPUT_PATH}/train.log"
echo ""
echo "[Next Steps]"
echo "  1. Check final results:"
echo "     cat ${OUTPUT_PATH}/eval/iter_030000/eval2d_render_test.yml"
echo ""
echo "  2. Compare with baseline (PSNR 28.487 dB, SSIM 0.9005):"
echo "     python scripts/compare_results.py ${OUTPUT_PATH}"
