#!/bin/bash

###############################################################################
# CoR-GS 完整版 - Foot 3 views 训练脚本
#
# 生成日期: 2025-11-26
# 目标: PSNR > 29.0 dB (超越 baseline 28.487 dB)
# 数据集: Foot 3 views (稀疏视角医学 CT 重建)
#
# 实现的创新点:
#   - Stage 2: Co-pruning (修剪两个模型间不匹配的 Gaussians)
#   - Stage 3: Pseudo-view Co-regularization (虚拟视角协同正则化)
#
# 使用方法:
#   bash scripts/train_corgs_foot3_30k.sh
#
# 参考论文:
#   CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization (ECCV 2024)
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
OUTPUT_PATH="output/${TIMESTAMP}_foot_3views_corgs_full"

# ========== 训练配置 (论文推荐值) ==========
ITERATIONS=30000
DENSIFY_FROM_ITER=500
DENSIFY_UNTIL_ITER=15000
DENSIFICATION_INTERVAL=100

# ========== 双模型配置 ==========
GAUSSIANS_N=2  # 双模型

# ========== Co-pruning (Stage 2) 参数 ==========
ENABLE_COPRUNE=true
COPRUNE_THRESHOLD=0.1        # 距离阈值 τ (归一化 CT 场景)
COPRUNE_INTERVAL=5           # 每 5 个 densification 循环执行一次
COPRUNE_START_ITER=500       # 开始迭代
COPRUNE_END_ITER=15000       # 结束迭代 (与 densify_until_iter 同步)
COPRUNE_MIN_POINTS=5000      # 最小点数保护

# ========== Pseudo-view Co-regularization (Stage 3) 参数 ==========
ENABLE_CORGS=true
CORGS_PSEUDO_START_ITER=2000  # 论文推荐 2000
CORGS_PSEUDO_WEIGHT=1.0       # 权重 λ_p (论文默认 1.0)
CORGS_POOL_SIZE=1000          # Pseudo-view 池大小
CORGS_RAMP_ITERS=500          # Warm-up 迭代数

# 检查数据集是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "[Error] Dataset not found: $DATA_PATH"
    echo "  Please ensure the dataset path is correct."
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 打印配置信息
echo ""
echo "=========================================="
echo " CoR-GS Full Implementation"
echo " (Stage 2: Co-pruning + Stage 3: Pseudo-view)"
echo "=========================================="
echo ""
echo "[Dataset]"
echo "  Path: $DATA_PATH"
echo "  Output: $OUTPUT_PATH"
echo ""
echo "[Training Config]"
echo "  Iterations: $ITERATIONS"
echo "  Densify: [$DENSIFY_FROM_ITER, $DENSIFY_UNTIL_ITER] @ interval $DENSIFICATION_INTERVAL"
echo "  Models: $GAUSSIANS_N (dual model)"
echo ""
echo "[Stage 2: Co-pruning]"
echo "  Enabled: $ENABLE_COPRUNE"
echo "  Threshold: $COPRUNE_THRESHOLD"
echo "  Interval: every $COPRUNE_INTERVAL densification cycles"
echo "  Range: [$COPRUNE_START_ITER, $COPRUNE_END_ITER]"
echo ""
echo "[Stage 3: Pseudo-view Co-regularization]"
echo "  Enabled: $ENABLE_CORGS"
echo "  Start iter: $CORGS_PSEUDO_START_ITER"
echo "  Weight: $CORGS_PSEUDO_WEIGHT"
echo "  Pool size: $CORGS_POOL_SIZE"
echo ""
echo "=========================================="
echo ""

# 保存配置到文件
cat > "$OUTPUT_PATH/config.txt" << EOF
CoR-GS Full Training Configuration
==================================
Timestamp: $TIMESTAMP
Dataset: $DATA_PATH

Training:
  iterations: $ITERATIONS
  densify_from_iter: $DENSIFY_FROM_ITER
  densify_until_iter: $DENSIFY_UNTIL_ITER
  densification_interval: $DENSIFICATION_INTERVAL
  gaussiansN: $GAUSSIANS_N

Stage 2 (Co-pruning):
  enable_coprune: $ENABLE_COPRUNE
  coprune_threshold: $COPRUNE_THRESHOLD
  coprune_interval: $COPRUNE_INTERVAL
  coprune_start_iter: $COPRUNE_START_ITER
  coprune_end_iter: $COPRUNE_END_ITER
  coprune_min_points: $COPRUNE_MIN_POINTS

Stage 3 (Pseudo-view Co-regularization):
  enable_corgs: $ENABLE_CORGS
  corgs_pseudo_start_iter: $CORGS_PSEUDO_START_ITER
  corgs_pseudo_weight: $CORGS_PSEUDO_WEIGHT
  corgs_pool_size: $CORGS_POOL_SIZE
  corgs_ramp_iters: $CORGS_RAMP_ITERS

Baseline Reference:
  PSNR: 28.487 dB
  SSIM: 0.9005
EOF

# 启动训练
echo "[Training] Starting CoR-GS training..."
echo ""

python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
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
    --corgs_pool_size $CORGS_POOL_SIZE \
    --corgs_ramp_iters $CORGS_RAMP_ITERS \
    --eval \
    --test_iterations 1 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 \
    2>&1 | tee "$OUTPUT_PATH/train.log"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo " Training Completed Successfully!"
    echo "=========================================="
    echo ""
    echo "[Results Location]"
    echo "  Output: $OUTPUT_PATH"
    echo "  Log: $OUTPUT_PATH/train.log"
    echo ""
    echo "[Next Steps]"
    echo "  1. View TensorBoard:"
    echo "     tensorboard --logdir=$OUTPUT_PATH --port 6006"
    echo ""
    echo "  2. Check final results:"
    echo "     cat $OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml"
    echo ""
    echo "  3. Compare with baseline (PSNR 28.487 dB, SSIM 0.9005):"
    echo "     python scripts/compare_results.py $OUTPUT_PATH"
    echo ""
else
    echo ""
    echo "[Error] Training failed. Check log: $OUTPUT_PATH/train.log"
    exit 1
fi
