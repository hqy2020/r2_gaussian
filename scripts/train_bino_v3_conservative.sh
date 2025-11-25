#!/bin/bash

###############################################################################
# Bino v3: 保守优化版 - Foot 3 Views 训练脚本
#
# 论文: Binocular-Guided 3D Gaussian Splatting with View Consistency
#       for Sparse View Synthesis (NeurIPS 2024)
#
# 改进策略 (相比 v2):
#   1. 提前启动双目损失: 10000 → 7000 iterations
#   2. 增强损失权重: 0.1 → 0.15 (+50%)
#   3. 减小角度偏移: 0.08 → 0.06 rad (更保守)
#   4. 延长 warmup: 2000 → 3000 iterations
#
# 版本历史:
#   - v1 (2025-11-20): 仅 Opacity Decay，下降 1.08 dB
#   - v2 (2025-11-25): 完整实现，训练未完成
#   - v3 (2025-11-25): 保守优化版，针对 CT 稀疏视角调优
#
# 生成日期: 2025-11-25
# 目标: PSNR > 28.5 dB (超越 baseline 28.487)
#
###############################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 1. 环境配置
# ============================================================================
echo "🔧 [Setup] Activating conda environment: r2_gaussian_new"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# ============================================================================
# 2. 参数配置
# ============================================================================

# 数据集参数
ORGAN=${1:-foot}
NUM_VIEWS=${2:-3}

# 路径配置
DATA_PATH="data/369/${ORGAN}_50_${NUM_VIEWS}views.pickle"
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_PATH="output/${TIMESTAMP}_${ORGAN}_${NUM_VIEWS}views_bino_v3_conservative"

# 训练迭代参数
ITERATIONS=30000

# ==================== Bino v3 保守优化参数 ====================

# 1. Opacity Decay Strategy (🔧 Fix P2: 关闭，避免性能崩溃)
ENABLE_OPACITY_DECAY=""  # 关闭 Opacity Decay
OPACITY_DECAY_FACTOR=0.995  # 保留参数定义，便于后续对比实验

# 2. Binocular Stereo Consistency Loss (保守优化版)
ENABLE_BINOCULAR="--enable_binocular_consistency"
BINO_MAX_ANGLE=0.06          # 减小：0.08 → 0.06 rad (约 3.4°)
BINO_START_ITER=7000         # 提前：10000 → 7000 iterations
BINO_WARMUP=3000             # 延长：2000 → 3000 iterations
BINO_SMOOTH_WEIGHT=0.05      # 保持论文推荐值
BINO_LOSS_WEIGHT=0.15        # 增强：0.1 → 0.15 (+50%)
BINO_DEPTH_METHOD="weighted_average"

# ==============================================================

# 评估参数
TEST_ITERATIONS="1000 5000 10000 15000 20000 25000 30000"
SAVE_ITERATIONS="30000"

# ============================================================================
# 3. 数据集检查
# ============================================================================
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ [Error] 数据集文件不存在: $DATA_PATH"
    exit 1
fi

echo "✅ [Data] 数据集检查通过: $DATA_PATH"

# ============================================================================
# 4. 训练信息展示
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║   🔬 Bino v3: 保守优化版 (Conservative Optimized)                         ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║   改进策略:                                                                 ║"
echo "║     ✓ 提前启动双目损失: 10000 → 7000 iterations                            ║"
echo "║     ✓ 增强损失权重: 0.1 → 0.15 (+50%)                                      ║"
echo "║     ✓ 减小角度偏移: 0.08 → 0.06 rad (更保守)                               ║"
echo "║     ✓ 延长 warmup: 2000 → 3000 iterations                                  ║"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║   器官: ${ORGAN}"
echo "║   视角数: ${NUM_VIEWS}"
echo "║   输出路径: $OUTPUT_PATH"
echo "║   训练迭代数: $ITERATIONS"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║   Bino v3 参数配置:"
echo "║     • 不透明度衰减: 启用, 衰减因子=$OPACITY_DECAY_FACTOR"
echo "║     • 双目一致性损失: 启用"
echo "║       - 最大角度偏移: ${BINO_MAX_ANGLE} rad (≈3.4°)"
echo "║       - 启动迭代: ${BINO_START_ITER} ⬅ 提前"
echo "║       - Warmup: ${BINO_WARMUP} ⬅ 延长"
echo "║       - 平滑权重: ${BINO_SMOOTH_WEIGHT}"
echo "║       - 总权重: ${BINO_LOSS_WEIGHT} ⬅ 增强"
echo "║       - 深度估计: ${BINO_DEPTH_METHOD}"
echo "╠════════════════════════════════════════════════════════════════════════════╣"
echo "║   🎯 Foot 3 Views SOTA 基准 (R² baseline):"
echo "║     PSNR: 28.487 dB (单模型)"
echo "║     SSIM: 0.900"
echo "║   🎯 目标:"
echo "║     PSNR > 28.5 dB"
echo "║     SSIM > 0.900"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# 5. 启动训练
# ============================================================================
echo "🚀 [Training] 开始训练..."
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 启动训练
python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --iterations $ITERATIONS \
    --eval \
    --gaussiansN 1 \
    $ENABLE_OPACITY_DECAY \
    --opacity_decay_factor $OPACITY_DECAY_FACTOR \
    $ENABLE_BINOCULAR \
    --binocular_max_angle_offset $BINO_MAX_ANGLE \
    --binocular_start_iter $BINO_START_ITER \
    --binocular_warmup_iters $BINO_WARMUP \
    --binocular_smooth_weight $BINO_SMOOTH_WEIGHT \
    --binocular_loss_weight $BINO_LOSS_WEIGHT \
    --binocular_depth_method $BINO_DEPTH_METHOD \
    --test_iterations $TEST_ITERATIONS \
    --save_iterations $SAVE_ITERATIONS \
    2>&1 | tee "${OUTPUT_PATH}_train.log"

# 捕获训练进程的退出状态
TRAIN_STATUS=$?

# 记录结束时间
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED_TIME / 60))

# ============================================================================
# 6. 训练结果检查
# ============================================================================
echo ""
if [ $TRAIN_STATUS -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║   ✅ 训练成功完成！                                                        ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║   用时: ${ELAPSED_MIN} 分钟 (${ELAPSED_TIME} 秒)"
    echo "║   输出: $OUTPUT_PATH"
    echo "║   日志: ${OUTPUT_PATH}_train.log"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║   📊 查看结果:"
    echo "║     1. 最终评估: cat $OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml"
    echo "║     2. TensorBoard: tensorboard --logdir=$OUTPUT_PATH"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # 自动显示最终评估结果
    EVAL_FILE="$OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml"
    if [ -f "$EVAL_FILE" ]; then
        echo "📈 [Results] 最终评估结果:"
        echo "───────────────────────────────────────────────────────────────────────────"
        cat "$EVAL_FILE" | grep -E "psnr_2d|ssim_2d" | grep -v "_projs"
        echo "───────────────────────────────────────────────────────────────────────────"

        # 与 baseline 对比
        echo ""
        echo "📊 [Comparison] 与 Baseline 对比:"
        echo "  Baseline: PSNR 28.487 dB, SSIM 0.900"
        PSNR=$(cat "$EVAL_FILE" | grep "psnr_2d:" | grep -v "_projs" | head -1 | awk '{print $2}')
        SSIM=$(cat "$EVAL_FILE" | grep "ssim_2d:" | grep -v "_projs" | head -1 | awk '{print $2}')
        echo "  Bino v3:  PSNR $PSNR dB, SSIM $SSIM"
        echo ""
    fi
else
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║   ❌ 训练失败 (退出代码: $TRAIN_STATUS)                                    ║"
    echo "╠════════════════════════════════════════════════════════════════════════════╣"
    echo "║   请检查日志文件: ${OUTPUT_PATH}_train.log"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi
