#!/bin/bash

###############################################################################
# Bino (Binocular-Guided 3DGS) - Foot 3 Views 训练脚本
#
# 论文: Binocular-Guided 3D Gaussian Splatting with View Consistency
#       for Sparse View Synthesis (NeurIPS 2024)
#
# 核心创新: Opacity Decay Strategy (不透明度衰减策略)
#   - 在密化过程中应用指数衰减，自动过滤低梯度冗余Gaussians
#   - 衰减因子: 0.995 (论文推荐值)
#
# 生成日期: 2025-11-20
# 目标: 验证 Bino 方法在 Foot 3 views 数据集上的效果
#
# 使用方法:
#   bash scripts/train_bino_foot3.sh [器官名称] [视角数]
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

# 数据集参数（支持命令行参数覆盖）
ORGAN=${1:-foot}           # 器官名称 (foot, chest, head, abdomen, pancreas)
NUM_VIEWS=${2:-3}          # 视角数 (3, 6, 9)

# 路径配置
DATA_PATH="data/369/${ORGAN}_50_${NUM_VIEWS}views.pickle"
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_PATH="output/${TIMESTAMP}_${ORGAN}_${NUM_VIEWS}views_bino"

# 训练迭代参数
ITERATIONS=30000

# Bino 核心参数
ENABLE_OPACITY_DECAY="--enable_opacity_decay"     # 启用不透明度衰减
OPACITY_DECAY_FACTOR=0.995                        # 衰减因子（论文推荐值）

# 评估参数（修复之前的错误：移除末尾的 1）
TEST_ITERATIONS="1000 5000 10000 15000 20000 25000 30000"
SAVE_ITERATIONS="30000"

# ============================================================================
# 3. 数据集检查
# ============================================================================
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ [Error] 数据集文件不存在: $DATA_PATH"
    echo "   可用数据集列表:"
    ls -lh data/369/*.pickle 2>/dev/null || echo "   未找到任何数据集文件"
    exit 1
fi

echo ""
echo "✅ [Data] 数据集检查通过: $DATA_PATH"

# ============================================================================
# 4. 训练信息展示
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   🔬 Bino: Binocular-Guided 3D Gaussian Splatting                 ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║   论文: NeurIPS 2024                                               ║"
echo "║   核心方法: Opacity Decay Strategy (不透明度衰减)                 ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║   器官: ${ORGAN}"
echo "║   视角数: ${NUM_VIEWS}"
echo "║   数据集: $DATA_PATH"
echo "║   输出路径: $OUTPUT_PATH"
echo "║   训练迭代数: $ITERATIONS"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║   Bino 参数配置:"
echo "║     • 不透明度衰减: 启用"
echo "║     • 衰减因子: $OPACITY_DECAY_FACTOR"
echo "║     • 测试迭代: $TEST_ITERATIONS"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# 5. 启动训练
# ============================================================================
echo "🚀 [Training] 开始训练..."
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 启动训练（后台运行，将输出重定向到日志文件）
python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --iterations $ITERATIONS \
    --eval \
    $ENABLE_OPACITY_DECAY \
    --opacity_decay_factor $OPACITY_DECAY_FACTOR \
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
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║   ✅ 训练成功完成！                                                ║"
    echo "╠════════════════════════════════════════════════════════════════════╣"
    echo "║   用时: ${ELAPSED_MIN} 分钟 (${ELAPSED_TIME} 秒)"
    echo "║   输出: $OUTPUT_PATH"
    echo "║   日志: ${OUTPUT_PATH}_train.log"
    echo "╠════════════════════════════════════════════════════════════════════╣"
    echo "║   📊 查看结果:"
    echo "║     1. 最终评估: cat $OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml"
    echo "║     2. TensorBoard: tensorboard --logdir=$OUTPUT_PATH"
    echo "║     3. 训练日志: less ${OUTPUT_PATH}_train.log"
    echo "╠════════════════════════════════════════════════════════════════════╣"
    echo "║   🎯 Foot 3 Views SOTA 基准 (R² baseline):"
    echo "║     PSNR: 28.4873 dB"
    echo "║     SSIM: 0.9005"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""

    # 自动显示最终评估结果（如果存在）
    EVAL_FILE="$OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml"
    if [ -f "$EVAL_FILE" ]; then
        echo "📈 [Results] 最终评估结果:"
        echo "─────────────────────────────────────────────────────────────────"
        cat "$EVAL_FILE" | grep -E "psnr_2d|ssim_2d" | grep -v "_projs"
        echo "─────────────────────────────────────────────────────────────────"
        echo ""
    fi
else
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║   ❌ 训练失败 (退出代码: $TRAIN_STATUS)                            ║"
    echo "╠════════════════════════════════════════════════════════════════════╣"
    echo "║   请检查日志文件: ${OUTPUT_PATH}_train.log"
    echo "║   常见问题:"
    echo "║     1. GPU 内存不足 (OOM)"
    echo "║     2. 数据集路径错误"
    echo "║     3. 依赖包缺失"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    exit 1
fi

# ============================================================================
# 7. 可选：运行评估脚本
# ============================================================================
# 如果想在训练后自动运行 30k 步评估，取消注释下面的行：
# echo "🔍 [Eval] 运行 30k 步评估..."
# bash scripts/eval_single.sh "$OUTPUT_PATH" 30000
