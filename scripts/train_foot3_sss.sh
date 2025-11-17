#!/bin/bash

###############################################################################
# SSS (Student Splatting and Scooping) - foot 3 views 训练脚本
#
# 生成日期: 2025-11-17
# 目标: PSNR ≥ 28.8 dB (超越 baseline 28.547 dB)
# 数据集: foot 3 views (稀疏视角医学 CT 重建)
#
# 使用方法:
#   bash scripts/train_foot3_sss.sh
###############################################################################

set -e  # 遇到错误立即退出

# 激活 conda 环境
echo "🔧 [Setup] Activating conda environment: r2_gaussian_new"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 训练参数
DATA_PATH="data/369/foot_50_3views.pickle"
OUTPUT_PATH="output/2025_11_17_foot_3views_sss"
ITERATIONS=10000

# SSS 超参数 (针对 foot 3 views 调优)
NU_LR=0.001         # nu 学习率
OPACITY_LR=0.01     # opacity 学习率

# 检查数据集是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ [Error] 数据集不存在: $DATA_PATH"
    echo "   请确保数据集路径正确,或运行数据准备脚本"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 启动训练
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🎓 SSS-R²: Student Splatting and Scooping               ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║   数据集: $DATA_PATH"
echo "║   输出: $OUTPUT_PATH"
echo "║   迭代数: $ITERATIONS"
echo "║   SSS 参数: nu_lr=$NU_LR, opacity_lr=$OPACITY_LR"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --iterations $ITERATIONS \
    --eval \
    --enable_sss \
    --nu_lr_init $NU_LR \
    --opacity_lr_init $OPACITY_LR \
    --test_iterations 1 5000 10000 \
    --save_iterations 10000

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ [Success] 训练完成!"
    echo "   结果保存在: $OUTPUT_PATH"
    echo ""
    echo "📊 [Next Steps] 查看结果:"
    echo "   1. TensorBoard: tensorboard --logdir=$OUTPUT_PATH/tensorboard"
    echo "   2. 评估结果: cat $OUTPUT_PATH/eval/iter_010000/eval2d_render_test.yml"
    echo "   3. 对比 baseline: python scripts/compare_results.py $OUTPUT_PATH output/foot_3_1013"
else
    echo "❌ [Error] 训练失败,请检查日志"
    exit 1
fi
