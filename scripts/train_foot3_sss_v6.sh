#!/bin/bash

###############################################################################
# SSS-v6-FIX (Bug修复版本) - foot 3 views 快速测试脚本
#
# 生成日期: 2025-11-18
# 修复内容:
#   1. 🐛 Bug 1: Densification 负值传播 → 基于 density 重新初始化
#   2. 🐛 Bug 2: Balance Loss 梯度失效 → 直接惩罚负值 + 鼓励正值
#   3. 🐛 Bug 3: Opacity 激活范围过大 → 从 [-1,1] 改为 [-0.2,1.0]
#
# 训练策略: 直接 30k 完整训练验证修复效果
###############################################################################

set -e  # 遇到错误立即退出

# 激活 conda 环境
echo "🔧 [Setup] Activating conda environment: r2_gaussian_new"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 训练参数
DATA_PATH="data/369/foot_50_3views.pickle"
OUTPUT_PATH="output/2025_11_18_foot_3views_sss_v6"
ITERATIONS=30000  # 完整训练: 30k

# SSS 超参数 (v6: 与 v5 一致，但代码已修复)
NU_LR=0.001         # nu 学习率
OPACITY_LR=0.01     # opacity 学习率

# 检查数据集是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ [Error] 数据集文件不存在: $DATA_PATH"
    echo "   请确保数据集路径正确"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_PATH"

# 启动训练
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   🔧 SSS-v6-FIX: Student Splatting and Scooping (Bug修复)  ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║   数据集: $DATA_PATH"
echo "║   输出: $OUTPUT_PATH"
echo "║   迭代数: $ITERATIONS (快速测试)"
echo "║   SSS 参数: nu_lr=$NU_LR, opacity_lr=$OPACITY_LR"
echo "║"
echo "║   ✅ Bug Fixes:"
echo "║     1. Densification: 正值初始化 (防止负值传播)"
echo "║     2. Balance Loss: 直接梯度 (修复梯度失效)"
echo "║     3. Opacity Range: [-0.2, 1.0] (缩小负值范围)"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 记录训练日志
LOGFILE="${OUTPUT_PATH}_train.log"

python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --iterations $ITERATIONS \
    --eval \
    --enable_sss \
    --nu_lr_init $NU_LR \
    --opacity_lr_init $OPACITY_LR \
    --test_iterations 1 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    2>&1 | tee "$LOGFILE"

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ [Success] SSS-v6 完整训练完成!"
    echo "   结果保存在: $OUTPUT_PATH"
    echo "   日志保存在: $LOGFILE"
    echo ""
    echo "📊 [诊断] 关键指标检查:"
    echo ""

    # 提取最后一次的 opacity balance 日志
    echo "   🔧 Opacity Balance (最后一次记录):"
    grep -E "SSS-v6-FIX.*Iter|Balance:|Extremes:" "$LOGFILE" | tail -6

    echo ""
    echo "   📈 2D 测试集 PSNR (iter 30000):"
    if [ -f "$OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml" ]; then
        grep "psnr_2d:" "$OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml" | head -1
    else
        echo "   ⚠️  评估文件未生成"
    fi

    echo ""
    echo "📊 [结果对比]:"
    echo "   Baseline (30k): PSNR=28.31 dB, SSIM=0.898"
    echo "   FSGS (30k):     PSNR=28.45 dB, SSIM=0.901"
    echo "   SSS-v5 (30k):   PSNR=20.16 dB (失败) ❌"
    echo "   SSS-v6 (30k):   查看上方结果 ⬆️"
else
    echo "❌ [Error] 训练失败,请检查日志: $LOGFILE"
    exit 1
fi
