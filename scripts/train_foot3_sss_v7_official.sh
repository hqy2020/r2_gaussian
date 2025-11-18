#!/bin/bash

###############################################################################
# SSS-v7-OFFICIAL (官方实现) - foot 3 views 训练脚本
#
# 生成日期: 2025-11-18
# 基于: https://github.com/realcrane/3D-student-splatting-and-scooping
#
# 修复内容 (所有 5 个 bug):
#   1. ✅ Bug 1: 启用 SSS (use_student_t = args.enable_sss)
#   2. ✅ Bug 2: 恢复 tanh 激活函数 ([-1, 1] 官方范围)
#   3. ✅ Bug 3: 移除渐进式 Scooping 限制
#   4. ✅ Bug 4: 替换 Balance Loss (官方 L1 正则化)
#   5. ✅ Bug 5: 实现组件回收机制 (替代传统 densification)
#
# 训练策略: 30k 完整训练，验证官方实现
###############################################################################

set -e  # 遇到错误立即退出

# 激活 conda 环境
echo "🔧 [Setup] Activating conda environment: r2_gaussian_new"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 训练参数
DATA_PATH="data/369/foot_50_3views.pickle"
OUTPUT_PATH="output/2025_11_18_foot_3views_sss_v7_official"
ITERATIONS=30000  # 完整训练

# SSS 官方超参数 (基于官方仓库)
NU_LR=0.001           # nu 学习率 (官方默认)
OPACITY_LR=0.005      # opacity 学习率 (官方默认，从 0.01 降低)
NU_DEGREE=10          # Student-t 自由度初始值 (官方默认)

# 组件回收参数 (官方)
OPACITY_THRESHOLD=0.005     # 低 opacity 阈值
OPACITY_REG=0.01            # Balance Loss 权重 (L1 正则)

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
echo "║   🌟 SSS-v7-OFFICIAL: Student Splatting (官方实现)        ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║   数据集: $DATA_PATH"
echo "║   输出: $OUTPUT_PATH"
echo "║   迭代数: $ITERATIONS"
echo "║"
echo "║   📊 官方参数:"
echo "║     • Opacity 激活: tanh [-1, 1]"
echo "║     • Balance Loss: L1 正则化 (权重 $OPACITY_REG)"
echo "║     • 组件回收: 5% threshold=$OPACITY_THRESHOLD"
echo "║     • Nu degree: $NU_DEGREE"
echo "║     • Learning rates: nu=$NU_LR, opacity=$OPACITY_LR"
echo "║"
echo "║   ✅ 修复的 5 个 Bug:"
echo "║     1. 启用 SSS (--enable_sss)"
echo "║     2. tanh 激活函数 ([-1, 1] 官方范围)"
echo "║     3. 移除渐进式 Scooping 限制"
echo "║     4. L1 正则化 Balance Loss"
echo "║     5. 组件回收机制 (替代 densification)"
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
    echo "✅ [Success] SSS-v7-OFFICIAL 训练完成!"
    echo "   结果保存在: $OUTPUT_PATH"
    echo "   日志保存在: $LOGFILE"
    echo ""
    echo "📊 [诊断] 关键指标检查:"
    echo ""

    # 提取最后一次的 opacity balance 日志
    echo "   🎯 Opacity Balance (官方实现，最后一次记录):"
    grep -E "SSS-Official.*Iter|Balance:|opacity" "$LOGFILE" | tail -10

    echo ""
    echo "   ♻️ 组件回收记录:"
    grep -E "SSS-Recycle" "$LOGFILE" | tail -5

    echo ""
    echo "   📈 2D 测试集 PSNR (iter 30000):"
    if [ -f "$OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml" ]; then
        grep "psnr_2d:" "$OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml" | head -1
    else
        echo "   ⚠️  评估文件未生成"
    fi

    echo ""
    echo "📊 [性能对比]:"
    echo "   Baseline (30k):     PSNR=28.31 dB, SSIM=0.898"
    echo "   FSGS (30k):         PSNR=28.45 dB, SSIM=0.901"
    echo "   SSS-v5 (Bug版):     PSNR=20.16 dB ❌ (失败)"
    echo "   SSS-v6 (部分修复):  训练中断 ⏸️"
    echo "   SSS-v7 (官方实现):  查看上方结果 ⬆️ (预期 >28 dB)"
    echo ""
    echo "🎯 [预期目标]:"
    echo "   • PSNR: ≥ 28 dB (接近或超过 Baseline)"
    echo "   • Opacity 平衡: 40-60% 正值 / 40-60% 负值"
    echo "   • 组件回收: 稳定执行，~5% 比例"
    echo "   • 训练稳定: 无 NaN, 无崩溃"
else
    echo "❌ [Error] 训练失败，请检查日志: $LOGFILE"
    exit 1
fi
