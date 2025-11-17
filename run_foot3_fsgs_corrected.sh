#!/bin/bash

# FSGS 完整实验脚本 - Foot 3 Views (修正版)
# 目标：使用 FSGS 伪视角生成 + Proximity-guided 密化超越 baseline
# Baseline: PSNR=28.547, SSIM=0.9008 (data/369/foot_50_3views.pickle)
# 目标: PSNR ≥ 28.8 dB

# 设置工作目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 获取当前日期 (格式: YYYY_MM_DD)
DATE=$(date +%Y_%m_%d)

# ⚠️ 修正: 使用正确的数据集路径
DATA_FILE="/home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle"
INIT_FILE="/home/qyhu/Documents/r2_ours/r2_gaussian/data/369/init_foot_50_3views.npy"

# 输出目录
OUTPUT_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/output/${DATE}_foot_3views_fsgs_corrected"

# 日志文件
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "=========================================="
echo "FSGS Complete System - Foot 3 Views (CORRECTED)"
echo "=========================================="
echo "📅 日期: $DATE"
echo "📂 数据: $DATA_FILE"
echo "📂 初始化: $INIT_FILE"
echo "📂 输出: $OUTPUT_DIR"
echo ""
echo "🎯 FSGS 配置:"
echo "   - Proximity-guided Densification: ✅"
echo "   - Depth Supervision: ❌ (disabled)"
echo "   - Pseudo Views Generation: ✅"
echo "   - Pseudo Label Weight: 0.3 (强化)"
echo "   - Proximity Threshold: 4.0 (更激进)"
echo "   - Training Iterations: 15000"
echo "   - Strategy: Proximity + Pseudo Views"
echo "=========================================="
echo ""

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 错误: 数据文件不存在: $DATA_FILE"
    exit 1
fi

if [ ! -f "$INIT_FILE" ]; then
    echo "❌ 错误: 初始化文件不存在: $INIT_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 开始训练
echo "🚀 开始训练..."
echo "开始时间: $(date)"
echo ""

source /home/qyhu/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# ⚠️ 修正: 移除 --initialization 参数,使用 --ply_path
python train.py \
    -s "$DATA_FILE" \
    -m "$OUTPUT_DIR" \
    --ply_path "$INIT_FILE" \
    --iterations 15000 \
    --test_iterations 5000 10000 15000 \
    --save_iterations 5000 10000 15000 \
    --quiet \
    --eval \
    --enable_fsgs_proximity \
    --proximity_threshold 4.0 \
    --enable_medical_constraints \
    --proximity_organ_type foot \
    --proximity_k_neighbors 8 \
    --fsgs_depth_model disabled \
    --enable_fsgs_pseudo_views \
    --num_fsgs_pseudo_views 10 \
    --fsgs_noise_std 0.05 \
    --fsgs_start_iter 2000 \
    --pseudo_label_weight 0.3 \
    --densify_until_iter 12000 \
    2>&1 | tee "$LOG_FILE"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成！"
    echo "完成时间: $(date)"
    echo ""
    echo "📊 查看结果:"
    echo "   - 日志: $LOG_FILE"
    echo "   - 输出: $OUTPUT_DIR"
    echo ""

    # 提取最终评估结果
    if [ -f "${OUTPUT_DIR}/eval/iter_015000/eval2d_render_test.yml" ]; then
        echo "🎯 最终指标 (iter 15000):"
        grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_015000/eval2d_render_test.yml"
        grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_015000/eval2d_render_test.yml"
    fi

    if [ -f "${OUTPUT_DIR}/eval/iter_010000/eval2d_render_test.yml" ]; then
        echo ""
        echo "🎯 中间指标 (iter 10000):"
        grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_010000/eval2d_render_test.yml"
        grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_010000/eval2d_render_test.yml"
    fi
else
    echo ""
    echo "❌ 训练失败！"
    echo "失败时间: $(date)"
    echo "请检查日志: $LOG_FILE"
    exit 1
fi

echo "=========================================="
echo "实验完成！"
echo "=========================================="
