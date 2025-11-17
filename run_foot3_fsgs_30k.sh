#!/bin/bash

# FSGS 扩展训练脚本 - Foot 3 Views (30000 轮)
# 目标：延长训练时间，对齐 baseline (30k)，看能否追平或超越
# Baseline: PSNR=28.547, SSIM=0.9008 (训练 30000 轮)
# 15k 结果: PSNR=28.313, SSIM=0.9003 (低于 baseline)
# 目标: 30k 后达到 PSNR ≥ 28.6 dB

# 设置工作目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 获取当前日期 (格式: YYYY_MM_DD)
DATE=$(date +%Y_%m_%d)

# 数据集路径
DATA_FILE="/home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle"
INIT_FILE="/home/qyhu/Documents/r2_ours/r2_gaussian/data/369/init_foot_50_3views.npy"

# 输出目录
OUTPUT_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/output/${DATE}_foot_3views_fsgs_30k"

# 日志文件
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "=========================================="
echo "FSGS Extended Training - Foot 3 Views (30k)"
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
echo "   - Pseudo Label Weight: 0.3"
echo "   - Proximity Threshold: 4.0"
echo "   - Training Iterations: 30000 (与 baseline 对齐)"
echo "   - Test Iterations: 5k/10k/15k/20k/25k/30k"
echo "   - Densify Until: 25000 (延长密化期)"
echo "=========================================="
echo ""
echo "⏱️  预计训练时间: ~8-10 小时"
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

# 30000 轮训练，延长密化期到 25000
python train.py \
    -s "$DATA_FILE" \
    -m "$OUTPUT_DIR" \
    --ply_path "$INIT_FILE" \
    --iterations 30000 \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
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
    --densify_until_iter 25000 \
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

    # 提取所有评估结果
    echo "🎯 完整评估结果:"
    echo ""
    for iter in 5000 10000 15000 20000 25000 30000; do
        ITER_DIR=$(printf "%06d" $iter)
        EVAL_FILE="${OUTPUT_DIR}/eval/iter_${ITER_DIR}/eval2d_render_test.yml"
        if [ -f "$EVAL_FILE" ]; then
            echo "Iteration $iter:"
            grep "psnr_2d:" "$EVAL_FILE" | head -1
            grep "ssim_2d:" "$EVAL_FILE" | head -1
            echo ""
        fi
    done

    echo "🎯 最终指标 (iter 30000):"
    if [ -f "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" ]; then
        grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | head -1
        grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | head -1
        echo ""
        echo "📈 对比 Baseline:"
        echo "   Baseline: PSNR=28.547, SSIM=0.9008"
        FINAL_PSNR=$(grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | head -1 | awk '{print $2}')
        echo "   FSGS 30k: PSNR=$FINAL_PSNR"
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
