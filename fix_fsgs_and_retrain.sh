#!/bin/bash

# FSGS 修复版实验 - Foot 3 Views
# 目标: 修复 FSGS 未运行的问题,提升到至少 28.5 dB

cd /home/qyhu/Documents/r2_ours/r2_gaussian

export CUDA_VISIBLE_DEVICES=0

DATE=$(date +%Y_%m_%d)
DATA_FILE="/home/qyhu/Documents/r2_ours/r2_gaussian/data/r2-sax-nerf/0_foot_cone_3views.pickle"
INIT_FILE="/home/qyhu/Documents/r2_ours/r2_gaussian/data/r2-sax-nerf/init_0_foot_cone_3views.npy"
OUTPUT_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/output/${DATE}_foot_3views_fsgs_FIXED"
LOG_FILE="${OUTPUT_DIR}/training.log"

echo "=========================================="
echo "FSGS FIXED - Foot 3 Views"
echo "=========================================="
echo "📅 日期: $DATE"
echo "📂 数据: $DATA_FILE"
echo "📂 输出: $OUTPUT_DIR"
echo ""
echo "🔧 修复内容:"
echo "   1. 移除 enable_fsgs_depth 条件判断"
echo "   2. 提升 pseudo_label_weight: 0.05 → 0.3"
echo "   3. 降低 proximity_threshold: 6.0 → 4.0"
echo "   4. 增加训练迭代: 10000 → 15000"
echo "   5. 调整 densify_until_iter: 15000 → 12000"
echo "=========================================="
echo ""

# 检查数据文件
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 错误: 数据文件不存在: $DATA_FILE"
    exit 1
fi

if [ ! -f "$INIT_FILE" ]; then
    echo "❌ 错误: 初始化文件不存在: $INIT_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "🚀 开始训练..."
echo "开始时间: $(date)"
echo ""

source /home/qyhu/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 🌟 关键修复: 提升所有 FSGS 参数
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
    --fsgs_depth_model disabled \
    --enable_fsgs_pseudo_views \
    --num_fsgs_pseudo_views 10 \
    --fsgs_noise_std 0.05 \
    --fsgs_start_iter 2000 \
    --pseudo_label_weight 0.3 \
    --densify_until_iter 12000 \
    2>&1 | tee "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成！"
    echo "完成时间: $(date)"
    echo ""
    echo "📊 查看结果:"
    echo "   - 日志: $LOG_FILE"
    echo "   - 输出: $OUTPUT_DIR"
    echo ""

    # 提取最终指标
    if [ -f "${OUTPUT_DIR}/eval/iter_015000/eval2d_render_test.yml" ]; then
        echo "🎯 最终指标:"
        grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_015000/eval2d_render_test.yml"
        grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_015000/eval2d_render_test.yml"
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
