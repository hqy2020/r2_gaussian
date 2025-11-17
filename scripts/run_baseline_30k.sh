#!/bin/bash
# Baseline 训练 - 30000 iterations
# 用于 GR-Gaussian 对比实验

# 激活环境
source /home/qyhu/anaconda3/bin/activate r2_gaussian_new

# 工作目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 时间戳
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================"
echo "Baseline Training - Foot 3 Views"
echo "========================================"
echo "开始时间: $START_TIME"
echo "预计时长: 30-35 分钟"
echo "迭代次数: 30000"
echo "========================================"

# 运行训练
python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_17_foot_3views_baseline_30k \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000

# 完成
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "========================================"
echo "✓ Baseline 训练完成"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "输出目录: output/2025_11_17_foot_3views_baseline_30k/"
echo "========================================"
