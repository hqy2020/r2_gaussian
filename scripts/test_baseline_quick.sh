#!/bin/bash
# 快速测试 baseline 训练流程

# 激活环境
source /home/qyhu/anaconda3/bin/activate r2_gaussian_new

# 工作目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 快速测试 (1000 iterations)
python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_17_foot_3views_baseline_quicktest \
    --iterations 1000 \
    --test_iterations 1000 \
    --save_iterations 1000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 500

echo "✓ 快速测试完成"
