#!/bin/bash
# 快速 Baseline 验证 - 10000 iterations
# 用于对比 GR-Gaussian 效果

# 激活环境
source /home/qyhu/anaconda3/bin/activate r2_gaussian_new

# 工作目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 时间戳
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================"
echo "Quick Baseline Validation - 10k"
echo "========================================"
echo "开始时间: $START_TIME"
echo "预计时长: 1 小时"
echo "迭代次数: 10000"
echo "========================================"

# 运行训练
python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_17_quick_baseline_10k \
    --iterations 10000 \
    --test_iterations 1000 3000 5000 7000 10000 \
    --save_iterations 10000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 5000

# 完成
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "========================================"
echo "✓ Quick Baseline 训练完成"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "输出目录: output/2025_11_17_quick_baseline_10k/"
echo "========================================"
