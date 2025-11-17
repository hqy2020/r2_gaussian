#!/bin/bash
# GR-Gaussian 30k 训练 - 优化版
# 降低 Graph 更新频率以提升性能

# 激活环境
source /home/qyhu/anaconda3/bin/activate r2_gaussian_new

# 工作目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 时间戳
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================"
echo "GR-Gaussian 30k Training - Optimized"
echo "========================================"
echo "开始时间: $START_TIME"
echo "预计时长: 1-2 小时"
echo "迭代次数: 30000"
echo "Graph Laplacian: k=6, λ=8e-4"
echo "更新间隔: 1000 iterations (优化)"
echo "========================================"

# 运行训练
python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_17_gr_gaussian_30k_optimized \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    --enable_graph_laplacian \
    --graph_k 6 \
    --graph_lambda_lap 8e-4 \
    --graph_update_interval 1000

# 完成
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "========================================"
echo "✓ GR-Gaussian 训练完成"
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "输出目录: output/2025_11_17_gr_gaussian_30k_optimized/"
echo "========================================"
