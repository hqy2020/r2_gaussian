#!/bin/bash

# IPSM快速验证脚本 (500迭代)
# 目的: 确认IPSM代码可运行，无crash
# 时间: 约5-10分钟

echo "========================================"
echo "IPSM快速验证 (500迭代)"
echo "========================================"

# 激活环境
echo "激活conda环境: r2_gaussian_new"
conda activate r2_gaussian_new

# 运行训练
python train.py \
    -s /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle \
    -m output/ipsm_test_500_$(date +%m%d_%H%M) \
    --gaussiansN 1 \
    --enable_ipsm \
    --iterations 500 \
    --ipsm_start_iter 100 \
    --ipsm_end_iter 400 \
    --lambda_ipsm 0.1 \
    --lambda_ipsm_depth 0.5 \
    --lambda_ipsm_geo 4.0

echo ""
echo "========================================"
echo "验证完成！"
echo "检查要点:"
echo "  1. 程序是否正常启动"
echo "  2. iter 100是否成功加载扩散模型"
echo "  3. IPSM loss是否正常计算(不是NaN/Inf)"
echo "  4. iter 400是否成功卸载扩散模型"
echo "  5. Total loss是否正常下降"
echo "  6. 是否有CUDA OOM错误"
echo "========================================"
