#!/bin/bash

# 简化版 IPSM 快速验证脚本
# 目的: 验证简化版 IPSM 代码可运行且有效果
# 时间: 约 5-10 分钟

echo "========================================"
echo "简化版 IPSM 快速验证 (5000 迭代)"
echo "========================================"

# 激活环境
echo "激活 conda 环境: r2_gaussian_new"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 生成输出目录名
OUTPUT_DIR="output/2025_11_26_ipsm_simple_foot3_5k_$(date +%H%M)"

echo "输出目录: $OUTPUT_DIR"

# 运行训练
python train.py \
    -s /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle \
    -m $OUTPUT_DIR \
    --gaussiansN 1 \
    --enable_ipsm \
    --iterations 5000 \
    --ipsm_start_iter 1000 \
    --ipsm_end_iter 4500 \
    --lambda_ipsm_geo 0.1 \
    --lambda_ipsm_tv 0.01 \
    --ipsm_pseudo_angle_range 10.0 \
    --ipsm_min_angle_diff 2.0 \
    --ipsm_warmup_iters 500 \
    --test_iterations 1000 2500 5000

echo ""
echo "========================================"
echo "验证完成！"
echo "检查要点:"
echo "  1. 程序是否正常启动"
echo "  2. iter 1000 是否显示 IPSM guidance started"
echo "  3. IPSM loss 是否正常计算 (不是 NaN/Inf)"
echo "  4. Geo loss 是否有变化（非 0）"
echo "  5. Total loss 是否正常下降"
echo "========================================"
echo ""
echo "查看结果: ls -la $OUTPUT_DIR/eval/"
