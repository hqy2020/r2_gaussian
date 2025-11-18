#!/bin/bash
# CoR-GS Bug 修复快速测试脚本
# 运行 100 iterations 验证代码可运行性

echo "========================================"
echo "CoR-GS Bug 修复快速测试"
echo "========================================"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 设置测试参数
MODEL_PATH="output/2025_11_18_foot_3views_corgs_test_bugfix"
ITERATIONS=100
DATA_PATH="/home/qyhu/Documents/r2_ours/r2_gaussian/data/foot_3views"

echo "测试参数："
echo "  - 模型路径: $MODEL_PATH"
echo "  - 迭代次数: $ITERATIONS"
echo "  - 数据路径: $DATA_PATH"
echo ""

# 检查数据路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 错误：数据路径不存在: $DATA_PATH"
    echo "请根据实际数据路径修改此脚本"
    exit 1
fi

# 运行训练
python train.py \
    --source_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --iterations $ITERATIONS \
    --test_iterations $ITERATIONS \
    --save_iterations $ITERATIONS \
    --enable_pseudo_coreg \
    --lambda_pseudo 0.1 \
    --pseudo_start_iter 0 \
    --gaussiansN 2 \
    --coreg \
    2>&1 | tee test_corgs_bugfix.log

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
echo "请检查日志文件: test_corgs_bugfix.log"
echo ""
echo "关键检查点："
echo "1. 是否成功生成 10,000 个 pseudo-view？"
echo "2. 是否输出了 Pseudo Co-reg Loss？"
echo "3. 是否有任何错误或警告？"
echo ""
