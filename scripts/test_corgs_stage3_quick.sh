#!/bin/bash

# CoR-GS Stage 3 快速验证脚本（100 iterations）
# 用途: 验证集成后的基础功能是否正常
# 预计耗时: ~3-5 分钟

set -e  # 遇到错误立即退出

# 激活环境
echo "激活 r2_gaussian_new 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 清理旧输出（如有）
OUTPUT_DIR="output/test_corgs_stage3_quick"
if [ -d "$OUTPUT_DIR" ]; then
    echo "清理旧输出目录: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

# 运行快速验证
echo ""
echo "=========================================="
echo "CoR-GS Stage 3 快速验证测试"
echo "=========================================="
echo "配置:"
echo "  - 数据集: data/369 (Foot 3 views)"
echo "  - 迭代次数: 100"
echo "  - 高斯模型数: 2 (粗模型 + 精细模型)"
echo "  - Pseudo-view 启动: iteration 50"
echo "  - Lambda_pseudo: 1.0"
echo "  - Noise_std: 0.02"
echo "=========================================="
echo ""

python train.py \
    --source_path data/369 \
    --model_path "$OUTPUT_DIR" \
    --iterations 100 \
    --gaussiansN 2 \
    --coreg \
    --enable_pseudo_coreg \
    --lambda_pseudo 1.0 \
    --pseudo_noise_std 0.02 \
    --pseudo_start_iter 50 \
    --test_iterations 100 \
    --save_iterations -1 \
    --quiet

echo ""
echo "=========================================="
echo "✅ 快速验证完成！"
echo "=========================================="
echo ""
echo "检查清单:"
echo "  1. 查看控制台输出:"
echo "     - 应出现 '✅ CoR-GS Stage 3 modules available'"
echo "     - iterations 50-100 应出现 '[Pseudo Co-reg]' 日志"
echo ""
echo "  2. 查看 TensorBoard:"
echo "     tensorboard --logdir $OUTPUT_DIR"
echo "     - 应出现 'train_loss_patches/pseudo_coreg_*' 指标"
echo ""
echo "  3. 验证成功标准:"
echo "     - 无 Python 异常"
echo "     - Pseudo-view loss 值正常（不为 NaN/Inf）"
echo "     - SSIM 在 [0, 1] 范围内"
echo "=========================================="
