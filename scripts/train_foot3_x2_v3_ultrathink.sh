#!/bin/bash

# X²-Gaussian v3 终极版本 - 强正则化 + 超保守调制
# 目标：解决过拟合问题，改善测试视角泛化能力
#
# 核心改进：
# 1. sigmoid 调制范围 [0.7, 1.3]（±30%），比 v2 更保守
# 2. TV 正则化提升 10 倍：0.0002 → 0.002
# 3. Decoder 学习率降低到 Encoder 的 0.5 倍
# 4. 目标：PSNR > 28.6 dB，否则判定 X²-Gaussian 不适合 3-view 场景

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_x2_v3_ultrathink"

echo "========================================"
echo "X²-Gaussian v3 终极版本训练"
echo "========================================"
echo "启动时间: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "关键参数:"
echo "  - 调制方式: sigmoid [0.7, 1.3]"
echo "  - TV 正则化: lambda_plane_tv=0.002 (10倍增强)"
echo "  - Decoder 学习率: 0.001 (Encoder 的 0.5 倍)"
echo "  - 迭代次数: 30000"
echo "========================================"

conda run -n r2_gaussian_new python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path ${OUTPUT_DIR} \
    --gaussiansN 1 \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --kplanes_decoder_hidden 128 \
    --kplanes_decoder_layers 3 \
    --kplanes_lr_init 0.002 \
    --kplanes_lr_final 0.0002 \
    --lambda_plane_tv 0.002 \
    --tv_loss_type l2 \
    --iterations 30000 \
    --position_lr_init 0.0002 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.00005 \
    --test_iterations 5000 10000 20000 30000 \
    --save_iterations 30000 \
    --eval

echo ""
echo "训练完成时间: $(date)"
echo "查看结果: cat ${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"
