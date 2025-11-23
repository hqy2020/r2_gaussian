#!/bin/bash

###############################################################################
# X²-Gaussian 修复后完整训练脚本 (Foot-3 views)
#
# 修复内容：
# 1. ✅ 添加 DensityMLPDecoder 类（3 层 MLP，hidden_dim=128）
# 2. ✅ 修改 get_density 使用 decoder 而非简单 mean
# 3. ✅ 优化器集成 decoder 参数
# 4. ✅ 学习率调度器支持 decoder
#
# 预期改进：
# - Baseline: PSNR 28.49 dB, SSIM 0.900
# - 目标: PSNR > 29.0 dB, SSIM > 0.910 (+0.5 dB / +0.01)
#
# 日期：2025-11-24
###############################################################################

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# 训练命名（修复版）
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
ORGAN="foot"
VIEWS=3
TECHNIQUE="x2_fixed"
OUTPUT_DIR="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_${TECHNIQUE}"

echo "======================================================================"
echo "🚀 X²-Gaussian 修复版训练 (Foot-3 views)"
echo "======================================================================"
echo "输出目录: $OUTPUT_DIR"
echo "数据集: data/369/foot_50_3views.pickle"
echo "Baseline: PSNR 28.49 dB, SSIM 0.900"
echo "目标: PSNR > 29.0 dB, SSIM > 0.910"
echo "预计时间: 8-10 小时"
echo "======================================================================"
echo ""

# 核心参数（对齐 X²-Gaussian 原版）
ENABLE_KPLANES="--enable_kplanes"
KPLANES_RESOLUTION=64
KPLANES_DIM=32
KPLANES_DECODER_HIDDEN=128  # MLP Decoder 隐藏层维度
KPLANES_DECODER_LAYERS=3    # MLP Decoder 层数

# K-Planes 学习率（已修正为原版设置）
KPLANES_LR_INIT=0.002
KPLANES_LR_FINAL=0.0002
KPLANES_LR_MAX_STEPS=30000

# TV 正则化（对齐 X²-Gaussian L2 损失）
LAMBDA_PLANE_TV=0.0002
TV_LOSS_TYPE="l2"

# 训练超参数
ITERATIONS=30000
DENSIFY_UNTIL_ITER=15000
POSITION_LR_INIT=0.0002
DENSITY_LR_INIT=0.01
SCALING_LR_INIT=0.005
ROTATION_LR_INIT=0.001

# 开始训练
python3 train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path "$OUTPUT_DIR" \
    --eval \
    --iterations $ITERATIONS \
    --position_lr_init $POSITION_LR_INIT \
    --density_lr_init $DENSITY_LR_INIT \
    --scaling_lr_init $SCALING_LR_INIT \
    --rotation_lr_init $ROTATION_LR_INIT \
    --densify_until_iter $DENSIFY_UNTIL_ITER \
    $ENABLE_KPLANES \
    --kplanes_resolution $KPLANES_RESOLUTION \
    --kplanes_dim $KPLANES_DIM \
    --kplanes_decoder_hidden $KPLANES_DECODER_HIDDEN \
    --kplanes_decoder_layers $KPLANES_DECODER_LAYERS \
    --kplanes_lr_init $KPLANES_LR_INIT \
    --kplanes_lr_final $KPLANES_LR_FINAL \
    --kplanes_lr_max_steps $KPLANES_LR_MAX_STEPS \
    --lambda_plane_tv $LAMBDA_PLANE_TV \
    --tv_loss_type $TV_LOSS_TYPE \
    2>&1 | tee "${OUTPUT_DIR}_train.log"

echo ""
echo "======================================================================"
echo "✅ 训练完成！"
echo "======================================================================"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: ${OUTPUT_DIR}_train.log"
echo ""
echo "查看结果："
echo "  cd $OUTPUT_DIR/eval"
echo "  cat */eval2d_render_test.yml"
echo ""
echo "对比 baseline："
echo "  Baseline: PSNR 28.49 dB, SSIM 0.900"
echo "  期望提升: +0.5 dB / +0.01"
echo "======================================================================"
