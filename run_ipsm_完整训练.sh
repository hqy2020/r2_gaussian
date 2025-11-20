#!/bin/bash

# IPSM完整训练脚本 (30,000迭代)
# 目的: 与baseline对比，验证IPSM效果
# 时间: 约1-2小时
# 警告: 会消耗1只小动物 🐾

echo "========================================"
echo "IPSM完整训练 (30,000迭代)"
echo "Foot-3视角数据集"
echo "警告: 此训练将消耗约1-2小时和1只小动物🐾"
echo "========================================"
read -p "确认开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "训练取消"
    exit 1
fi

# 激活环境
echo "激活conda环境: r2_gaussian_new"
conda activate r2_gaussian_new

# 生成时间戳
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_ipsm"

echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行训练
python train.py \
    -s /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle \
    -m $OUTPUT_DIR \
    --gaussiansN 1 \
    --enable_ipsm \
    --lambda_ipsm 1.0 \
    --lambda_ipsm_depth 0.5 \
    --lambda_ipsm_geo 4.0 \
    --ipsm_eta_r 0.1 \
    --ipsm_eta_d 0.1 \
    --ipsm_mask_tau 0.3 \
    --ipsm_mask_tau_geo 0.1 \
    --ipsm_cfg_scale 7.5 \
    --ipsm_start_iter 2000 \
    --ipsm_end_iter 9500 \
    --ipsm_pseudo_angle_range 15.0 \
    --iterations 30000

echo ""
echo "========================================"
echo "训练完成！"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "下一步操作:"
echo "  1. 运行评估:"
echo "     python test.py -m $OUTPUT_DIR"
echo ""
echo "  2. 查看TensorBoard:"
echo "     tensorboard --logdir $OUTPUT_DIR --port 6006"
echo ""
echo "  3. 对比baseline结果:"
echo "     Baseline (Foot-3):"
echo "       PSNR: 28.4873"
echo "       SSIM: 0.9005"
echo ""
echo "     期望IPSM结果:"
echo "       PSNR: > 28.5 (+0.5%)"
echo "       SSIM: > 0.901 (+0.05%)"
echo "========================================"
