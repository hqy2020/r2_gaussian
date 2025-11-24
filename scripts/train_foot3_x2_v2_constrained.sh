#!/bin/bash

# X²-Gaussian v2 训练脚本 - 约束 MLP 输出版本
# 修改：添加 Tanh 激活函数 + 温和调制 (±50%)
# 预期：PSNR 28.7~28.9 dB（比 v1 的 28.431 dB 提升 0.3~0.5 dB）

DATASET="data/369/foot_50_3views.pickle"
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_x2_v2_constrained"

echo "=========================================="
echo "X²-Gaussian v2 训练启动"
echo "=========================================="
echo "数据集: ${DATASET}"
echo "输出目录: ${OUTPUT_DIR}"
echo "训练配置:"
echo "  - gaussiansN=1 (单模型)"
echo "  - K-Planes: 64x64, dim=32"
echo "  - MLP Decoder: 3层128维 + Tanh约束"
echo "  - 调制范围: [0.5, 1.5] (±50%)"
echo "  - 迭代次数: 30000"
echo "=========================================="

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 启动训练
python train.py \
    -s ${DATASET} \
    -m ${OUTPUT_DIR} \
    --gaussiansN 1 \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --kplanes_decoder_hidden 128 \
    --kplanes_decoder_layers 3 \
    --kplanes_lr_init 0.002 \
    --kplanes_lr_final 0.0002 \
    --lambda_plane_tv 0.0002 \
    --tv_loss_type l2 \
    --iterations 30000 \
    --densify_until_iter 15000 \
    --test_iterations 5000 10000 20000 30000

echo ""
echo "训练启动成功！"
echo "监控命令："
echo "  tail -f ${OUTPUT_DIR}.log"
echo "  tensorboard --logdir ${OUTPUT_DIR} --port 6006"
