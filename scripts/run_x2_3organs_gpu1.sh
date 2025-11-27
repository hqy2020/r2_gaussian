#!/bin/bash

# X² v3 并行训练脚本：Head, Abdomen, Pancreas 3 views (GPU 1)
# 创建时间：2025-11-24
# 基于 Foot X² v3 成功配置

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
GPU_ID=1

echo "=========================================="
echo "X² v3 并行训练启动"
echo "时间戳: $TIMESTAMP"
echo "GPU: $GPU_ID"
echo "器官: Head, Abdomen, Pancreas"
echo "=========================================="

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 并行启动 3 个训练（后台运行）
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --source_path data/369/head_50_3views.pickle \
    --ply_path data/369/init_head_50_3views.npy \
    --model_path output/${TIMESTAMP}_head_3views_x2_v3 \
    --iterations 30000 \
    --eval \
    --gaussiansN 1 \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --kplanes_decoder_hidden 128 \
    --kplanes_decoder_layers 3 \
    --kplanes_lr_init 0.002 \
    --kplanes_lr_final 0.0002 \
    --lambda_plane_tv 0.002 \
    > output/${TIMESTAMP}_head_3views_x2_v3.log 2>&1 &

HEAD_PID=$!
echo "[启动] Head 训练 - PID: $HEAD_PID"
sleep 2

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --source_path data/369/abdomen_50_3views.pickle \
    --ply_path data/369/init_abdomen_50_3views.npy \
    --model_path output/${TIMESTAMP}_abdomen_3views_x2_v3 \
    --iterations 30000 \
    --eval \
    --gaussiansN 1 \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --kplanes_decoder_hidden 128 \
    --kplanes_decoder_layers 3 \
    --kplanes_lr_init 0.002 \
    --kplanes_lr_final 0.0002 \
    --lambda_plane_tv 0.002 \
    > output/${TIMESTAMP}_abdomen_3views_x2_v3.log 2>&1 &

ABDOMEN_PID=$!
echo "[启动] Abdomen 训练 - PID: $ABDOMEN_PID"
sleep 2

CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
    --source_path data/369/pancreas_50_3views.pickle \
    --ply_path data/369/init_pancreas_50_3views.npy \
    --model_path output/${TIMESTAMP}_pancreas_3views_x2_v3 \
    --iterations 30000 \
    --eval \
    --gaussiansN 1 \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --kplanes_decoder_hidden 128 \
    --kplanes_decoder_layers 3 \
    --kplanes_lr_init 0.002 \
    --kplanes_lr_final 0.0002 \
    --lambda_plane_tv 0.002 \
    > output/${TIMESTAMP}_pancreas_3views_x2_v3.log 2>&1 &

PANCREAS_PID=$!
echo "[启动] Pancreas 训练 - PID: $PANCREAS_PID"

echo ""
echo "=========================================="
echo "所有训练已启动（后台运行）"
echo "=========================================="
echo "Head    - PID: $HEAD_PID    - 日志: output/${TIMESTAMP}_head_3views_x2_v3.log"
echo "Abdomen - PID: $ABDOMEN_PID - 日志: output/${TIMESTAMP}_abdomen_3views_x2_v3.log"
echo "Pancreas- PID: $PANCREAS_PID- 日志: output/${TIMESTAMP}_pancreas_3views_x2_v3.log"
echo ""
echo "监控命令:"
echo "  watch -n 1 nvidia-smi                 # GPU 使用"
echo "  ps aux | grep 'train.py' | grep -v grep  # 进程状态"
echo "  tail -f output/${TIMESTAMP}_head_3views_x2_v3.log     # Head 日志"
echo "  tail -f output/${TIMESTAMP}_abdomen_3views_x2_v3.log  # Abdomen 日志"
echo "  tail -f output/${TIMESTAMP}_pancreas_3views_x2_v3.log # Pancreas 日志"
echo ""
echo "预期完成时间: 约 8-10 小时"
echo "=========================================="
