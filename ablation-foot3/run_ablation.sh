#!/bin/bash
# SPS n_points ablation experiment
source /home/qyhu/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new
cd /home/qyhu/Documents/r2_ours/r2_gaussian

TIMESTAMP="2025_12_04_15_50"
echo "Starting 5 parallel training experiments at $TIMESTAMP"

# GPU 0: 25k, 50k, 125k
CUDA_VISIBLE_DEVICES=0 nohup python train.py -s data/369/foot_50_3views.pickle \
  -m ablation-foot3/${TIMESTAMP}_npoints_25000 \
  --ply_path ablation-foot3/init_foot_25000.npy \
  --iterations 30000 > ablation-foot3/train_25k.log 2>&1 &
echo "Started 25k on GPU 0, PID: $!"

CUDA_VISIBLE_DEVICES=0 nohup python train.py -s data/369/foot_50_3views.pickle \
  -m ablation-foot3/${TIMESTAMP}_npoints_50000 \
  --ply_path ablation-foot3/init_foot_50000.npy \
  --iterations 30000 > ablation-foot3/train_50k.log 2>&1 &
echo "Started 50k on GPU 0, PID: $!"

CUDA_VISIBLE_DEVICES=0 nohup python train.py -s data/369/foot_50_3views.pickle \
  -m ablation-foot3/${TIMESTAMP}_npoints_125000 \
  --ply_path ablation-foot3/init_foot_125000.npy \
  --iterations 30000 > ablation-foot3/train_125k.log 2>&1 &
echo "Started 125k on GPU 0, PID: $!"

# GPU 1: 75k, 100k
CUDA_VISIBLE_DEVICES=1 nohup python train.py -s data/369/foot_50_3views.pickle \
  -m ablation-foot3/${TIMESTAMP}_npoints_75000 \
  --ply_path ablation-foot3/init_foot_75000.npy \
  --iterations 30000 > ablation-foot3/train_75k.log 2>&1 &
echo "Started 75k on GPU 1, PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup python train.py -s data/369/foot_50_3views.pickle \
  -m ablation-foot3/${TIMESTAMP}_npoints_100000 \
  --ply_path ablation-foot3/init_foot_100000.npy \
  --iterations 30000 > ablation-foot3/train_100k.log 2>&1 &
echo "Started 100k on GPU 1, PID: $!"

echo "All 5 training jobs started!"
sleep 3
ps aux | grep "train.py.*npoints" | grep -v grep
