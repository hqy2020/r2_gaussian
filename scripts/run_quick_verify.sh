#!/bin/bash
# Full Combo 快速验证脚本 (10k iterations)
# 用法: ./scripts/run_quick_verify.sh <器官> <视角数> <GPU>

set -e

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

ORGAN=${1:-foot}
VIEWS=${2:-3}
GPU=${3:-0}

TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
OUTPUT="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_combo_E_quick"

DATA_PATH="data/369/${ORGAN}_50_${VIEWS}views.pickle"
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据集不存在: $DATA_PATH"
    exit 1
fi

echo "=== Full Combo 快速验证 (10k iterations) ==="
echo "器官: $ORGAN"
echo "视角: ${VIEWS}views"
echo "GPU: $GPU"
echo "输出: $OUTPUT"

mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT" \
    --iterations 10000 \
    --test_iterations 5000 10000 \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --lambda_plane_tv 0.002 \
    --tv_loss_type l2 \
    --enable_fsgs_depth \
    --enable_medical_constraints \
    --depth_pseudo_weight 0.03 \
    --proximity_threshold 5.0 \
    --proximity_k_neighbors 5 \
    --start_sample_pseudo 5000 \
    --enable_binocular_consistency \
    --binocular_max_angle_offset 0.05 \
    --binocular_start_iter 5000 \
    --binocular_warmup_iters 2000 \
    --binocular_loss_weight 0.1 \
    2>&1 | tee "${OUTPUT}/training.log"

echo "快速验证完成！结果: $OUTPUT"
