#!/bin/bash
# R²-Gaussian 组合实验脚本
# 用法: ./scripts/run_combo_experiments.sh <组合> <器官> <视角数> <GPU>
# 示例: ./scripts/run_combo_experiments.sh A foot 3 0

set -e

# 取消所有代理设置
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy
unset NO_PROXY
unset no_proxy
unset ALL_PROXY
unset all_proxy

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

COMBO=$1  # A, B, C, D, E
ORGAN=$2  # foot, chest, head, abdomen, pancreas
VIEWS=$3  # 3, 6, 9
GPU=${4:-0}    # 默认 GPU 0

if [ -z "$COMBO" ] || [ -z "$ORGAN" ] || [ -z "$VIEWS" ]; then
    echo "用法: $0 <组合A/B/C/D/E> <器官> <视角数> [GPU]"
    echo "组合说明:"
    echo "  A: X²-Gaussian (K-Planes 空间调制)"
    echo "  B: X² + FSGS (K-Planes + 深度监督)"
    echo "  C: X² + Bino (K-Planes + 双目一致性)"
    echo "  D: FSGS + Bino (深度监督 + 双目一致性)"
    echo "  E: Full Combo (K-Planes + FSGS + Bino)"
    echo ""
    echo "注: Init-PCD 密度加权采样需要先运行 initialize_pcd.py"
    exit 1
fi

TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
OUTPUT="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_combo_${COMBO}"

# 数据集路径
DATA_PATH="data/369/${ORGAN}_50_${VIEWS}views.pickle"
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据集不存在: $DATA_PATH"
    exit 1
fi

# 公共参数
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000"

# 根据组合选择参数
case $COMBO in
    A)
        echo "=== 组合 A: X²-Gaussian (K-Planes) ==="
        COMBO_FLAGS="--enable_kplanes \
            --kplanes_resolution 64 \
            --kplanes_dim 32 \
            --lambda_plane_tv 0.002 \
            --tv_loss_type l2"
        ;;
    B)
        echo "=== 组合 B: X² + FSGS (K-Planes + 深度监督) ==="
        COMBO_FLAGS="--enable_kplanes \
            --kplanes_resolution 64 \
            --kplanes_dim 32 \
            --lambda_plane_tv 0.002 \
            --tv_loss_type l2 \
            --enable_fsgs_depth \
            --enable_medical_constraints \
            --depth_pseudo_weight 0.03 \
            --proximity_threshold 5.0 \
            --proximity_k_neighbors 5 \
            --start_sample_pseudo 5000"
        ;;
    C)
        echo "=== 组合 C: X² + Bino (K-Planes + 双目一致性) ==="
        COMBO_FLAGS="--enable_kplanes \
            --kplanes_resolution 64 \
            --kplanes_dim 32 \
            --lambda_plane_tv 0.002 \
            --tv_loss_type l2 \
            --enable_binocular_consistency \
            --binocular_max_angle_offset 0.05 \
            --binocular_start_iter 10000 \
            --binocular_warmup_iters 3000 \
            --binocular_loss_weight 0.1"
        ;;
    D)
        echo "=== 组合 D: FSGS + Bino (深度监督 + 双目一致性) ==="
        COMBO_FLAGS="--enable_fsgs_depth \
            --enable_medical_constraints \
            --depth_pseudo_weight 0.05 \
            --enable_binocular_consistency \
            --binocular_max_angle_offset 0.06 \
            --binocular_start_iter 7000 \
            --binocular_loss_weight 0.15"
        ;;
    E)
        echo "=== 组合 E: Full Combo (K-Planes + FSGS + Bino) [修复版] ==="
        COMBO_FLAGS="--gaussiansN 1 \
            --enable_kplanes \
            --kplanes_resolution 64 \
            --kplanes_dim 32 \
            --lambda_plane_tv 0.002 \
            --tv_loss_type l2 \
            --enable_fsgs_depth \
            --enable_fsgs_proximity \
            --enable_medical_constraints \
            --depth_pseudo_weight 0.03 \
            --proximity_threshold 5.0 \
            --proximity_k_neighbors 5 \
            --start_sample_pseudo 5000 \
            --enable_binocular_consistency \
            --binocular_max_angle_offset 0.06 \
            --binocular_start_iter 7000 \
            --binocular_warmup_iters 3000 \
            --binocular_loss_weight 0.15"
        ;;
    *)
        echo "错误: 未知组合 '$COMBO'，请使用 A/B/C/D/E"
        exit 1
        ;;
esac

echo "GPU: $GPU"
echo "数据: $DATA_PATH"
echo "输出: $OUTPUT"
echo "开始训练..."

mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT" \
    $COMMON_FLAGS \
    $COMBO_FLAGS \
    2>&1 | tee "${OUTPUT}/training.log"

echo "训练完成！结果保存在: $OUTPUT"
