#!/bin/bash
# 超参数调优实验脚本
# 针对 Bino 和 IX 组合进行调参
# 用法: ./scripts/tuning_experiments.sh <EXP_ID> <GPU>

set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# ============== 公共参数 ==============
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000 --gaussiansN 1"
DATA_FOOT3="data/369/foot_50_3views.pickle"
DATA_FOOT3_DENSITY="data/density-369/foot_50_3views.pickle"

# ============== X²-Gaussian 基础参数 ==============
X_BASE="--enable_kplanes --kplanes_resolution 64"

# ============== 实验定义 ==============
run_experiment() {
    local EXP_NAME=$1
    local DATA_PATH=$2
    local EXTRA_FLAGS=$3
    local GPU=$4

    TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
    OUTPUT="output/tuning/${TIMESTAMP}_foot_3views_${EXP_NAME}"

    echo "=============================================="
    echo "实验: ${EXP_NAME}"
    echo "数据: ${DATA_PATH}"
    echo "输出: ${OUTPUT}"
    echo "GPU: ${GPU}"
    echo "=============================================="

    mkdir -p "$OUTPUT"

    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s "$DATA_PATH" \
        -m "$OUTPUT" \
        $COMMON_FLAGS \
        $EXTRA_FLAGS \
        2>&1 | tee "${OUTPUT}/training.log"

    echo "完成: ${EXP_NAME}"
}

# ============== Bino 调参实验 ==============

# Bino-v2: 降低权重 + 延迟启动
bino_v2() {
    run_experiment "Bino_v2" "$DATA_FOOT3" \
        "--enable_binocular_consistency \
         --binocular_max_angle_offset 0.06 \
         --binocular_start_iter 10000 \
         --binocular_warmup_iters 5000 \
         --binocular_loss_weight 0.08" $1
}

# Bino-v3: 更保守的角度
bino_v3() {
    run_experiment "Bino_v3" "$DATA_FOOT3" \
        "--enable_binocular_consistency \
         --binocular_max_angle_offset 0.04 \
         --binocular_start_iter 7000 \
         --binocular_warmup_iters 5000 \
         --binocular_loss_weight 0.05" $1
}

# Bino-v4: 更晚启动 + 更大角度
bino_v4() {
    run_experiment "Bino_v4" "$DATA_FOOT3" \
        "--enable_binocular_consistency \
         --binocular_max_angle_offset 0.08 \
         --binocular_start_iter 12000 \
         --binocular_warmup_iters 3000 \
         --binocular_loss_weight 0.10" $1
}

# ============== IX 组合调参实验 ==============

# IX-v2: 降低 TV 正则化
ix_v2() {
    run_experiment "IX_v2" "$DATA_FOOT3_DENSITY" \
        "$X_BASE \
         --kplanes_dim 32 \
         --lambda_plane_tv 0.001 \
         --tv_loss_type l2" $1
}

# IX-v3: 更小特征维度
ix_v3() {
    run_experiment "IX_v3" "$DATA_FOOT3_DENSITY" \
        "$X_BASE \
         --kplanes_dim 16 \
         --lambda_plane_tv 0.002 \
         --tv_loss_type l2" $1
}

# ============== IB 组合调参实验 ==============

# IB-v2: 密度初始化 + 优化后的 Bino
ib_v2() {
    run_experiment "IB_v2" "$DATA_FOOT3_DENSITY" \
        "--enable_binocular_consistency \
         --binocular_max_angle_offset 0.06 \
         --binocular_start_iter 10000 \
         --binocular_warmup_iters 5000 \
         --binocular_loss_weight 0.08" $1
}

# ============== 主逻辑 ==============
case "$1" in
    "bino_v2") bino_v2 ${2:-0} ;;
    "bino_v3") bino_v3 ${2:-0} ;;
    "bino_v4") bino_v4 ${2:-0} ;;
    "ix_v2") ix_v2 ${2:-0} ;;
    "ix_v3") ix_v3 ${2:-0} ;;
    "ib_v2") ib_v2 ${2:-0} ;;
    "all_bino")
        echo "运行所有 Bino 调参实验..."
        bino_v2 ${2:-0}
        bino_v3 ${2:-0}
        bino_v4 ${2:-0}
        ;;
    "all_ix")
        echo "运行所有 IX 调参实验..."
        ix_v2 ${2:-0}
        ix_v3 ${2:-0}
        ;;
    "all")
        echo "运行所有调参实验..."
        bino_v2 ${2:-0}
        bino_v3 ${2:-0}
        bino_v4 ${2:-0}
        ix_v2 ${2:-0}
        ix_v3 ${2:-0}
        ib_v2 ${2:-0}
        ;;
    *)
        echo "用法: $0 <EXP_ID> <GPU>"
        echo ""
        echo "实验 ID:"
        echo "  bino_v2   - Bino: weight=0.08, start=10000, warmup=5000"
        echo "  bino_v3   - Bino: weight=0.05, start=7000, angle=0.04"
        echo "  bino_v4   - Bino: weight=0.10, start=12000, angle=0.08"
        echo "  ix_v2     - IX: TV=0.001, dim=32"
        echo "  ix_v3     - IX: TV=0.002, dim=16"
        echo "  ib_v2     - IB: 密度初始化 + Bino-v2 参数"
        echo ""
        echo "批量运行:"
        echo "  all_bino  - 运行所有 Bino 实验"
        echo "  all_ix    - 运行所有 IX 实验"
        echo "  all       - 运行所有实验"
        exit 1
        ;;
esac

echo "实验完成!"
