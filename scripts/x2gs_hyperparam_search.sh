#!/bin/bash
# ==============================================================================
# X²-Gaussian TV 正则化超参数搜索脚本
# 目标：找到在 Foot-3 和 Abdomen-9 上都超过 baseline 的最佳 lambda_plane_tv
# ==============================================================================

set -e

# 配置
CONDA_ENV="r2_gaussian_new"
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# 搜索空间
TV_LAMBDAS=(0.0001 0.0005 0.001 0.002 0.005)

# Baseline 参考值 (从消融实验获得)
# Foot-3: 28.7314 dB (baseline), 目标 > 28.4873 dB (SOTA)
# Abdomen-9: 36.9478 dB (baseline), 目标 > 29.2896 dB (SOTA)

# GPU 配置
GPU_FOOT=0
GPU_ABDOMEN=1

# 数据路径
DATA_DIR="data/369"
OUTPUT_BASE="output/x2gs_tv_search"

usage() {
    echo "Usage: $0 [foot|abdomen|both]"
    echo ""
    echo "Arguments:"
    echo "  foot     - Run Foot-3 views experiments on GPU $GPU_FOOT"
    echo "  abdomen  - Run Abdomen-9 views experiments on GPU $GPU_ABDOMEN"
    echo "  both     - Run both datasets in parallel (default)"
    exit 1
}

run_single_experiment() {
    local GPU=$1
    local ORGAN=$2
    local VIEWS=$3
    local TV_LAMBDA=$4
    local DATA_FILE=$5

    # 构建实验名称
    local TV_STR=$(echo $TV_LAMBDA | sed 's/0\.//g' | sed 's/^0*//g')
    if [ -z "$TV_STR" ]; then TV_STR="0"; fi
    local EXP_NAME="${TIMESTAMP}_${ORGAN}_${VIEWS}views_x2_tv${TV_STR}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"

    echo "============================================================"
    echo "[$(date +%H:%M:%S)] Starting: $EXP_NAME"
    echo "  GPU: $GPU"
    echo "  Data: $DATA_FILE"
    echo "  TV Lambda: $TV_LAMBDA"
    echo "  Output: $OUTPUT_DIR"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s "${DATA_DIR}/${DATA_FILE}" \
        -m "$OUTPUT_DIR" \
        --iterations 30000 \
        --gaussiansN 1 \
        --enable_kplanes \
        --kplanes_resolution 64 \
        --kplanes_dim 32 \
        --kplanes_lr_init 0.002 \
        --kplanes_lr_final 0.0002 \
        --lambda_plane_tv $TV_LAMBDA \
        --tv_loss_type l2 \
        --eval \
        2>&1 | tee "${OUTPUT_DIR}.log"

    # 提取结果
    if [ -f "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" ]; then
        PSNR=$(grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        SSIM=$(grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        echo "[RESULT] $EXP_NAME: PSNR=$PSNR, SSIM=$SSIM"
    fi
}

run_foot_experiments() {
    echo "========================================"
    echo "Running Foot-3 views experiments on GPU $GPU_FOOT"
    echo "========================================"

    for TV_LAMBDA in "${TV_LAMBDAS[@]}"; do
        run_single_experiment $GPU_FOOT "foot" 3 $TV_LAMBDA "foot_50_3views.pickle"
    done

    echo ""
    echo "========================================"
    echo "Foot-3 experiments completed!"
    echo "========================================"
}

run_abdomen_experiments() {
    echo "========================================"
    echo "Running Abdomen-9 views experiments on GPU $GPU_ABDOMEN"
    echo "========================================"

    for TV_LAMBDA in "${TV_LAMBDAS[@]}"; do
        run_single_experiment $GPU_ABDOMEN "abdomen" 9 $TV_LAMBDA "abdomen_50_9views.pickle"
    done

    echo ""
    echo "========================================"
    echo "Abdomen-9 experiments completed!"
    echo "========================================"
}

# 激活 conda 环境
echo "Activating conda environment: $CONDA_ENV"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 解析参数
MODE=${1:-both}

case $MODE in
    foot)
        run_foot_experiments
        ;;
    abdomen)
        run_abdomen_experiments
        ;;
    both)
        echo "Starting parallel experiments on both GPUs..."
        run_foot_experiments &
        PID_FOOT=$!
        run_abdomen_experiments &
        PID_ABDOMEN=$!

        echo "Waiting for experiments to complete..."
        echo "  Foot PID: $PID_FOOT"
        echo "  Abdomen PID: $PID_ABDOMEN"

        wait $PID_FOOT
        wait $PID_ABDOMEN

        echo ""
        echo "========================================"
        echo "All experiments completed!"
        echo "========================================"
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "Results saved to: $OUTPUT_BASE"
echo "Run 'python scripts/analyze_x2gs_search.py' to analyze results"
