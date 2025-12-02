#!/bin/bash
# ==============================================================================
# X²-Gaussian TV 正则化超参数搜索 - 并行启动脚本
#
# 使用方法:
#   nohup bash scripts/x2gs_search_parallel.sh &> output/x2gs_tv_search/search.log &
#
# 功能:
#   - 在 GPU 0 上运行 Foot-3 views 实验
#   - 在 GPU 1 上运行 Abdomen-9 views 实验
#   - 两组实验并行执行
# ==============================================================================

set -e

# 配置
CONDA_ENV="r2_gaussian_new"
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# TV Lambda 搜索空间
TV_LAMBDAS=(0.0001 0.0005 0.001 0.002 0.005)

# 数据路径
DATA_DIR="data/369"
OUTPUT_BASE="output/x2gs_tv_search"

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 激活 conda 环境
echo "[$(date)] Activating conda environment: $CONDA_ENV"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "[$(date)] Starting X²-Gaussian TV Lambda search"
echo "  Timestamp: $TIMESTAMP"
echo "  TV Lambdas: ${TV_LAMBDAS[*]}"
echo "  Output: $OUTPUT_BASE"
echo ""

# GPU 0: Foot-3 views 实验
run_foot_experiments() {
    GPU=0
    ORGAN="foot"
    VIEWS=3
    DATA_FILE="foot_50_3views.pickle"

    echo "[$(date)] [GPU $GPU] Starting Foot-3 experiments..."

    for TV_LAMBDA in "${TV_LAMBDAS[@]}"; do
        # 构建实验名称 (移除小数点)
        TV_STR=$(printf "%.4f" $TV_LAMBDA | sed 's/0\.//g' | sed 's/^0*//g')
        if [ -z "$TV_STR" ]; then TV_STR="0"; fi
        EXP_NAME="${TIMESTAMP}_${ORGAN}_${VIEWS}views_x2_tv${TV_STR}"
        OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"

        echo ""
        echo "[$(date)] [GPU $GPU] ========================================"
        echo "[$(date)] [GPU $GPU] Starting: $EXP_NAME"
        echo "[$(date)] [GPU $GPU] TV Lambda: $TV_LAMBDA"
        echo "[$(date)] [GPU $GPU] ========================================"

        CUDA_VISIBLE_DEVICES=$GPU python train.py \
            -s "${DATA_DIR}/${DATA_FILE}" \
            -m "$OUTPUT_DIR" \
            --iterations 30000 \
            --test_iterations 5000 10000 20000 30000 \
            --save_iterations 30000 \
            --gaussiansN 1 \
            --enable_kplanes \
            --kplanes_resolution 64 \
            --kplanes_dim 32 \
            --kplanes_lr_init 0.002 \
            --kplanes_lr_final 0.0002 \
            --lambda_plane_tv $TV_LAMBDA \
            --tv_loss_type l2 \
            --eval \
            2>&1 | tee "${OUTPUT_BASE}/${EXP_NAME}.log"

        # 提取结果
        if [ -f "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" ]; then
            PSNR=$(grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
            SSIM=$(grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
            echo "[$(date)] [GPU $GPU] [RESULT] $EXP_NAME: PSNR=$PSNR, SSIM=$SSIM"
        fi
    done

    echo "[$(date)] [GPU $GPU] Foot-3 experiments completed!"
}

# GPU 1: Abdomen-9 views 实验
run_abdomen_experiments() {
    GPU=1
    ORGAN="abdomen"
    VIEWS=9
    DATA_FILE="abdomen_50_9views.pickle"

    echo "[$(date)] [GPU $GPU] Starting Abdomen-9 experiments..."

    for TV_LAMBDA in "${TV_LAMBDAS[@]}"; do
        # 构建实验名称 (移除小数点)
        TV_STR=$(printf "%.4f" $TV_LAMBDA | sed 's/0\.//g' | sed 's/^0*//g')
        if [ -z "$TV_STR" ]; then TV_STR="0"; fi
        EXP_NAME="${TIMESTAMP}_${ORGAN}_${VIEWS}views_x2_tv${TV_STR}"
        OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"

        echo ""
        echo "[$(date)] [GPU $GPU] ========================================"
        echo "[$(date)] [GPU $GPU] Starting: $EXP_NAME"
        echo "[$(date)] [GPU $GPU] TV Lambda: $TV_LAMBDA"
        echo "[$(date)] [GPU $GPU] ========================================"

        CUDA_VISIBLE_DEVICES=$GPU python train.py \
            -s "${DATA_DIR}/${DATA_FILE}" \
            -m "$OUTPUT_DIR" \
            --iterations 30000 \
            --test_iterations 5000 10000 20000 30000 \
            --save_iterations 30000 \
            --gaussiansN 1 \
            --enable_kplanes \
            --kplanes_resolution 64 \
            --kplanes_dim 32 \
            --kplanes_lr_init 0.002 \
            --kplanes_lr_final 0.0002 \
            --lambda_plane_tv $TV_LAMBDA \
            --tv_loss_type l2 \
            --eval \
            2>&1 | tee "${OUTPUT_BASE}/${EXP_NAME}.log"

        # 提取结果
        if [ -f "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" ]; then
            PSNR=$(grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
            SSIM=$(grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
            echo "[$(date)] [GPU $GPU] [RESULT] $EXP_NAME: PSNR=$PSNR, SSIM=$SSIM"
        fi
    done

    echo "[$(date)] [GPU $GPU] Abdomen-9 experiments completed!"
}

# 并行启动两组实验
echo "[$(date)] Launching parallel experiments..."

run_foot_experiments &
PID_FOOT=$!

run_abdomen_experiments &
PID_ABDOMEN=$!

echo "[$(date)] Started processes:"
echo "  Foot-3 (GPU 0): PID $PID_FOOT"
echo "  Abdomen-9 (GPU 1): PID $PID_ABDOMEN"

# 等待完成
wait $PID_FOOT
echo "[$(date)] Foot-3 experiments finished"

wait $PID_ABDOMEN
echo "[$(date)] Abdomen-9 experiments finished"

echo ""
echo "[$(date)] ========================================"
echo "[$(date)] All X²-Gaussian TV search experiments completed!"
echo "[$(date)] ========================================"
echo ""
echo "Run the following command to analyze results:"
echo "  python scripts/analyze_x2gs_search.py"
