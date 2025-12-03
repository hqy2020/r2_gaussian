#!/bin/bash
# ==============================================================================
# X²-Gaussian 重新训练脚本
# TV Lambda=0.0001 实验 (需要重新训练，无checkpoint)
# ==============================================================================

set -e

CONDA_ENV="r2_gaussian_new"
DATA_DIR="data/369"
OUTPUT_BASE="output/x2gs_tv_search"
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "[$(date)] 重新启动 X²-Gaussian TV Lambda=0.0001 训练"

# GPU 0: Foot-3 views
run_foot() {
    OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_foot_3views_x2_tv1"
    echo "[$(date)] [GPU 0] Foot-3 views 训练..."

    CUDA_VISIBLE_DEVICES=0 python train.py \
        -s "${DATA_DIR}/foot_50_3views.pickle" \
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
        --lambda_plane_tv 0.0001 \
        --tv_loss_type l2 \
        --eval \
        2>&1 | tee "${OUTPUT_DIR}.log"

    if [ -f "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" ]; then
        PSNR=$(grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        SSIM=$(grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        echo "[$(date)] [GPU 0] [完成] Foot-3: PSNR=$PSNR, SSIM=$SSIM"
        echo "Foot-3 (TV=0.0001): PSNR=$PSNR, SSIM=$SSIM" >> "${OUTPUT_BASE}/results_summary.txt"
    fi
}

# GPU 1: Abdomen-9 views
run_abdomen() {
    OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_abdomen_9views_x2_tv1"
    echo "[$(date)] [GPU 1] Abdomen-9 views 训练..."

    CUDA_VISIBLE_DEVICES=1 python train.py \
        -s "${DATA_DIR}/abdomen_50_9views.pickle" \
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
        --lambda_plane_tv 0.0001 \
        --tv_loss_type l2 \
        --eval \
        2>&1 | tee "${OUTPUT_DIR}.log"

    if [ -f "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" ]; then
        PSNR=$(grep "psnr_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        SSIM=$(grep "ssim_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
        echo "[$(date)] [GPU 1] [完成] Abdomen-9: PSNR=$PSNR, SSIM=$SSIM"
        echo "Abdomen-9 (TV=0.0001): PSNR=$PSNR, SSIM=$SSIM" >> "${OUTPUT_BASE}/results_summary.txt"
    fi
}

# 并行启动
run_foot &
PID_FOOT=$!
run_abdomen &
PID_ABDOMEN=$!

echo "[$(date)] 启动进程: Foot=$PID_FOOT, Abdomen=$PID_ABDOMEN"

wait $PID_FOOT
wait $PID_ABDOMEN

echo "[$(date)] ========================================"
echo "[$(date)] X²-Gaussian TV=0.0001 训练完成!"
echo "[$(date)] ========================================"
echo ""
echo "结果汇总:"
cat "${OUTPUT_BASE}/results_summary.txt" 2>/dev/null || echo "无结果"
