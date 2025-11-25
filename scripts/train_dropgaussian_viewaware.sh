#!/bin/bash
# DropGaussian 视角感知 + 分阶段策略实验
# 实验日期: 2025-11-25
# 改进点:
#   1. 分阶段: warmup 5000 iter 不 drop
#   2. 视角感知: 远离训练视角时降低 drop rate

TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_DIR="data/369"
CONDA_ENV="r2_gaussian_new"

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "=============================================="
echo "DropGaussian ViewAware + Staged 实验"
echo "时间戳: $TIMESTAMP"
echo "=============================================="

# 3 views 实验
run_experiment() {
    local VIEWS=$1
    local ORGAN=$2
    local DATA_FILE="${DATA_DIR}/${ORGAN}_50_${VIEWS}views.pickle"
    local OUTPUT_DIR="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_dropgaussian_viewaware"
    local LOG_FILE="${OUTPUT_DIR}.log"

    echo ""
    echo ">>> 启动实验: ${ORGAN} ${VIEWS} views"
    echo "    数据: ${DATA_FILE}"
    echo "    输出: ${OUTPUT_DIR}"

    python train.py \
        -s "${DATA_FILE}" \
        -m "${OUTPUT_DIR}" \
        --iterations 30000 \
        --use_drop_gaussian \
        --drop_gamma 0.2 \
        --drop_full_iter 10000 \
        --drop_view_aware \
        --drop_warmup_iter 5000 \
        --drop_dist_scale 0.6 \
        --drop_min_factor 0.2 \
        --num_train_views ${VIEWS} \
        --gaussiansN 1 \
        > "${LOG_FILE}" 2>&1 &

    echo "    PID: $!"
    echo "    日志: ${LOG_FILE}"
}

# 启动 Foot 数据集的 3/6/9 views 实验
run_experiment 3 "foot"
sleep 5  # 避免 GPU 竞争

run_experiment 6 "foot"
sleep 5

run_experiment 9 "foot"

echo ""
echo "=============================================="
echo "所有实验已启动!"
echo "使用以下命令监控:"
echo "  tail -f output/${TIMESTAMP}_foot_*views_dropgaussian_viewaware.log"
echo "  nvidia-smi -l 5"
echo "=============================================="

# 等待所有后台任务完成
wait
echo "所有实验完成!"
