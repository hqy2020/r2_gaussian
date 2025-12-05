#!/bin/bash
# ============================================================================
# SPAGS 3k 全部 15 个实验并行启动
# ============================================================================
# GPU 0: 8 个实验 (3views 全部 + 6views chest/foot/head)
# GPU 1: 7 个实验 (6views abdomen/pancreas + 9views 全部)
# ============================================================================

set -e

# 取消代理
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
LOG_DIR="output/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================================"
echo "SPAGS 3k 全部 15 个实验并行启动"
echo "============================================================================"
echo "时间戳: $TIMESTAMP"
echo "日志目录: $LOG_DIR"
echo "============================================================================"

# 公共参数
COMMON="--iterations 30000 --test_iterations 10000 20000 30000"
GAR="--enable_binocular_consistency --binocular_loss_weight 0.08 --binocular_max_angle_offset 0.04 --binocular_start_iter 5000 --binocular_warmup_iters 3000 --enable_fsgs_proximity --proximity_threshold 5.0 --proximity_k_neighbors 5 --enable_medical_constraints"
ADM="--enable_kplanes --kplanes_resolution 64 --kplanes_dim 32 --lambda_plane_tv 0.002 --tv_loss_type l2"

# 启动函数
run_exp() {
    local ORGAN=$1
    local VIEWS=$2
    local GPU=$3
    local DATA="data/369/${ORGAN}_50_${VIEWS}views.pickle"
    local PLY="data/369-sps-3k/init_${ORGAN}_50_${VIEWS}views.npy"
    local OUT="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_spags_3k"
    local LOG="${LOG_DIR}/${ORGAN}_${VIEWS}views.log"

    echo ">>> 启动: ${ORGAN}_${VIEWS}views (GPU $GPU)"

    CUDA_VISIBLE_DEVICES=$GPU nohup python train.py \
        -s "$DATA" -m "$OUT" $COMMON $GAR $ADM --ply_path "$PLY" \
        > "$LOG" 2>&1 &

    echo $!
}

# ============================================================================
# GPU 0: 8 个实验
# ============================================================================
echo ""
echo ">>> GPU 0: 启动 8 个实验"

# 3views 全部 (5个)
run_exp chest 3 0
run_exp foot 3 0
run_exp head 3 0
run_exp abdomen 3 0
run_exp pancreas 3 0

# 6views 部分 (3个)
run_exp chest 6 0
run_exp foot 6 0
run_exp head 6 0

# ============================================================================
# GPU 1: 7 个实验
# ============================================================================
echo ""
echo ">>> GPU 1: 启动 7 个实验"

# 6views 部分 (2个)
run_exp abdomen 6 1
run_exp pancreas 6 1

# 9views 全部 (5个)
run_exp chest 9 1
run_exp foot 9 1
run_exp head 9 1
run_exp abdomen 9 1
run_exp pancreas 9 1

echo ""
echo "============================================================================"
echo "全部 15 个实验已启动！"
echo "============================================================================"
echo ""
echo "监控命令:"
echo "  nvidia-smi -l 5"
echo "  ps aux | grep train.py | grep spags_3k | wc -l"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "等待完成后运行汇总:"
echo "  python cc-agent/scripts/summarize_spags_results.py --pattern spags_3k"
echo ""
