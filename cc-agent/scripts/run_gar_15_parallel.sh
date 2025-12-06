#!/bin/bash
# ============================================================================
# GAR 方法 15 个实验并行启动
# ============================================================================
# 验证 GAR（几何感知细化）在所有 5器官 × 3视角 场景下的效果
# GPU 0: 8 个实验 (3views 全部 + 6views 前3个)
# GPU 1: 7 个实验 (6views 后2个 + 9views 全部)
# ============================================================================

set -e

# 取消代理
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
LOG_DIR="output/logs_gar_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================================"
echo "GAR 方法 15 个实验并行启动"
echo "============================================================================"
echo "时间戳: $TIMESTAMP"
echo "日志目录: $LOG_DIR"
echo "============================================================================"

# 启动函数（后台运行）
run_gar() {
    local ORGAN=$1
    local VIEWS=$2
    local GPU=$3
    local LOG="${LOG_DIR}/${ORGAN}_${VIEWS}views.log"

    echo ">>> 启动: ${ORGAN}_${VIEWS}views (GPU $GPU)"
    nohup ./cc-agent/scripts/run_spags_ablation.sh gar $ORGAN $VIEWS $GPU \
        > "$LOG" 2>&1 &
    echo $!
}

# ============================================================================
# GPU 0: 8 个实验
# ============================================================================
echo ""
echo ">>> GPU 0: 启动 8 个实验"

# 3views 全部 (5个)
run_gar foot 3 0
run_gar chest 3 0
run_gar head 3 0
run_gar abdomen 3 0
run_gar pancreas 3 0

# 6views 部分 (3个)
run_gar foot 6 0
run_gar chest 6 0
run_gar head 6 0

# ============================================================================
# GPU 1: 7 个实验
# ============================================================================
echo ""
echo ">>> GPU 1: 启动 7 个实验"

# 6views 部分 (2个)
run_gar abdomen 6 1
run_gar pancreas 6 1

# 9views 全部 (5个)
run_gar foot 9 1
run_gar chest 9 1
run_gar head 9 1
run_gar abdomen 9 1
run_gar pancreas 9 1

echo ""
echo "============================================================================"
echo "全部 15 个 GAR 实验已启动！"
echo "============================================================================"
echo ""
echo "监控命令:"
echo "  nvidia-smi -l 5"
echo "  ps aux | grep train.py | grep gar | wc -l"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "等待完成后运行汇总:"
echo "  python cc-agent/scripts/summarize_spags_results.py --pattern gar"
echo ""
