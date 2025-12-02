#!/bin/bash
# FSGS 超参数并行搜索脚本
# 在两个 GPU 上并行运行 Foot-3 和 Abdomen-9 的搜索实验
# 用法: ./scripts/fsgs_search_parallel.sh [ITERATIONS]

set -e

cd /home/qyhu/Documents/r2_ours/r2_gaussian

ITERATIONS=${1:-10000}
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_DIR="output/fsgs_search/logs_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "FSGS 超参数并行搜索"
echo "迭代次数: ${ITERATIONS}"
echo "日志目录: ${LOG_DIR}"
echo "=============================================="

# 导出迭代次数供子脚本使用
export ITERATIONS

# GPU 0: Foot-3views (场景 0)
echo "启动 GPU 0: Foot-3views 搜索..."
nohup bash scripts/fsgs_hyperparam_search.sh scene 0 0 > "${LOG_DIR}/foot3_search.log" 2>&1 &
PID_FOOT=$!
echo "Foot-3 PID: $PID_FOOT"

# GPU 1: Abdomen-9views (场景 1)
echo "启动 GPU 1: Abdomen-9views 搜索..."
nohup bash scripts/fsgs_hyperparam_search.sh scene 1 1 > "${LOG_DIR}/abdomen9_search.log" 2>&1 &
PID_ABD=$!
echo "Abdomen-9 PID: $PID_ABD"

echo ""
echo "两个搜索任务已在后台启动"
echo ""
echo "监控命令:"
echo "  tail -f ${LOG_DIR}/foot3_search.log"
echo "  tail -f ${LOG_DIR}/abdomen9_search.log"
echo ""
echo "查看结果:"
echo "  cat output/fsgs_search/results.csv"
echo ""
echo "分析结果:"
echo "  python scripts/analyze_fsgs_search.py"
echo ""
echo "进程状态:"
echo "  ps aux | grep fsgs_hyperparam_search"

# 保存 PID 信息
echo "FOOT3_PID=$PID_FOOT" > "${LOG_DIR}/pids.txt"
echo "ABDOMEN9_PID=$PID_ABD" >> "${LOG_DIR}/pids.txt"
echo "START_TIME=$(date)" >> "${LOG_DIR}/pids.txt"
echo "ITERATIONS=$ITERATIONS" >> "${LOG_DIR}/pids.txt"
