#!/bin/bash
# 运行 baseline 15 场景，全部在 GPU 0

ORGANS=("chest" "foot" "head" "abdomen" "pancreas")
VIEWS=(3 6 9)
GPU=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/batch_logs_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[baseline] 开始训练 15 场景 (GPU $GPU)"
cd "$SCRIPT_DIR"

task_num=0
for organ in "${ORGANS[@]}"; do
    for view in "${VIEWS[@]}"; do
        task_num=$((task_num + 1))
        log_file="${LOG_DIR}/baseline_${organ}_${view}views.log"
        echo "[$(date '+%H:%M:%S')] [$task_num/15] baseline $organ ${view}views"
        ./cc-agent/scripts/run_spags_ablation.sh baseline "$organ" "$view" "$GPU" > "$log_file" 2>&1
    done
done
echo "[baseline] 全部完成！"
