#!/bin/bash
# ============================================================================
# 并行运行所有 15 个 Baseline 实验（全部同时启动）
# 5 器官 × 3 视角 = 15 实验
# 2 GPU 轮流分配
# ============================================================================

set -e

cd /home/qyhu/Documents/r2_ours/r2_gaussian

SCRIPT="./cc-agent/scripts/run_spags_ablation.sh"
ORGANS=("foot" "chest" "head" "abdomen" "pancreas")
VIEWS=("3" "6" "9")

echo "============================================================================"
echo "同时启动 15 个 Baseline 实验"
echo "器官: ${ORGANS[*]}"
echo "视角: ${VIEWS[*]}"
echo "GPU: 0, 1 (轮流分配)"
echo "============================================================================"
echo ""

# 收集所有 PID
PIDS=()
GPU=0
TASK_NUM=0

for organ in "${ORGANS[@]}"; do
    for view in "${VIEWS[@]}"; do
        TASK_NUM=$((TASK_NUM + 1))
        LOG_FILE="output/baseline_${organ}_${view}views.log"

        echo "[$(date +%H:%M:%S)] 启动任务 ${TASK_NUM}/15: baseline $organ ${view}views on GPU $GPU"

        # 后台启动任务
        nohup bash "$SCRIPT" baseline "$organ" "$view" "$GPU" > "$LOG_FILE" 2>&1 &
        PIDS+=($!)

        # 轮流切换 GPU
        GPU=$(( (GPU + 1) % 2 ))
    done
done

echo ""
echo "============================================================================"
echo "所有 15 个任务已启动！PIDs: ${PIDS[*]}"
echo "============================================================================"
echo ""

# 等待所有任务完成
echo "等待所有任务完成..."
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    wait "$pid" 2>/dev/null
    echo "[$(date +%H:%M:%S)] 任务 $((i+1))/15 完成 (PID: $pid)"
done

echo ""
echo "============================================================================"
echo "所有 15 个 Baseline 实验完成！"
echo "============================================================================"
