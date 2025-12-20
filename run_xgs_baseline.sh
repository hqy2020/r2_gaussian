#!/bin/bash
# 运行 xgaussian 和 baseline 各 15 场景，2 个并行进程

METHODS=("xgaussian" "baseline")
ORGANS=("chest" "foot" "head" "abdomen" "pancreas")
VIEWS=(3 6 9)
MAX_JOBS=2
GPUS=(0 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/batch_logs_xgs_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "启动 xgaussian + baseline 训练"
echo "=============================================="
echo "总任务数: 30 (2 方法 × 15 场景)"
echo "并行进程: 2"
echo "日志目录: $LOG_DIR"
echo "=============================================="

cd "$SCRIPT_DIR"

task_num=0
gpu_idx=0

for method in "${METHODS[@]}"; do
    for organ in "${ORGANS[@]}"; do
        for view in "${VIEWS[@]}"; do
            # 等待可用槽位
            while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
                sleep 10
            done

            gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}
            gpu_idx=$((gpu_idx + 1))
            task_num=$((task_num + 1))

            log_file="${LOG_DIR}/${method}_${organ}_${view}views.log"
            
            echo "[$(date '+%H:%M:%S')] [$task_num/30] Started: $method $organ ${view}views (GPU $gpu)"
            
            ./cc-agent/scripts/run_spags_ablation.sh "$method" "$organ" "$view" "$gpu" > "$log_file" 2>&1 &
            
            sleep 2
        done
    done
done

echo ""
echo "所有任务已提交，等待完成..."
wait

echo ""
echo "=============================================="
echo "训练完成！"
echo "日志目录: $LOG_DIR"
echo "=============================================="
