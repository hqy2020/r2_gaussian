#!/bin/bash
# =============================================================================
# 批量训练脚本：5 种方法 × 15 场景 = 75 个任务
# 使用 5 个并行进程，2 个 GPU 交替分配
# =============================================================================

# 不使用 set -e，因为 (()) 操作符可能返回非零

# 配置
METHODS=("sps" "adm" "gar" "sps_adm" "sps_gar" "gar_adm" "spags")
ORGANS=("chest" "foot" "head" "abdomen" "pancreas")
VIEWS=(3 6 9)
MAX_JOBS=7
GPUS=(0 1)

# 脚本路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
ABLATION_SCRIPT="${SCRIPT_DIR}/run_spags_ablation.sh"

# 日志目录
LOG_DIR="${ROOT_DIR}/batch_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 状态文件
STATUS_FILE="${LOG_DIR}/status.txt"
PROGRESS_FILE="${LOG_DIR}/progress.txt"

echo "=============================================="
echo "批量训练启动"
echo "=============================================="
echo "方法: ${METHODS[*]}"
echo "器官: ${ORGANS[*]}"
echo "视角: ${VIEWS[*]}"
echo "总任务数: $((${#METHODS[@]} * ${#ORGANS[@]} * ${#VIEWS[@]}))"
echo "最大并行数: $MAX_JOBS"
echo "GPU 列表: ${GPUS[*]}"
echo "日志目录: $LOG_DIR"
echo "=============================================="

# 切换到根目录
cd "$ROOT_DIR"

# 任务计数
total_tasks=$((${#METHODS[@]} * ${#ORGANS[@]} * ${#VIEWS[@]}))
completed_tasks=0
gpu_idx=0

# 启动时间
start_time=$(date +%s)

# 记录进度函数
log_progress() {
    local current=$1
    local total=$2
    local method=$3
    local organ=$4
    local view=$5
    local gpu=$6
    local status=$7

    local elapsed=$(($(date +%s) - start_time))
    local hours=$((elapsed / 3600))
    local mins=$(((elapsed % 3600) / 60))

    echo "[$(date '+%H:%M:%S')] [$current/$total] $status: $method $organ ${view}views (GPU $gpu) - 已用时: ${hours}h${mins}m" | tee -a "$PROGRESS_FILE"
}

# 等待可用槽位
wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done
}

# 运行单个任务
run_task() {
    local method=$1
    local organ=$2
    local view=$3
    local gpu=$4
    local task_num=$5

    local log_file="${LOG_DIR}/${method}_${organ}_${view}views.log"

    # 启动训练
    "$ABLATION_SCRIPT" "$method" "$organ" "$view" "$gpu" > "$log_file" 2>&1

    # 记录完成状态
    if [ $? -eq 0 ]; then
        echo "DONE: $method $organ ${view}views" >> "$STATUS_FILE"
    else
        echo "FAIL: $method $organ ${view}views" >> "$STATUS_FILE"
    fi
}

# 主循环
task_num=0
for method in "${METHODS[@]}"; do
    for organ in "${ORGANS[@]}"; do
        for view in "${VIEWS[@]}"; do
            # 等待可用槽位
            wait_for_slot

            # 分配 GPU（轮询）
            gpu=${GPUS[$((gpu_idx % ${#GPUS[@]}))]}
            ((gpu_idx++))

            # 任务计数
            ((task_num++))

            # 记录启动
            log_progress $task_num $total_tasks "$method" "$organ" "$view" "$gpu" "Started"

            # 后台启动任务
            run_task "$method" "$organ" "$view" "$gpu" "$task_num" &

            # 短暂延迟避免竞争
            sleep 2
        done
    done
done

# 等待所有任务完成
echo ""
echo "所有任务已提交，等待完成..."
wait

# 统计结果
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
mins=$(((total_time % 3600) / 60))

echo ""
echo "=============================================="
echo "批量训练完成!"
echo "=============================================="
echo "总耗时: ${hours}h${mins}m"
echo "成功: $(grep -c "^DONE" "$STATUS_FILE" 2>/dev/null || echo 0)"
echo "失败: $(grep -c "^FAIL" "$STATUS_FILE" 2>/dev/null || echo 0)"
echo "日志目录: $LOG_DIR"
echo "=============================================="

# 显示失败任务
if grep -q "^FAIL" "$STATUS_FILE" 2>/dev/null; then
    echo ""
    echo "失败的任务:"
    grep "^FAIL" "$STATUS_FILE"
fi
