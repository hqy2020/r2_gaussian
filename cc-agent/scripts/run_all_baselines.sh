#!/bin/bash
# ============================================================================
# 批量基线实验脚本 - 5 方法 × 15 场景
# ============================================================================
# 用法:
#   ./cc-agent/scripts/run_all_baselines.sh
#
# 任务分配:
#   进程 1: naf      (GPU 0) - 15 场景
#   进程 2: saxnerf  (GPU 1) - 15 场景
#   进程 3: tensorf  (GPU 0) - 15 场景
#   进程 4: xgaussian(GPU 1) - 15 场景
#   进程 5: baseline (GPU 0) - 15 场景 (r2gs)
#
# 输出目录: output/baselines_comparison/<method>/
# ============================================================================

set -e

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 输出前缀
OUTPUT_PREFIX="baselines_comparison"
mkdir -p "output/${OUTPUT_PREFIX}"

# 日志目录
LOG_DIR="output/${OUTPUT_PREFIX}/logs"
mkdir -p "$LOG_DIR"

# 5 个方法和对应的 GPU 分配
METHODS=("naf" "saxnerf" "tensorf" "xgaussian" "baseline")
GPUS=(0 1 0 1 0)
METHOD_NAMES=("NAF" "SAX-NeRF" "TensoRF" "X-Gaussian" "R2-Gaussian")

# 5 个器官
ORGANS=("foot" "chest" "head" "abdomen" "pancreas")

# 3 种视角
VIEWS=(3 6 9)

# 打印实验概览
echo ""
echo "============================================================================"
echo "批量基线实验"
echo "============================================================================"
echo "方法: ${METHODS[*]}"
echo "GPU:  ${GPUS[*]}"
echo "器官: ${ORGANS[*]}"
echo "视角: ${VIEWS[*]}"
echo "总实验数: $((${#METHODS[@]} * ${#ORGANS[@]} * ${#VIEWS[@]})) (5×5×3=75)"
echo "输出目录: output/${OUTPUT_PREFIX}/"
echo "============================================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始实验..."

# 为每个方法启动后台进程
PIDS=()
for i in {0..4}; do
    METHOD=${METHODS[$i]}
    GPU=${GPUS[$i]}
    METHOD_NAME=${METHOD_NAMES[$i]}
    LOG_FILE="${LOG_DIR}/${METHOD}.log"

    echo "[进程 $((i+1))] 启动 $METHOD_NAME (GPU $GPU) -> $LOG_FILE"

    (
        echo "============================================================================"
        echo "$METHOD_NAME 实验 (GPU $GPU)"
        echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================================"

        for ORGAN in "${ORGANS[@]}"; do
            for VIEW in "${VIEWS[@]}"; do
                echo ""
                echo ">>> [$METHOD] $ORGAN ${VIEW}views (GPU $GPU)"
                echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"

                # 调用 run_spags_ablation.sh
                # 参数: <配置> <器官> <视角> <GPU> <输出前缀>
                ./cc-agent/scripts/run_spags_ablation.sh \
                    "$METHOD" "$ORGAN" "$VIEW" "$GPU" "${OUTPUT_PREFIX}/${METHOD}" \
                    || echo "[错误] $METHOD $ORGAN ${VIEW}views 失败"

                echo "<<< [$METHOD] $ORGAN ${VIEW}views 完成"
            done
        done

        echo ""
        echo "============================================================================"
        echo "$METHOD_NAME 全部完成"
        echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================================================"
    ) > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "所有进程已启动 (PIDs: ${PIDS[*]})"
echo "日志文件:"
for i in {0..4}; do
    echo "  - ${METHODS[$i]}: ${LOG_DIR}/${METHODS[$i]}.log"
done
echo ""
echo "使用以下命令查看实时日志:"
echo "  tail -f ${LOG_DIR}/*.log"
echo ""
echo "使用以下命令查看进程状态:"
echo "  ps aux | grep run_spags_ablation"
echo ""

# 等待所有进程完成
echo "等待所有实验完成..."
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    METHOD=${METHODS[$i]}
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[完成] ${METHOD} (PID: $PID)"
    else
        echo "[失败] ${METHOD} (PID: $PID, 退出码: $EXIT_CODE)"
    fi
done

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================================"
echo "所有实验完成！"
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "输出目录: output/${OUTPUT_PREFIX}/"
echo "============================================================================"

# 生成结果汇总
echo ""
echo "生成结果汇总..."
SUMMARY_FILE="output/${OUTPUT_PREFIX}/summary.txt"
{
    echo "============================================================================"
    echo "基线实验结果汇总"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "============================================================================"
    echo ""

    for METHOD in "${METHODS[@]}"; do
        echo "=== $METHOD ==="
        METHOD_DIR="output/${OUTPUT_PREFIX}/${METHOD}"
        if [ -d "$METHOD_DIR" ]; then
            # 查找该方法的所有输出目录
            find "$METHOD_DIR" -maxdepth 1 -type d -name "*views*" | sort | while read -r dir; do
                DIRNAME=$(basename "$dir")
                # 检查是否有测试结果
                if [ -f "$dir/test_results.json" ]; then
                    echo "  $DIRNAME: 完成 (有测试结果)"
                elif [ -f "$dir/training.log" ]; then
                    echo "  $DIRNAME: 完成 (训练日志存在)"
                else
                    echo "  $DIRNAME: 目录存在"
                fi
            done
        else
            echo "  (无输出)"
        fi
        echo ""
    done
} > "$SUMMARY_FILE"

echo "结果汇总已保存到: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
