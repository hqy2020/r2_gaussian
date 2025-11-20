#!/bin/bash

###############################################################################
# Bino 批量训练脚本 - 支持多器官、多视角
#
# 使用方法:
#   # 训练所有器官的 3 views
#   bash scripts/train_bino_batch.sh 3
#
#   # 训练所有器官的 6 views
#   bash scripts/train_bino_batch.sh 6
#
#   # 训练所有器官的 9 views
#   bash scripts/train_bino_batch.sh 9
#
#   # 训练所有器官的所有视角 (3, 6, 9)
#   bash scripts/train_bino_batch.sh all
#
###############################################################################

set -e

# 参数配置
NUM_VIEWS=${1:-3}  # 默认 3 views

# 器官列表（基于你的 SOTA 基准值）
ORGANS=("foot" "chest" "head" "abdomen" "pancreas")

# 视角列表
if [ "$NUM_VIEWS" == "all" ]; then
    VIEWS=(3 6 9)
else
    VIEWS=($NUM_VIEWS)
fi

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   🔬 Bino 批量训练任务                                             ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║   器官列表: ${ORGANS[@]}"
echo "║   视角列表: ${VIEWS[@]}"
echo "║   总任务数: $((${#ORGANS[@]} * ${#VIEWS[@]}))"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# 任务计数
TOTAL_TASKS=$((${#ORGANS[@]} * ${#VIEWS[@]}))
CURRENT_TASK=0
FAILED_TASKS=()
SUCCESSFUL_TASKS=()

# 批量训练
for VIEW in "${VIEWS[@]}"; do
    for ORGAN in "${ORGANS[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📋 [Task $CURRENT_TASK/$TOTAL_TASKS] 器官: $ORGAN | 视角: $VIEW"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # 检查数据集是否存在
        DATA_PATH="data/369/${ORGAN}_50_${VIEW}views.pickle"
        if [ ! -f "$DATA_PATH" ]; then
            echo "⚠️  [Skip] 数据集不存在: $DATA_PATH"
            FAILED_TASKS+=("${ORGAN}_${VIEW}views (数据集不存在)")
            continue
        fi

        # 启动训练
        if bash scripts/train_bino_foot3.sh "$ORGAN" "$VIEW"; then
            echo "✅ [Task $CURRENT_TASK/$TOTAL_TASKS] 完成: ${ORGAN}_${VIEW}views"
            SUCCESSFUL_TASKS+=("${ORGAN}_${VIEW}views")
        else
            echo "❌ [Task $CURRENT_TASK/$TOTAL_TASKS] 失败: ${ORGAN}_${VIEW}views"
            FAILED_TASKS+=("${ORGAN}_${VIEW}views")
        fi

        echo ""
    done
done

# 最终总结
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   🏁 批量训练完成                                                  ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║   总任务数: $TOTAL_TASKS"
echo "║   成功: ${#SUCCESSFUL_TASKS[@]}"
echo "║   失败: ${#FAILED_TASKS[@]}"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

if [ ${#SUCCESSFUL_TASKS[@]} -gt 0 ]; then
    echo "✅ 成功任务:"
    for task in "${SUCCESSFUL_TASKS[@]}"; do
        echo "   • $task"
    done
    echo ""
fi

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "❌ 失败任务:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "   • $task"
    done
    echo ""
fi

# 退出码
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi
