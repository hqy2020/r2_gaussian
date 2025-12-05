#!/bin/bash
# ============================================================================
# SPAGS 3k 完整实验批处理
# ============================================================================
# 运行所有 15 个实验 (5器官 × 3视角)
#
# 用法:
#   ./cc-agent/scripts/run_all_spags_3k.sh [模式]
#
# 模式:
#   serial   - 串行运行 (默认, 使用 GPU 0)
#   parallel - 并行运行 (GPU 0 和 GPU 1)
#
# 示例:
#   ./cc-agent/scripts/run_all_spags_3k.sh serial
#   ./cc-agent/scripts/run_all_spags_3k.sh parallel
# ============================================================================

set -e

MODE=${1:-serial}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_spags_3k.sh"

ORGANS=("chest" "foot" "head" "abdomen" "pancreas")
VIEWS=("3" "6" "9")

echo "============================================================================"
echo "SPAGS 3k 完整实验批处理"
echo "============================================================================"
echo "模式: $MODE"
echo "器官: ${ORGANS[*]}"
echo "视角: ${VIEWS[*]}"
echo "总计: $((${#ORGANS[@]} * ${#VIEWS[@]})) 个实验"
echo "============================================================================"
echo ""

if [ "$MODE" = "parallel" ]; then
    # 并行模式: GPU 0 跑3视角，GPU 1 跑6视角，然后9视角
    echo ">>> 并行模式: GPU 0 (3views) + GPU 1 (6views)"
    echo ""

    # 第一轮: 3views (GPU 0) + 6views (GPU 1)
    for organ in "${ORGANS[@]}"; do
        echo "=== 启动: ${organ} 3views (GPU 0) + 6views (GPU 1) ==="

        # 后台运行 3views
        bash "$TRAIN_SCRIPT" "$organ" 3 0 &
        PID_3V=$!

        # 后台运行 6views
        bash "$TRAIN_SCRIPT" "$organ" 6 1 &
        PID_6V=$!

        # 等待两个完成
        wait $PID_3V
        wait $PID_6V

        echo "=== 完成: ${organ} 3views + 6views ==="
        echo ""
    done

    # 第二轮: 9views
    echo ">>> 第二轮: 9views"
    for organ in "${ORGANS[@]}"; do
        echo "=== 启动: ${organ} 9views (GPU 0) ==="
        bash "$TRAIN_SCRIPT" "$organ" 9 0
        echo "=== 完成: ${organ} 9views ==="
    done

else
    # 串行模式
    echo ">>> 串行模式: 使用 GPU 0"
    echo ""

    for view in "${VIEWS[@]}"; do
        for organ in "${ORGANS[@]}"; do
            echo ""
            echo "============================================================"
            echo ">>> 开始: ${organ}_${view}views"
            echo ">>> 时间: $(date)"
            echo "============================================================"

            bash "$TRAIN_SCRIPT" "$organ" "$view" 0

            echo ">>> 完成: ${organ}_${view}views"
            echo ">>> 时间: $(date)"
        done
    done
fi

echo ""
echo "============================================================================"
echo "所有实验完成！"
echo "时间: $(date)"
echo "============================================================================"

# 生成汇总报告
echo ""
echo ">>> 生成结果汇总..."
python cc-agent/scripts/summarize_spags_results.py --pattern "spags_3k"

echo ""
echo ">>> 完成！"
