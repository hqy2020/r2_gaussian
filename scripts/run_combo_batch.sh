#!/bin/bash
# R²-Gaussian 批量组合实验脚本
# Phase 1: 快速验证 (10k iterations) - 筛选有效组合
# 用法: ./scripts/run_combo_batch.sh phase1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PHASE=${1:-phase1}

case $PHASE in
    phase1)
        echo "=== Phase 1: 快速验证 (10k iterations) ==="
        # GPU 0: 组合 A (Foot, Chest, Pancreas)
        # GPU 1: 组合 B (Foot, Chest, Pancreas)

        # 修改迭代数为 10k
        ITERS="--iterations 10000 --test_iterations 5000 10000"

        # GPU 0 任务 (后台运行)
        echo "启动 GPU 0 任务: 组合 A"
        nohup bash -c "
            for organ in foot chest pancreas; do
                ./scripts/run_combo_experiments.sh A \$organ 3 0
            done
        " > logs/phase1_gpu0.log 2>&1 &

        # GPU 1 任务 (后台运行)
        echo "启动 GPU 1 任务: 组合 B"
        nohup bash -c "
            for organ in foot chest pancreas; do
                ./scripts/run_combo_experiments.sh B \$organ 3 1
            done
        " > logs/phase1_gpu1.log 2>&1 &

        echo "Phase 1 已启动！监控日志:"
        echo "  tail -f logs/phase1_gpu0.log"
        echo "  tail -f logs/phase1_gpu1.log"
        ;;

    phase2)
        echo "=== Phase 2: 完整训练 (30k iterations) ==="
        # 需要根据 Phase 1 结果选择最佳组合

        ORGANS="foot chest head abdomen pancreas"
        VIEWS="3 6 9"

        # 示例：运行组合 A 的完整实验
        for organ in $ORGANS; do
            for view in $VIEWS; do
                echo "训练: 组合 A - $organ - ${view}views"
                ./scripts/run_combo_experiments.sh A "$organ" "$view" 0
            done
        done
        ;;

    quick_test)
        echo "=== 快速测试: 单个组合 ==="
        # 仅测试 Foot 3views，确保脚本正确
        ./scripts/run_combo_experiments.sh A foot 3 0
        ;;

    *)
        echo "用法: $0 <phase1|phase2|quick_test>"
        echo "  phase1: 快速验证 (10k，筛选有效组合)"
        echo "  phase2: 完整训练 (30k，选定组合)"
        echo "  quick_test: 快速测试 (单个组合)"
        exit 1
        ;;
esac
