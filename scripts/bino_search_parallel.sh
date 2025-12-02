#!/bin/bash
# Bino 超参数搜索 - 并行启动器
# 在 2 个 GPU 上同时跑 Foot-3 和 Abdomen-9
# 用法: ./scripts/bino_search_parallel.sh

set -e

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 创建搜索目录
SEARCH_DIR="output/bino_search_$(date +%Y%m%d)"
mkdir -p "$SEARCH_DIR"

echo "=============================================="
echo "Bino 超参数搜索 - 并行启动"
echo "=============================================="
echo "搜索空间: 18 组参数"
echo "场景: Foot-3views (GPU 0) + Abdomen-9views (GPU 1)"
echo "预计时间: 10-11 小时"
echo "输出目录: $SEARCH_DIR"
echo "=============================================="
echo ""

# 检查 GPU
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

read -p "确认启动? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "取消"
    exit 0
fi

# 启动 GPU 0 - Foot-3views
echo "启动 GPU 0: Foot-3views..."
nohup bash scripts/bino_hyperparam_search.sh foot 0 > "$SEARCH_DIR/foot_3views.log" 2>&1 &
PID0=$!
echo "  PID: $PID0"

# 启动 GPU 1 - Abdomen-9views
echo "启动 GPU 1: Abdomen-9views..."
nohup bash scripts/bino_hyperparam_search.sh abdomen 1 > "$SEARCH_DIR/abdomen_9views.log" 2>&1 &
PID1=$!
echo "  PID: $PID1"

# 保存 PID
echo "$PID0" > "$SEARCH_DIR/foot.pid"
echo "$PID1" > "$SEARCH_DIR/abdomen.pid"

echo ""
echo "=============================================="
echo "已启动 2 个并行搜索"
echo "=============================================="
echo ""
echo "监控命令:"
echo "  tail -f $SEARCH_DIR/foot_3views.log"
echo "  tail -f $SEARCH_DIR/abdomen_9views.log"
echo ""
echo "查看进度:"
echo "  ls $SEARCH_DIR/ | grep -E 'foot|abdomen' | wc -l"
echo ""
echo "查看结果:"
echo "  cat $SEARCH_DIR/foot_3views_results.csv"
echo "  cat $SEARCH_DIR/abdomen_9views_results.csv"
echo ""
echo "分析结果 (完成后运行):"
echo "  python scripts/analyze_bino_search.py $SEARCH_DIR"
echo ""
