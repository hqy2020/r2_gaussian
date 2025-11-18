#!/bin/bash

###############################################################################
# SSS-v7-OFFICIAL 训练启动脚本（防中断版）
#
# 使用 nohup 在后台运行，防止 shell 关闭导致训练中断
# 生成日期: 2025-11-18
###############################################################################

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  🚀 启动 SSS-v7-OFFICIAL 训练（防中断模式）                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📂 项目路径: $PROJECT_ROOT"
echo "⏰ 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 激活 conda 环境
echo "🔧 激活 conda 环境: r2_gaussian_new"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 检查环境
echo "✅ Python: $(which python)"
echo "✅ Conda env: $CONDA_DEFAULT_ENV"
echo ""

# 日志文件路径
NOHUP_LOG="output/2025_11_18_foot_3views_sss_v7_official_nohup.log"
PID_FILE="output/2025_11_18_foot_3views_sss_v7_official.pid"

# 检查是否已有训练在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  检测到已有训练进程在运行 (PID: $OLD_PID)"
        echo "   请先停止旧进程，或删除 PID 文件: $PID_FILE"
        exit 1
    else
        echo "🧹 清理旧的 PID 文件"
        rm -f "$PID_FILE"
    fi
fi

# 启动训练（后台运行）
echo "🚀 启动训练..."
echo "   日志文件: $NOHUP_LOG"
echo "   PID 文件: $PID_FILE"
echo ""

nohup bash scripts/train_foot3_sss_v7_official.sh > "$NOHUP_LOG" 2>&1 &
TRAIN_PID=$!

# 保存 PID
echo $TRAIN_PID > "$PID_FILE"

echo "✅ 训练已启动！"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 训练信息"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔢 进程 PID: $TRAIN_PID"
echo "📄 日志文件: $NOHUP_LOG"
echo "⏱️  预计耗时: 8-10 小时（30,000 步）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 监控命令："
echo ""
echo "  # 实时查看日志（推荐）"
echo "  tail -f $NOHUP_LOG"
echo ""
echo "  # 查看训练进度"
echo "  grep 'Train:' $NOHUP_LOG | tail -20"
echo ""
echo "  # 查看 SSS Balance Loss"
echo "  grep 'SSS-Official' $NOHUP_LOG"
echo ""
echo "  # 检查进程状态"
echo "  ps -p $TRAIN_PID -o pid,etime,%cpu,%mem,cmd"
echo ""
echo "  # 停止训练"
echo "  kill $TRAIN_PID"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 等待 3 秒，检查进程是否正常启动
sleep 3

if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "✅ 训练进程正常运行中..."
    echo ""
    echo "💡 提示："
    echo "   - 可以安全关闭当前终端，训练会继续在后台运行"
    echo "   - 使用 'tail -f $NOHUP_LOG' 随时查看进度"
    echo "   - 关键检查点: iter 1000, 7000, 15000, 30000"
    echo ""
else
    echo "❌ 训练进程启动失败！请检查日志: $NOHUP_LOG"
    rm -f "$PID_FILE"
    exit 1
fi
