#!/bin/bash
# R²-Gaussian 并行消融实验启动器
# 在多个 GPU 上并行运行实验
# 用法: ./scripts/ablation_parallel.sh

set -e

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 创建日志目录
LOG_DIR="output/ablation/logs_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "R²-Gaussian 消融实验 - 并行启动"
echo "=============================================="
echo "实验矩阵: 16 配置 × 3 场景 = 48 实验"
echo "日志目录: $LOG_DIR"
echo ""

# 检查可用 GPU
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

# 定义任务分配
# GPU 0: 场景 0 (Foot-3views) 的 16 个配置
# GPU 1: 场景 1 (Chest-6views) 的 16 个配置
# GPU 2: 场景 2 (Abdomen-9views) 的 16 个配置 (如果有第三个 GPU)

echo "启动方案:"
echo "  GPU 0: Foot-3views (16 配置)"
echo "  GPU 1: Chest-6views (16 配置)"
echo "  等待这 32 个完成后再跑 Abdomen-9views"
echo ""

read -p "确认启动? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "取消"
    exit 0
fi

# 启动 GPU 0 - Foot-3views
echo "启动 GPU 0: Foot-3views..."
nohup bash scripts/ablation_study.sh scene 0 0 > "$LOG_DIR/gpu0_foot3.log" 2>&1 &
PID0=$!
echo "  PID: $PID0"
echo "$PID0" > "$LOG_DIR/gpu0.pid"

# 启动 GPU 1 - Chest-6views
echo "启动 GPU 1: Chest-6views..."
nohup bash scripts/ablation_study.sh scene 1 1 > "$LOG_DIR/gpu1_chest6.log" 2>&1 &
PID1=$!
echo "  PID: $PID1"
echo "$PID1" > "$LOG_DIR/gpu1.pid"

echo ""
echo "=============================================="
echo "已启动 2 个 GPU 并行运行"
echo "=============================================="
echo "监控命令:"
echo "  tail -f $LOG_DIR/gpu0_foot3.log"
echo "  tail -f $LOG_DIR/gpu1_chest6.log"
echo ""
echo "查看进度:"
echo "  ls -la output/ablation/ | grep -E '(foot|chest|abdomen)' | wc -l"
echo ""
echo "等待完成后运行 Abdomen-9views:"
echo "  bash scripts/ablation_study.sh scene 2 0"
echo ""

# 保存实验信息
cat > "$LOG_DIR/experiment_info.txt" << EOF
消融实验启动信息
================
启动时间: $(date)
实验数量: 48 (16 配置 × 3 场景)

场景定义:
  0: Foot-3views (最稳定测试场景)
  1: Chest-6views (低对比度软组织)
  2: Abdomen-9views (高对比度腹部)

配置定义:
  0: Baseline (无技术)
  1: I (Init-PCD 密度加权)
  2: X (X²-Gaussian K-Planes)
  3: F (FSGS 深度监督)
  4: B (Bino 双目一致性)
  5-15: 各种组合

GPU 分配:
  GPU 0: 场景 0 (PID: $PID0)
  GPU 1: 场景 1 (PID: $PID1)

预计耗时:
  单个实验: ~35-40 分钟
  单场景 16 配置: ~10 小时
  总计: ~20 小时 (2 GPU 并行)
EOF

echo "实验信息已保存到: $LOG_DIR/experiment_info.txt"
