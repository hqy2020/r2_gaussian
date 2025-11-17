#!/bin/bash
# GR-Gaussian 完整实验脚本
# 包含 4 个核心消融实验
# 生成时间: 2025-11-17
# 负责专家: 深度学习调参与分析专家

set -e  # 遇到错误立即退出

# 激活 conda 环境
source /home/qyhu/anaconda3/bin/activate r2_gaussian_new

# 工作目录
WORK_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian"
cd "$WORK_DIR"

# 日志目录
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ====================================================================
# 实验 1: Baseline 重现
# ====================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实验 1: Baseline 重现${NC}"
echo -e "${GREEN}========================================${NC}"
echo "开始时间: $(date)"
echo "预计时长: 30~35 分钟"

python train.py \
    --source_path data/369 \
    --model_path output/2025_11_17_foot_3views_baseline_rerun \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    2>&1 | tee "$LOG_DIR/exp1_baseline_$TIMESTAMP.log"

echo -e "${GREEN}✓ 实验 1 完成${NC}"
echo "输出目录: output/2025_11_17_foot_3views_baseline_rerun/"
echo ""

# ====================================================================
# 实验 2: GL-Base (标准 Graph Laplacian, λ=8e-4)
# ====================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实验 2: GL-Base (λ_lap=8e-4)${NC}"
echo -e "${GREEN}========================================${NC}"
echo "开始时间: $(date)"
echo "预计时长: 35~40 分钟 (含 Graph 构建开销)"

python train.py \
    --source_path data/369 \
    --model_path output/2025_11_17_foot_3views_gl_base \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    --enable_graph_laplacian \
    --graph_k 6 \
    --graph_lambda_lap 8e-4 \
    2>&1 | tee "$LOG_DIR/exp2_gl_base_$TIMESTAMP.log"

echo -e "${GREEN}✓ 实验 2 完成${NC}"
echo "输出目录: output/2025_11_17_foot_3views_gl_base/"
echo ""

# ====================================================================
# 实验 3: GL-Strong (强正则化, λ=2e-3)
# ====================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实验 3: GL-Strong (λ_lap=2e-3)${NC}"
echo -e "${GREEN}========================================${NC}"
echo "开始时间: $(date)"
echo "预计时长: 35~40 分钟"

python train.py \
    --source_path data/369 \
    --model_path output/2025_11_17_foot_3views_gl_strong \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    --enable_graph_laplacian \
    --graph_k 6 \
    --graph_lambda_lap 2e-3 \
    2>&1 | tee "$LOG_DIR/exp3_gl_strong_$TIMESTAMP.log"

echo -e "${GREEN}✓ 实验 3 完成${NC}"
echo "输出目录: output/2025_11_17_foot_3views_gl_strong/"
echo ""

# ====================================================================
# 实验 4: GL-Weak (弱正则化, λ=2e-4) - 验证下界
# ====================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实验 4: GL-Weak (λ_lap=2e-4)${NC}"
echo -e "${GREEN}========================================${NC}"
echo "开始时间: $(date)"
echo "预计时长: 35~40 分钟"

python train.py \
    --source_path data/369 \
    --model_path output/2025_11_17_foot_3views_gl_weak \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    --enable_graph_laplacian \
    --graph_k 6 \
    --graph_lambda_lap 2e-4 \
    2>&1 | tee "$LOG_DIR/exp4_gl_weak_$TIMESTAMP.log"

echo -e "${GREEN}✓ 实验 4 完成${NC}"
echo "输出目录: output/2025_11_17_foot_3views_gl_weak/"
echo ""

# ====================================================================
# 完成总结
# ====================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "完成时间: $(date)"
echo ""
echo "实验结果目录:"
echo "  1. Baseline:  output/2025_11_17_foot_3views_baseline_rerun/"
echo "  2. GL-Base:   output/2025_11_17_foot_3views_gl_base/"
echo "  3. GL-Strong: output/2025_11_17_foot_3views_gl_strong/"
echo "  4. GL-Weak:   output/2025_11_17_foot_3views_gl_weak/"
echo ""
echo "日志文件:"
echo "  $LOG_DIR/exp*_$TIMESTAMP.log"
echo ""
echo -e "${YELLOW}下一步: 运行结果分析脚本${NC}"
echo "  python scripts/analyze_gr_gaussian_results.py"
