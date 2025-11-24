#!/bin/bash

# GR-Gaussian 完整实验：单模型 + Graph Laplacian + PGA
# 数据集：Foot-3 views
# 迭代次数：30000
# 日期：2025-11-24

# 使用环境的 Python 直接运行（绕过 conda activate）
PYTHON_BIN="/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"

# 设置实验参数
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_gr_pga_30k"
DATA_PATH="data/369/foot_50_3views.pickle"
LOG_FILE="gr_pga_foot3_30k.log"

# 训练配置
ITERATIONS=30000
GAUSSIANS_N=1  # 单模型（必须）
ENABLE_COREG=false  # 禁用协同配准（单模型不需要）

# GR-Gaussian 核心参数
ENABLE_GRAPH_LAPLACIAN=true
GRAPH_K=6  # KNN 邻居数量（论文推荐）
GRAPH_LAMBDA_LAP=0.008  # Graph Laplacian 权重（提升 10 倍，之前 0.0008 太小）
GRAPH_UPDATE_INTERVAL=500  # 图更新间隔（诊断报告显示 500 效果好）

# PGA 参数
ENABLE_PGA=true
PGA_LAMBDA_G=0.0001  # PGA 强度系数（论文推荐 1e-4）

# 基础训练参数
DENSIFY_UNTIL=15000
POSITION_LR_INIT=0.0002
DENSITY_LR_INIT=0.01

echo "=========================================="
echo "GR-Gaussian 完整训练（Graph Laplacian + PGA）"
echo "=========================================="
echo "数据集: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "模型数量: $GAUSSIANS_N (单模型)"
echo "迭代次数: $ITERATIONS"
echo ""
echo "GR-Gaussian 配置:"
echo "  - Graph Laplacian: $ENABLE_GRAPH_LAPLACIAN"
echo "  - Graph K: $GRAPH_K"
echo "  - Graph Lambda: $GRAPH_LAMBDA_LAP"
echo "  - Graph Update Interval: $GRAPH_UPDATE_INTERVAL"
echo "  - PGA: $ENABLE_PGA"
echo "  - PGA Lambda: $PGA_LAMBDA_G"
echo "=========================================="
echo ""

# 启动训练
$PYTHON_BIN train.py \
  --source_path "$DATA_PATH" \
  --model_path "$OUTPUT_DIR" \
  --eval \
  --iterations $ITERATIONS \
  --gaussiansN $GAUSSIANS_N \
  --enable_graph_laplacian \
  --graph_k $GRAPH_K \
  --graph_lambda_lap $GRAPH_LAMBDA_LAP \
  --graph_update_interval $GRAPH_UPDATE_INTERVAL \
  --enable_pga \
  --pga_lambda_g $PGA_LAMBDA_G \
  --densify_until_iter $DENSIFY_UNTIL \
  --position_lr_init $POSITION_LR_INIT \
  --density_lr_init $DENSITY_LR_INIT \
  --test_iterations 5000 10000 20000 30000 \
  --save_iterations 30000 \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "训练完成！"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo ""
echo "查看结果："
echo "  cat $OUTPUT_DIR/eval/iter_030000/eval2d_render_test.yml"
echo "=========================================="
