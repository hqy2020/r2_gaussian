#!/bin/bash

# ==========================================
# 方案 A: GR-Gaussian 训练（禁用 PGA）
# ==========================================
# 目标：验证 PGA 是导致崩溃的根本原因
# 配置：单模型 + Graph Laplacian（无 PGA）
# 预期：PSNR 27-28 dB（接近 12:15 实验）
# ==========================================

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 输出目录
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_gr_no_pga_30k"

echo "=========================================="
echo "方案 A: GR-Gaussian 训练（禁用 PGA）"
echo "=========================================="
echo "数据集: data/369/foot_50_3views.pickle"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型数量: 1 (单模型)"
echo "迭代次数: 30000"
echo ""
echo "GR-Gaussian 配置:"
echo "  - Graph Laplacian: true"
echo "  - Graph K: 6"
echo "  - Graph Lambda: 0.008"
echo "  - Graph Update Interval: 500"
echo "  - PGA: FALSE ✅"
echo "=========================================="
echo ""

python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path ${OUTPUT_DIR} \
  --eval \
  --iterations 30000 \
  --gaussiansN 1 \
  --enable_graph_laplacian \
  --graph_k 6 \
  --graph_lambda_lap 0.008 \
  --graph_update_interval 500 \
  --test_iterations 1 5000 10000 20000 30000 \
  --save_iterations 30000

echo ""
echo "=========================================="
echo "训练完成！"
echo "结果目录: ${OUTPUT_DIR}"
echo "查看结果: cat ${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"
echo "=========================================="
