#!/bin/bash

# ==========================================
# 方案 B + C: GR-Gaussian 完整训练（修复版）
# ==========================================
# 修复内容：
#   - 方案 B: _prune_optimizer 保留梯度
#   - 方案 C: PGA 时序调整（densification 后跳过 PGA）
# 配置：单模型 + Graph Laplacian + PGA（全部启用）
# 预期：PSNR 28-29 dB（超越 baseline）
# ==========================================

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 输出目录
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_gr_pga_fixed_30k"

echo "=========================================="
echo "方案 B+C: GR-Gaussian 训练（修复版）"
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
echo "  - PGA: TRUE ✅"
echo "  - PGA Lambda: 1e-4"
echo ""
echo "修复内容:"
echo "  ✅ 方案 B: _prune_optimizer 保留梯度"
echo "  ✅ 方案 C: PGA 时序调整"
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
  --enable_pga \
  --pga_lambda_g 1e-4 \
  --test_iterations 1 5000 10000 20000 30000 \
  --save_iterations 30000

echo ""
echo "=========================================="
echo "训练完成！"
echo "结果目录: ${OUTPUT_DIR}"
echo "查看结果: cat ${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"
echo "=========================================="
