#!/bin/bash

# ==========================================
# GR-Gaussian PGA 修复验证（v2）
# ==========================================
# 关键修复：
#   ✅ PGA 正确作用于 NDC 梯度累积
#   ✅ 边权重公式修正为 exp(-d²/k)
#
# 使用已有初始化点云，专注验证 PGA + Graph Laplacian 修复效果
# ==========================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_gr_pga_fixed_v2"

echo "=========================================="
echo "GR-Gaussian PGA 修复验证 v2"
echo "=========================================="
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "修复内容："
echo "  ✅ PGA 作用于 NDC 梯度累积（正确）"
echo "  ✅ 使用绝对值 |Δρ_ij|"
echo "  ✅ 边权重 exp(-d²/k)"
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
echo "训练完成！结果目录: ${OUTPUT_DIR}"
echo "查看结果: cat ${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"
echo "Baseline: PSNR 28.487, SSIM 0.9005"
echo "=========================================="
