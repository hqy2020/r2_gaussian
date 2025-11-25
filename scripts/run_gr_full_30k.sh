#!/bin/bash

# ==========================================
# GR-Gaussian 完整实现（修复版）
# ==========================================
# 修复内容：
#   1. PGA 现在正确作用于 NDC 梯度累积（用于 ADC 分割决策）
#      公式: g_i += λ_g · Σ(|Δρ_ij|) / k
#   2. 边权重公式修正为论文公式: w_ij = exp(-d²/k)
#   3. 启用 De-Init 降噪初始化
#
# 三个核心组件：
#   ✅ De-Init: 高斯滤波降噪初始化 (sigma=3)
#   ✅ Graph Laplacian: 密度平滑正则化
#   ✅ PGA: 增强分割决策（正确实现）
#
# 目标：超越 baseline (PSNR 28.487, SSIM 0.9005)
# ==========================================

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 输出目录
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_gr_full_30k"

echo "=========================================="
echo "GR-Gaussian 完整实现（修复版）"
echo "=========================================="
echo "数据集: data/369/foot_50_3views.pickle"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型数量: 1 (单模型)"
echo "迭代次数: 30000"
echo ""
echo "GR-Gaussian 三大组件："
echo "  ✅ De-Init: sigma=3.0"
echo "  ✅ Graph Laplacian: k=6, λ_lap=0.008"
echo "  ✅ PGA (正确实现): λ_g=1e-4"
echo ""
echo "修复说明："
echo "  - PGA 现在作用于 NDC 梯度累积（add_densification_stats）"
echo "  - 使用绝对值: |Δρ_ij| = |ρ_i - ρ_j|"
echo "  - 边权重: exp(-d²/k) 而非 exp(-d/sigma)"
echo "=========================================="
echo ""

python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path ${OUTPUT_DIR} \
  --eval \
  --iterations 30000 \
  --gaussiansN 1 \
  --enable_denoise \
  --denoise_sigma 3.0 \
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
echo ""
echo "Baseline 对比："
echo "  PSNR: 28.487 dB"
echo "  SSIM: 0.9005"
echo "=========================================="
