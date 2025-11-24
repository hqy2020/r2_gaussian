#!/bin/bash
# 🚀 GR-Gaussian 快速修复版 - Foot-3 数据集 30k 训练
#
# 🎯 修复内容：
#   1. gaussiansN=1 (单模型，与 baseline 公平对比)
#   2. graph_update_interval=500 (从1000降低，预期+0.78 dB)
#   3. graph_lambda_lap=0.008 (从0.0008提升10倍)
#
# 📊 预期性能：PSNR > 28.8 dB, SSIM > 0.905
# ⏱️ 预计时间：6-8小时
# 📌 Baseline: PSNR 28.547 dB, SSIM 0.9008

DATE=$(date +%Y_%m_%d_%H_%M)
OUTPUT_DIR="output/${DATE}_gr_foot3_30k_quick"

echo "========================================="
echo "  🚀 GR-Gaussian 快速修复版 30k 训练"
echo "========================================="
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo ""

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 运行训练
python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path "$OUTPUT_DIR" \
  --gaussiansN 1 \
  --enable_graph_laplacian \
  --graph_k 6 \
  --graph_lambda_lap 0.008 \
  --graph_update_interval 500 \
  --iterations 30000 \
  --densify_until_iter 15000 \
  --densify_from_iter 500 \
  --densification_interval 100 \
  --test_iterations 5000 10000 20000 30000 \
  --save_iterations 30000 \
  --eval

echo ""
echo "========================================="
echo "  ✅ 实验完成"
echo "========================================="
echo "结束时间: $(date)"
echo "📊 查看结果: cat $OUTPUT_DIR/eval/iter_030000/eval2d_render_test.yml"
