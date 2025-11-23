#!/bin/bash
# GR-Gaussian Foot-3 数据集 30k 完整训练实验
#
# 目的：完整验证 GR-Gaussian 性能提升
# 时间：约6-8小时
# 目标性能：PSNR > 28.8 dB, SSIM > 0.905（超越baseline）

DATE=$(date +%Y_%m_%d_%H_%M)
OUTPUT_DIR="output/${DATE}_gr_foot3_30k_final"

echo "========================================="
echo "  GR-Gaussian 30k 完整训练实验"
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
  --enable_graph_laplacian \
  --graph_k 6 \
  --graph_lambda_lap 0.0008 \
  --graph_update_interval 1000 \
  --iterations 30000 \
  --densify_until_iter 15000 \
  --densify_from_iter 500 \
  --densification_interval 100 \
  --eval

echo ""
echo "========================================="
echo "  实验完成"
echo "========================================="
echo "结束时间: $(date)"
echo ""
echo "📊 请检查结果："
echo "   - Tensorboard: tensorboard --logdir $OUTPUT_DIR"
echo "   - 最终指标：查看 $OUTPUT_DIR/results.json"
echo ""
echo "🎯 目标：PSNR > 28.8 dB, SSIM > 0.905"
echo "📌 Baseline：PSNR 28.4873 dB, SSIM 0.9005"
