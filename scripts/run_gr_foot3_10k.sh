#!/bin/bash
# GR-Gaussian Foot-3 数据集 10k 快速验证实验
#
# 目的：快速验证 GR-Gaussian 实现正确性
# 时间：约2-3小时
# 验证目标：PSNR ≥ 28.5 dB（接近或超过baseline 28.49）

DATE=$(date +%Y_%m_%d_%H_%M)
OUTPUT_DIR="output/${DATE}_gr_foot3_10k_test"

echo "========================================="
echo "  GR-Gaussian 10k 快速验证实验"
echo "========================================="
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo ""

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 运行训练
python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path "$OUTPUT_DIR" \
  --enable_graph_laplacian \
  --graph_k 6 \
  --graph_lambda_lap 0.0008 \
  --graph_update_interval 1000 \
  --iterations 10000 \
  --densify_until_iter 5000 \
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
echo "   - 日志文件: $OUTPUT_DIR/train.log"
echo ""
echo "✅ 如果PSNR ≥ 28.5 dB，则可以进行30k完整训练"
echo "❌ 如果PSNR < 28.0 dB，需要调试或调整参数"
