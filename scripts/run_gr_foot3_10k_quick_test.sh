#!/bin/bash
# 🚀 GR-Gaussian 快速验证 - 10k iterations
#
# 用途：快速验证修复方案是否有效（2-3小时）
# 如果 10k 结果好（PSNR ≥ 28.5），再运行 30k 完整训练

DATE=$(date +%Y_%m_%d_%H_%M)
OUTPUT_DIR="output/${DATE}_gr_foot3_10k_quick_test"

echo "========================================="
echo "  🚀 GR-Gaussian 快速验证 (10k)"
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
  --enable_denoise \
  --denoise_sigma 3.0 \
  --iterations 10000 \
  --densify_until_iter 8000 \
  --densify_from_iter 500 \
  --densification_interval 100 \
  --test_iterations 5000 10000 \
  --save_iterations 10000 \
  --eval

echo ""
echo "========================================="
echo "  ✅ 快速验证完成"
echo "========================================="
echo "结束时间: $(date)"
echo ""
echo "📊 查看结果："
echo "   cat $OUTPUT_DIR/eval/iter_010000/eval2d_render_test.yml | grep 'psnr_2d\|ssim_2d' | head -2"
echo ""
echo "🎯 判断标准："
echo "   ✓ PSNR ≥ 28.5 dB → 运行 30k 完整训练"
echo "   ✗ PSNR < 28.5 dB → 需要进一步调整超参数"
