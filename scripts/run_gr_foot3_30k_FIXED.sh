#!/bin/bash
# 🔧 GR-Gaussian 修复版 - Foot-3 数据集 30k 完整训练
#
# 🎯 修复内容：
#   1. gaussiansN=1 (单模型，与 baseline 公平对比)
#   2. graph_update_interval=500 (从1000降低，预期+0.78 dB)
#   3. graph_lambda_lap=0.008 (从0.0008提升10倍，增强Graph Laplacian效果)
#   4. enable_denoise=true (启用 De-Init 降噪初始化，预期+0.2-0.5 dB)
#   5. denoise_sigma=3 (论文推荐值)
#
# 📊 预期性能：
#   目标: PSNR > 28.8 dB, SSIM > 0.910
#   提升: +0.5~1.5 dB (相比之前失败的实验)
#
# ⏱️ 预计时间：6-8小时
# 📌 Baseline: PSNR 28.547 dB, SSIM 0.9008 (单模型)

DATE=$(date +%Y_%m_%d_%H_%M)
OUTPUT_DIR="output/${DATE}_gr_foot3_30k_FIXED"

echo "========================================="
echo "  🔧 GR-Gaussian 修复版 30k 训练"
echo "========================================="
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo ""
echo "🎯 关键修复："
echo "   ✓ 单模型 (gaussiansN=1)"
echo "   ✓ graph_update_interval=500"
echo "   ✓ graph_lambda_lap=0.008 (10x)"
echo "   ✓ enable_denoise=true"
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
echo ""
echo "📊 查看结果："
echo "   - 最终评估: cat $OUTPUT_DIR/eval/iter_030000/eval2d_render_test.yml"
echo "   - Tensorboard: tensorboard --logdir $OUTPUT_DIR"
echo ""
echo "🎯 性能目标："
echo "   - PSNR > 28.8 dB (超越 baseline 28.547 dB)"
echo "   - SSIM > 0.910 (超越 baseline 0.9008)"
echo ""
echo "📈 预期提升："
echo "   - graph_update_interval=500: +0.78 dB"
echo "   - De-Init: +0.2~0.5 dB"
echo "   - lambda_lap 10x: +0.1~0.3 dB"
echo "   - 总计: +1.0~1.5 dB"
