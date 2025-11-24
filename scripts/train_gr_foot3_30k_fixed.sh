#!/bin/bash

# 🚀 GR-Gaussian 修复版 30k 训练
# 修复关键问题：graph_lambda_lap 从 0.008 改为 0.0008（论文推荐值）

# 设置输出目录（带时间戳）
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_gr_lambda_fixed"

echo "=========================================="
echo "🔧 GR-Gaussian 超参数修复版"
echo "=========================================="
echo "关键修复: λ_lap = 0.0008 (论文推荐值)"
echo "输出目录: $OUTPUT_DIR"
echo "预计时间: 6-8 小时"
echo "=========================================="

# 启动训练（使用 conda run 直接在环境中执行）
conda run -n r2_gaussian_new python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path "$OUTPUT_DIR" \
  --enable_graph_laplacian \
  --graph_k 6 \
  --graph_lambda_lap 0.0008 \
  --graph_update_interval 500 \
  --iterations 30000 \
  --position_lr_init 0.0002 \
  --densify_until_iter 15000 \
  --densify_grad_threshold 0.00005 \
  --test_iterations 5000 10000 20000 30000 \
  --save_iterations 30000 \
  --eval \
  2>&1 | tee "${OUTPUT_DIR}.log"

echo ""
echo "✅ 训练完成!"
echo "结果保存在: $OUTPUT_DIR"
echo "日志文件: ${OUTPUT_DIR}.log"
