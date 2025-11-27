#!/bin/bash
# X²-Gaussian 对齐实验脚本
# 日期：2025-01-23
# 目标：验证 P0 修改后的 K-Planes + TV 正则化效果

set -e  # 遇到错误立即退出

# 配置参数
DATASET="data/369/foot_50_3views.pickle"
OUTPUT_DIR="output/2025_01_23_foot_3views_x2_aligned"
ITERATIONS=30000
LAMBDA_PLANE_TV=0.0002  # TV 正则化权重

echo "======================================================================"
echo "🚀 X²-Gaussian 对齐实验"
echo "======================================================================"
echo "数据集: ${DATASET}"
echo "输出目录: ${OUTPUT_DIR}"
echo "训练迭代: ${ITERATIONS}"
echo "K-Planes TV 权重: ${LAMBDA_PLANE_TV}"
echo ""
echo "关键修改（对齐 X²-Gaussian 原版）："
echo "  1. K-Planes 初始化: xavier → uniform(0.1, 0.5)"
echo "  2. TV 损失公式: L1+mean → L2+sum/count"
echo "  3. K-Planes 学习率: 0.00016 → 0.002 (12.5×)"
echo "  4. TV 损失类型: l1 → l2"
echo "======================================================================"
echo ""

# 激活环境
echo "📦 激活 conda 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 验证环境
echo "✓ Python: $(which python)"
echo "✓ CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# 检查数据集
if [ ! -f "${DATASET}" ]; then
    echo "❌ 错误：数据集不存在 ${DATASET}"
    exit 1
fi
echo "✓ 数据集存在"
echo ""

# 运行训练
echo "🏃 开始训练..."
echo "预计时间：约 1-2 小时（30K iterations）"
echo ""

python train.py \
    -s "${DATASET}" \
    -m "${OUTPUT_DIR}" \
    --iterations ${ITERATIONS} \
    --test_iterations ${ITERATIONS} \
    --enable_kplanes \
    --lambda_plane_tv ${LAMBDA_PLANE_TV} \
    --quiet

echo ""
echo "======================================================================"
echo "✅ 训练完成！"
echo "======================================================================"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "📊 查看结果："
echo "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}"
echo "  评估指标: cat ${OUTPUT_DIR}/eval/results.txt"
echo ""
echo "🎯 成功标准（Foot-3 views baseline: PSNR=28.49, SSIM=0.9005）："
echo "  - 最低要求: PSNR ≥ 28.49"
echo "  - 理想目标: PSNR > 28.7"
echo "======================================================================"
