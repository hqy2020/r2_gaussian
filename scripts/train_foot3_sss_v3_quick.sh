#!/bin/bash

# 🎯 SSS v3 快速验证（2000 迭代）- 修复 opacity 参数传递 bug
# 目标：验证 opacity 是否开始学习

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
ORGAN="foot"
VIEWS=3
TECHNIQUE="sss_v3_quick_fix"
OUTPUT_DIR="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_${TECHNIQUE}"

echo "🚀 启动 SSS v3 快速验证（2000 迭代）"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "🎯 关键目标: 验证 opacity 参数是否学习"
echo "----------------------------------------"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate r2_gaussian_new

python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path ${OUTPUT_DIR} \
    --iterations 2000 \
    --test_iterations 1 500 1000 1500 2000 \
    --checkpoint_iterations 2000 \
    --enable_sss \
    --opacity_lr_init 0.005 \
    --nu_lr_init 0.001 \
    --opacity_reg_weight 0.0 \
    --opacity_threshold 0.005 \
    --max_recycle_ratio 0.05 \
    --densify_until_iter 1500 \
    2>&1 | tee ${OUTPUT_DIR}/train.log

echo ""
echo "✅ 训练完成！"
echo ""
echo "🔍 关键验证点："
echo "1. 检查 opacity 是否从 [0.5, 0.5] 变化："
grep "SSS-Official" ${OUTPUT_DIR}/train.log | tail -3
echo ""
echo "2. 最终性能（目标 PSNR > 17 dB @ iter 2000）："
grep "ITER 2000" ${OUTPUT_DIR}/train.log
echo ""
echo "📊 完整日志: ${OUTPUT_DIR}/train.log"

