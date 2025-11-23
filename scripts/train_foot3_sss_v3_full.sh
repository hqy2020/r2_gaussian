#!/bin/bash

# 🎯 SSS v3 完整训练（30000 迭代）- 修复 opacity 参数后的完整验证
# 目标：PSNR > 28.8 dB, SSIM > 0.905

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
ORGAN="foot"
VIEWS=3
TECHNIQUE="sss_v3_full"
OUTPUT_DIR="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_${TECHNIQUE}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 SSS v3 完整训练启动"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "🎯 目标性能: PSNR > 28.8 dB, SSIM > 0.905"
echo "⏱️  预计时间: 6-8 小时"
echo "🔧 核心技术: Student's t + Signed Opacity + Component Recycling"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate r2_gaussian_new

# 检查修复是否存在
if grep -q "opacity if opacity is not None else density" r2_gaussian/gaussian/render_query.py; then
    echo "✅ render_query.py 修复已应用"
else
    echo "❌ 警告：render_query.py 可能未修复！"
    exit 1
fi

echo ""
echo "🎬 开始训练..."
echo ""

python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path ${OUTPUT_DIR} \
    --iterations 30000 \
    --test_iterations 1 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 10000 20000 30000 \
    --checkpoint_iterations 10000 20000 30000 \
    --enable_sss \
    --opacity_lr_init 0.005 \
    --opacity_lr_final 0.0005 \
    --nu_lr_init 0.001 \
    --nu_lr_final 0.0001 \
    --opacity_reg_weight 0.0 \
    --opacity_threshold 0.005 \
    --max_recycle_ratio 0.05 \
    --densify_until_iter 15000 \
    2>&1 | tee ${OUTPUT_DIR}.log

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成！"
else
    echo "❌ 训练失败（退出码: $EXIT_CODE）"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 性能总结："
echo ""
grep "ITER 30000.*Evaluating" ${OUTPUT_DIR}.log | tail -1
echo ""
echo "🎯 Opacity 学习情况："
grep "SSS-Official.*Iter 30000" ${OUTPUT_DIR}.log | tail -1
echo ""
echo "📈 关键里程碑："
grep "ITER.*Evaluating" ${OUTPUT_DIR}.log | grep -E "(1000|5000|10000|20000|30000)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 完整日志: ${OUTPUT_DIR}.log"
echo "📁 模型输出: ${OUTPUT_DIR}/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

