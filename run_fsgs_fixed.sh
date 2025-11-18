#!/bin/bash

# ================================================================
# FSGS 修复后的完整训练脚本
# ================================================================
#
# 修复内容：
# 1. ✅ 修复 enhanced_densify_and_prune 的方法绑定错误
# 2. ✅ 修复 original_densify_and_prune 的调用签名
# 3. ✅ 支持通过命令行参数激活 FSGS
# 4. ✅ 添加详细的调试日志
# 5. ✅ 禁用 CoR-GS, SSS, GR（专注 FSGS）
#
# 预期改进：
# - 训练集 PSNR: 54.09 dB → 45.0 dB（减少过拟合）
# - 测试集 PSNR: 28.45 dB → 30.5-33.0 dB（提升泛化）
# - 训练/测试差距: 25.64 dB → 10-15 dB
# ================================================================

# 激活 conda 环境
conda activate r2_gaussian_new

# 设置数据路径
DATA_PATH="data/foot"
OUTPUT_PATH="output/2025_11_18_foot_3views_fsgs_fixed"

# 清除旧的输出（可选）
# rm -rf ${OUTPUT_PATH}

echo "================================================================"
echo "🚀 启动修复后的 FSGS 训练实验"
echo "================================================================"
echo "📁 数据路径: ${DATA_PATH}"
echo "📁 输出路径: ${OUTPUT_PATH}"
echo "🔧 FSGS Proximity-guided Densification: ENABLED"
echo "❌ CoR-GS: DISABLED"
echo "❌ SSS: DISABLED"
echo "❌ GR: DISABLED"
echo "================================================================"
echo ""

# 运行训练
python train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_PATH} \
  --port 6030 \
  --iterations 30000 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  --save_iterations 5000 10000 15000 20000 25000 30000 \
  --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
  --quiet \
  --eval \
  --enable_fsgs_proximity \
  --views 3

echo ""
echo "================================================================"
echo "✅ 训练完成！"
echo "================================================================"
echo "📊 查看结果："
echo "   - 训练日志: ${OUTPUT_PATH}/training.log"
echo "   - TensorBoard: tensorboard --logdir ${OUTPUT_PATH}"
echo "   - 评估结果: ${OUTPUT_PATH}/eval/iter_030000/eval2d_render_test.yml"
echo ""
echo "🔍 检查关键日志："
echo "   - FSGS 初始化: grep 'FSGS集成-优化版' ${OUTPUT_PATH}/training.log"
echo "   - FSGS 执行: grep 'FSGS-Proximity-Optimized' ${OUTPUT_PATH}/training.log"
echo "   - Densify 日志: grep 'Densify' ${OUTPUT_PATH}/training.log"
echo "================================================================"
