#!/bin/bash

# DropGaussian Curriculum Drop - 9 Views 验证实验
# 验证假设：DropGaussian 在 9 views（论文典型场景）下是否有效

# 激活 conda 环境
source ~/anaconda3/bin/activate r2_gaussian_new

# 实验配置
DATASET="data/369/foot_50_9views.pickle"
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
EXP_NAME="${TIMESTAMP}_foot_9views_dropgaussian_curriculum"
OUTPUT_DIR="output/${EXP_NAME}"

echo "=================================================="
echo "DropGaussian Curriculum Drop - 9 Views 验证实验"
echo "=================================================="
echo "数据集: ${DATASET}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "策略配置:"
echo "  - 视角数: 9 views（DropGaussian 论文典型场景）"
echo "  - drop_gamma: 0.1"
echo "  - drop_start_iter: 5000"
echo "  - drop_end_iter: 30000"
echo ""
echo "预期："
echo "  - DropGaussian 应该显著提升性能"
echo "  - 验证论文效果在多视角场景下的有效性"
echo "=================================================="
echo ""

# 启动训练
python3 train.py \
    --source_path "${DATASET}" \
    --model_path "${OUTPUT_DIR}" \
    --iterations 30000 \
    --eval \
    --use_drop_gaussian \
    --drop_gamma 0.1

echo ""
echo "=================================================="
echo "训练完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "=================================================="
