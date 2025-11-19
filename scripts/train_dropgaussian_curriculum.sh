#!/bin/bash

# DropGaussian Curriculum Drop 实验
# 严格遵循论文，动态 drop rate，最大值 0.1

# 激活 conda 环境
source ~/anaconda3/bin/activate r2_gaussian_new

# 实验配置
DATASET="data/369/foot_50_3views.pickle"
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
EXP_NAME="${TIMESTAMP}_foot_3views_dropgaussian_curriculum"
OUTPUT_DIR="output/${EXP_NAME}"

echo "=================================================="
echo "DropGaussian Curriculum Drop 实验"
echo "=================================================="
echo "数据集: ${DATASET}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "策略配置:"
echo "  - drop_gamma (最大值): 0.1"
echo "  - drop_start_iter: 5000 (前期稳定训练)"
echo "  - drop_end_iter: 30000"
echo "  - 前 5000 轮: drop_rate = 0"
echo "  - 5000-30000 轮: 线性增长到 0.1"
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
