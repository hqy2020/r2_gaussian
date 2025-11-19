#!/bin/bash

# ==============================================================================
# K-Planes + TV 正则化训练脚本（Foot 3 views）
# 用途：X²-Gaussian 创新点验证实验
# 作者：AI Assistant
# 日期：2025-01-19
# ==============================================================================

set -e  # 遇到错误立即退出

# 配置参数
CONDA_ENV="r2_gaussian_new"
DATA_PATH="data/foot_3views"
TIMESTAMP=$(date +%Y_%m_%d_%H%M%S)
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_kplanes_tv"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/train_kplanes_foot3_${TIMESTAMP}.log"

# 训练参数
ITERATIONS=30000
TEST_ITERATIONS=30000

# K-Planes 参数
ENABLE_KPLANES=true
KPLANES_RESOLUTION=64
KPLANES_DIM=32

# TV 正则化参数
LAMBDA_PLANE_TV=0.0002

# 打印配置信息
echo "=========================================="
echo "  K-Planes + TV 训练脚本"
echo "=========================================="
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "训练轮数: ${ITERATIONS}"
echo "K-Planes 分辨率: ${KPLANES_RESOLUTION}"
echo "K-Planes 特征维度: ${KPLANES_DIM}"
echo "TV 正则化系数: ${LAMBDA_PLANE_TV}"
echo "=========================================="

# 创建日志目录
mkdir -p ${LOG_DIR}

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo "开始训练..."
echo "日志保存到: ${LOG_FILE}"

# 启动训练（后台运行）
nohup python train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_DIR} \
  --enable_kplanes \
  --kplanes_resolution ${KPLANES_RESOLUTION} \
  --kplanes_dim ${KPLANES_DIM} \
  --lambda_plane_tv ${LAMBDA_PLANE_TV} \
  --iterations ${ITERATIONS} \
  --test_iterations ${TEST_ITERATIONS} \
  > ${LOG_FILE} 2>&1 &

TRAIN_PID=$!

echo "训练任务已启动！"
echo "进程 PID: ${TRAIN_PID}"
echo ""
echo "监控训练进度："
echo "  tail -f ${LOG_FILE}"
echo ""
echo "检查进程状态："
echo "  ps -p ${TRAIN_PID}"
echo ""
echo "停止训练："
echo "  kill ${TRAIN_PID}"
echo "=========================================="

# 等待一段时间后显示初始日志
sleep 5
echo ""
echo "初始日志输出："
tail -20 ${LOG_FILE}
