#!/bin/bash
# ==============================================================================
# K-Planes 修复后的完整实验脚本
#
# 修复内容：
# 1. ✅ 让 K-Planes 特征参与渲染（调制 density）
# 2. ✅ 启用 TV 正则化（lambda_plane_tv=0.0002）
# 3. ✅ 增强日志输出
#
# 作者：Claude Code Agent
# 日期：2025-01-19
# ==============================================================================

set -e

echo "========================================================================"
echo "🎯 K-Planes 修复后的实验"
echo "========================================================================"
echo "修复问题："
echo "  1. K-Planes 特征现在会调制 density（修复渲染集成）"
echo "  2. TV 正则化已启用（lambda_plane_tv=0.0002）"
echo "  3. 增强了日志输出（可以看到 TV loss 和 K-Planes 诊断信息）"
echo ""
echo "预期结果："
echo "  - 日志会显示 'K-Planes Encoder 已启用'"
echo "  - 进度条会显示 'tv_kp' (K-Planes TV loss)"
echo "  - 前 3 个迭代会输出 K-Planes 特征诊断信息"
echo "  - PSNR 应该 >= 28.49 (baseline) 或更高"
echo "========================================================================"

# 配置参数
CONDA_ENV="r2_gaussian_new"
DATA_PATH="data/foot_3views"
TIMESTAMP=$(date +%Y_%m_%d_%H%M%S)
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_kplanes_FIXED"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/kplanes_fixed_${TIMESTAMP}.log"

# 训练参数
ITERATIONS=30000
TEST_ITERATIONS=30000

# K-Planes 参数
KPLANES_RESOLUTION=64
KPLANES_DIM=32

# TV 正则化参数
LAMBDA_PLANE_TV=0.0002

# 创建日志目录
mkdir -p ${LOG_DIR}

# 激活 conda 环境
echo "激活环境: ${CONDA_ENV}"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

echo ""
echo "训练配置："
echo "  - 数据: ${DATA_PATH}"
echo "  - 输出: ${OUTPUT_DIR}"
echo "  - 日志: ${LOG_FILE}"
echo "  - 迭代: ${ITERATIONS}"
echo "  - K-Planes 分辨率: ${KPLANES_RESOLUTION}"
echo "  - K-Planes 特征维度: ${KPLANES_DIM}"
echo "  - TV 正则化系数: ${LAMBDA_PLANE_TV}"
echo ""
echo "========================================================================"
echo "开始训练..."
echo "========================================================================"

# 启动训练
python train.py \
  -s ${DATA_PATH} \
  -m ${OUTPUT_DIR} \
  --enable_kplanes \
  --kplanes_resolution ${KPLANES_RESOLUTION} \
  --kplanes_dim ${KPLANES_DIM} \
  --lambda_plane_tv ${LAMBDA_PLANE_TV} \
  --iterations ${ITERATIONS} \
  --test_iterations ${TEST_ITERATIONS} \
  2>&1 | tee ${LOG_FILE}

echo ""
echo "========================================================================"
echo "训练完成！"
echo "========================================================================"
echo "日志文件: ${LOG_FILE}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "查看 TensorBoard："
echo "  tensorboard --logdir ${OUTPUT_DIR} --port 6006"
echo ""
echo "检查关键指标："
echo "  grep 'K-Planes' ${LOG_FILE} | head -20"
echo "  grep 'tv_kp' ${LOG_FILE} | tail -10"
echo "  grep 'Evaluating' ${LOG_FILE}"
echo "========================================================================"
