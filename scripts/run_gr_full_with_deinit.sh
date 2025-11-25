#!/bin/bash

# ==========================================
# GR-Gaussian 完整实现（包含 De-Init 初始化）
# ==========================================
# 完整流程：
#   1. 使用 De-Init 生成降噪初始化点云
#   2. 使用降噪点云进行 GR-Gaussian 训练
#
# 三个核心组件：
#   ✅ De-Init: 高斯滤波降噪初始化 (sigma=3)
#   ✅ Graph Laplacian: 密度平滑正则化
#   ✅ PGA: 增强分割决策（正确实现）
#
# 目标：超越 baseline (PSNR 28.487, SSIM 0.9005)
# ==========================================

set -e  # 遇到错误立即退出

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 数据路径
DATA_PATH="data/369/foot_50_3views.pickle"
DATA_DIR=$(dirname ${DATA_PATH})
DATA_NAME=$(basename ${DATA_PATH} .pickle)

# 输出目录
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_foot_3views_gr_full_deinit_30k"

echo "=========================================="
echo "GR-Gaussian 完整实现（包含 De-Init）"
echo "=========================================="
echo ""

# ==========================================
# 步骤 1：使用 De-Init 生成降噪初始化点云
# ==========================================
echo "步骤 1：使用 De-Init 生成降噪初始化点云"
echo "------------------------------------------"

# 检查是否已存在降噪版本的初始化点云
DEINIT_PLY="${DATA_DIR}/init_${DATA_NAME}_deinit.npy"

if [ -f "${DEINIT_PLY}" ]; then
    echo "✅ 发现已有 De-Init 点云: ${DEINIT_PLY}"
    echo "   跳过初始化步骤，直接使用现有点云"
else
    echo "🔄 生成 De-Init 降噪点云..."
    python initialize_pcd.py \
        --data ${DATA_PATH} \
        --recon_method fdk \
        --n_points 50000 \
        --enable_denoise \
        --denoise_sigma 3.0

    # 重命名以区分
    ORIG_PLY="${DATA_DIR}/init_${DATA_NAME}.npy"
    if [ -f "${ORIG_PLY}" ]; then
        mv "${ORIG_PLY}" "${DEINIT_PLY}"
        echo "✅ De-Init 点云已生成: ${DEINIT_PLY}"
    fi
fi

echo ""

# ==========================================
# 步骤 2：GR-Gaussian 训练
# ==========================================
echo "步骤 2：GR-Gaussian 训练"
echo "------------------------------------------"
echo "数据集: ${DATA_PATH}"
echo "初始化点云: ${DEINIT_PLY}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "GR-Gaussian 配置："
echo "  ✅ De-Init: sigma=3.0"
echo "  ✅ Graph Laplacian: k=6, λ_lap=0.008"
echo "  ✅ PGA (正确实现): λ_g=1e-4"
echo ""
echo "修复说明："
echo "  - PGA 现在作用于 NDC 梯度累积（add_densification_stats）"
echo "  - 使用绝对值: |Δρ_ij| = |ρ_i - ρ_j|"
echo "  - 边权重: exp(-d²/k) 符合论文公式"
echo "=========================================="
echo ""

python train.py \
  --source_path ${DATA_PATH} \
  --ply_path ${DEINIT_PLY} \
  --model_path ${OUTPUT_DIR} \
  --eval \
  --iterations 30000 \
  --gaussiansN 1 \
  --enable_graph_laplacian \
  --graph_k 6 \
  --graph_lambda_lap 0.008 \
  --graph_update_interval 500 \
  --enable_pga \
  --pga_lambda_g 1e-4 \
  --test_iterations 1 5000 10000 20000 30000 \
  --save_iterations 30000

echo ""
echo "=========================================="
echo "训练完成！"
echo "结果目录: ${OUTPUT_DIR}"
echo ""
echo "查看结果:"
echo "  cat ${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"
echo ""
echo "Baseline 对比："
echo "  PSNR: 28.487 dB"
echo "  SSIM: 0.9005"
echo "=========================================="
