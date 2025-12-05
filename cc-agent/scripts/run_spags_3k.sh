#!/bin/bash
# ============================================================================
# SPAGS 3k 消融实验训练脚本
# ============================================================================
# 用法:
#   ./cc-agent/scripts/run_spags_3k.sh <器官> <视角数> [GPU]
#
# 示例:
#   ./cc-agent/scripts/run_spags_3k.sh foot 3 0
#   ./cc-agent/scripts/run_spags_3k.sh chest 6 1
#
# 配置:
#   - SPS: 3k 密度加权点云
#   - GAR: 5 邻居点, threshold=5.0
#   - ADM: lambda_tv=0.002
#   - 训练: 30k iterations
# ============================================================================

set -e

# 取消代理设置
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 参数解析
ORGAN=$1     # foot/chest/head/abdomen/pancreas
VIEWS=$2     # 3/6/9
GPU=${3:-0}  # 默认 GPU 0

if [ -z "$ORGAN" ] || [ -z "$VIEWS" ]; then
    echo "============================================================================"
    echo "SPAGS 3k 消融实验训练脚本"
    echo "============================================================================"
    echo ""
    echo "用法: $0 <器官> <视角数> [GPU]"
    echo ""
    echo "器官: foot, chest, head, abdomen, pancreas"
    echo "视角: 3, 6, 9"
    echo ""
    echo "示例:"
    echo "  $0 foot 3 0"
    echo "  $0 chest 6 1"
    echo ""
    echo "配置:"
    echo "  - SPS: 3k 密度加权点云"
    echo "  - GAR: 5 邻居点, threshold=5.0"
    echo "  - ADM: lambda_tv=0.002"
    exit 1
fi

# 时间戳
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# 数据集路径
DATA_PATH="data/369/${ORGAN}_50_${VIEWS}views.pickle"
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据集不存在: $DATA_PATH"
    exit 1
fi

# SPS 3k 点云路径
SPS_PCD_PATH="data/369-sps-3k/init_${ORGAN}_50_${VIEWS}views.npy"
if [ ! -f "$SPS_PCD_PATH" ]; then
    echo "错误: SPS 3k 点云不存在: $SPS_PCD_PATH"
    exit 1
fi

# 输出目录
OUTPUT="output/${TIMESTAMP}_${ORGAN}_${VIEWS}views_spags_3k"

# ============================================================================
# SPAGS 配置
# ============================================================================

# 公共参数
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000"

# GAR 参数 (5 邻居点)
GAR_FLAGS="--enable_binocular_consistency \
    --binocular_loss_weight 0.08 \
    --binocular_max_angle_offset 0.04 \
    --binocular_start_iter 5000 \
    --binocular_warmup_iters 3000 \
    --enable_fsgs_proximity \
    --proximity_threshold 5.0 \
    --proximity_k_neighbors 5 \
    --enable_medical_constraints"

# ADM 参数 (lambda_tv=0.002)
ADM_FLAGS="--enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --lambda_plane_tv 0.002 \
    --tv_loss_type l2"

# ============================================================================
# 训练
# ============================================================================

echo ""
echo "============================================================================"
echo "SPAGS 3k 消融实验"
echo "============================================================================"
echo "器官: $ORGAN"
echo "视角: $VIEWS"
echo "GPU: $GPU"
echo "数据: $DATA_PATH"
echo "SPS点云: $SPS_PCD_PATH (3k points)"
echo "输出: $OUTPUT"
echo ""
echo "技术配置:"
echo "  - SPS: 3k 密度加权点云"
echo "  - GAR: proximity_k=5, threshold=5.0"
echo "  - ADM: lambda_tv=0.002"
echo "============================================================================"
echo ""

mkdir -p "$OUTPUT"

# 记录配置
cat > "${OUTPUT}/spags_3k_config.txt" << EOF
SPAGS 3k 消融实验配置
====================
器官: $ORGAN
视角: $VIEWS
GPU: $GPU
时间: $(date)

技术配置:
  SPS: 3k 密度加权点云 ($SPS_PCD_PATH)
  GAR: proximity_k=5, threshold=5.0
  ADM: lambda_tv=0.002

完整命令:
CUDA_VISIBLE_DEVICES=$GPU python train.py \\
    -s $DATA_PATH \\
    -m $OUTPUT \\
    $COMMON_FLAGS \\
    $GAR_FLAGS \\
    $ADM_FLAGS \\
    --ply_path $SPS_PCD_PATH
EOF

# 执行训练
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT" \
    $COMMON_FLAGS \
    $GAR_FLAGS \
    $ADM_FLAGS \
    --ply_path "$SPS_PCD_PATH" \
    2>&1 | tee "${OUTPUT}/training.log"

echo ""
echo "============================================================================"
echo "训练完成！"
echo "输出目录: $OUTPUT"
echo "============================================================================"

# 生成可视化对比图
echo ""
echo ">>> 生成可视化对比图..."
python cc-agent/scripts/save_comparison_png.py \
    --model_path "$OUTPUT" \
    --iteration 30000 \
    2>&1 || echo "警告: 可视化生成失败 (脚本可能不存在)"

echo ""
echo ">>> 完成: ${ORGAN}_${VIEWS}views_spags_3k"
