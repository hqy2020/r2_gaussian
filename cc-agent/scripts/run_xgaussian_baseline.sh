#!/bin/bash
# ============================================================================
# X-Gaussian Baseline 对比实验脚本
# ============================================================================
# 用法:
#   ./cc-agent/scripts/run_xgaussian_baseline.sh <器官> <视角数> [GPU]
#
# 示例:
#   ./cc-agent/scripts/run_xgaussian_baseline.sh foot 3 0
#   ./cc-agent/scripts/run_xgaussian_baseline.sh chest 6 1
# ============================================================================

set -e

# 取消代理设置和清除可能冲突的 PYTHONPATH
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy
unset PYTHONPATH

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 项目路径
XGAUSSIAN_DIR="/home/qyhu/Documents/X-Gaussian"
R2_DATA_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/data/369"
CONFIG_DIR="$XGAUSSIAN_DIR/config/r2_comparison"

# 参数解析
ORGAN=$1     # foot/chest/head/abdomen/pancreas
VIEWS=$2     # 3/6/9
GPU=${3:-0}  # 默认 GPU 0

if [ -z "$ORGAN" ] || [ -z "$VIEWS" ]; then
    echo "============================================================================"
    echo "X-Gaussian Baseline 对比实验脚本"
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
    exit 1
fi

# 验证参数
case $ORGAN in
    foot|chest|head|abdomen|pancreas) ;;
    *) echo "错误: 无效器官 '$ORGAN'"; exit 1 ;;
esac

case $VIEWS in
    3|6|9) ;;
    *) echo "错误: 无效视角数 '$VIEWS'"; exit 1 ;;
esac

# 数据路径
DATA_PATH="$R2_DATA_DIR/${ORGAN}_50_${VIEWS}views.pickle"
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据集不存在: $DATA_PATH"
    exit 1
fi

# 时间戳和输出目录
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
OUTPUT_NAME="${TIMESTAMP}_${ORGAN}_${VIEWS}views_xgaussian"
OUTPUT_DIR="$XGAUSSIAN_DIR/output/r2_comparison/$OUTPUT_NAME"

# 配置文件路径
CONFIG_FILE="$CONFIG_DIR/${ORGAN}_${VIEWS}views.yaml"

# 确保目录存在
mkdir -p "$CONFIG_DIR"
mkdir -p "$OUTPUT_DIR"

# 生成配置文件
cat > "$CONFIG_FILE" << EOF
# X-Gaussian 配置: ${ORGAN} ${VIEWS}views
# 自动生成于 $(date)

# 数据路径
source_path: $DATA_PATH
scene: ${ORGAN}_${VIEWS}views

# 训练参数 (与 r2_gaussian 对齐 30000 iterations)
iterations: 30000

# 密集化参数
densification_interval: 200
densify_from_iter: 500
densify_until_iter: 8000
densify_grad_threshold: 2.6e-05

# 学习率
position_lr_init: 0.00019
position_lr_final: 1.9e-06
position_lr_delay_mult: 0.01
position_lr_max_steps: 30000
feature_lr: 0.002
opacity_lr: 0.008
radiodensity_lr: 0.05
scaling_lr: 0.005
rotation_lr: 0.001

# 其他
opacity_reset_interval: 4000
radiodensity_reset_interval: 2000
percent_dense: 0.01
lambda_dssim: 0.2
random_background: false

# 点云初始化间隔
interval: 8
EOF

echo ""
echo "============================================================================"
echo "X-Gaussian Baseline 对比实验"
echo "============================================================================"
echo "器官: $ORGAN"
echo "视角: $VIEWS"
echo "GPU: $GPU"
echo "数据: $DATA_PATH"
echo "配置: $CONFIG_FILE"
echo "输出: $OUTPUT_DIR"
echo "============================================================================"
echo ""

cd "$XGAUSSIAN_DIR"

# 执行训练
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config "$CONFIG_FILE" \
    --model_path "$OUTPUT_DIR" \
    --eval \
    --test_iterations 10000 20000 30000 \
    --save_iterations 30000 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================================================"
echo "训练完成！"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================================"
