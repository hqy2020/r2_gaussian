#!/bin/bash
# ============================================================================
# SAX-NeRF (Lineformer) 基线实验脚本
# ============================================================================
# 用法:
#   ./cc-agent/scripts/run_saxnerf_baseline.sh <organ> <views> [GPU]
#
# 器官: foot, chest, head, abdomen, pancreas
# 视角: 3, 6, 9
#
# 示例:
#   ./cc-agent/scripts/run_saxnerf_baseline.sh foot 3 0
#   ./cc-agent/scripts/run_saxnerf_baseline.sh chest 6 1
# ============================================================================

set -e

# 取消代理设置
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 项目路径
R2_ROOT="/home/qyhu/Documents/r2_ours/r2_gaussian"
SAX_ROOT="/home/qyhu/Documents/SAX-NeRF-master"

# 参数解析
ORGAN=$1
VIEWS=$2
GPU=${3:-0}

if [ -z "$ORGAN" ] || [ -z "$VIEWS" ]; then
    echo "============================================================================"
    echo "SAX-NeRF (Lineformer) 基线实验脚本"
    echo "============================================================================"
    echo ""
    echo "用法: $0 <organ> <views> [GPU]"
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
if [[ ! "$ORGAN" =~ ^(foot|chest|head|abdomen|pancreas)$ ]]; then
    echo "错误: 无效器官 '$ORGAN'"
    echo "可用器官: foot, chest, head, abdomen, pancreas"
    exit 1
fi

if [[ ! "$VIEWS" =~ ^(3|6|9)$ ]]; then
    echo "错误: 无效视角数 '$VIEWS'"
    echo "可用视角: 3, 6, 9"
    exit 1
fi

# 数据路径
DATA_PATH="${R2_ROOT}/data/369/${ORGAN}_50_${VIEWS}views.pickle"
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据集不存在: $DATA_PATH"
    exit 1
fi

# 输出目录
OUTPUT_DIR="${R2_ROOT}/output/baselines/saxnerf"
EXP_NAME="${ORGAN}_${VIEWS}views_saxnerf"
mkdir -p "${OUTPUT_DIR}/${EXP_NAME}"

# 生成临时配置文件
CONFIG_FILE="/tmp/saxnerf_${ORGAN}_${VIEWS}views_config.yaml"
cat > "$CONFIG_FILE" << EOF
exp:
  expname: ${EXP_NAME}
  expdir: ${OUTPUT_DIR}/
  datadir: ${DATA_PATH}
network:
  net_type: Lineformer
  num_layers: 4
  hidden_dim: 32
  skips:
  - 2
  out_dim: 1
  last_activation: sigmoid
  bound: 0.3
  line_size: 2
  dim_head: 4
  heads: 8
  num_blocks: 1
encoder:
  encoding: hashgrid
  input_dim: 3
  num_levels: 16
  level_dim: 2
  base_resolution: 16
  log2_hashmap_size: 19
render:
  n_samples: 256
  n_fine: 0
  perturb: true
  raw_noise_std: 0.0
  netchunk: 409600
train:
  epoch: 1500
  n_batch: 1
  n_rays: 1024
  lrate: 0.001
  lrate_gamma: 0.1
  lrate_step: 1500
  resume: false
  window_size:
  - 8
  - 8
  window_num: 16
log:
  i_eval: 250
  i_save: 500
EOF

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 打印信息
echo ""
echo "============================================================================"
echo "SAX-NeRF (Lineformer) 基线实验"
echo "============================================================================"
echo "器官: $ORGAN"
echo "视角: $VIEWS"
echo "GPU: $GPU"
echo "数据: $DATA_PATH"
echo "输出: ${OUTPUT_DIR}/${EXP_NAME}"
echo "配置: $CONFIG_FILE"
echo "============================================================================"
echo ""

# 运行训练 (SAX-NeRF 使用 train_mlg.py)
cd "$SAX_ROOT"
CUDA_VISIBLE_DEVICES=$GPU python train_mlg.py \
    --config "$CONFIG_FILE" \
    --gpu_id 0 \
    2>&1 | tee "${OUTPUT_DIR}/${EXP_NAME}/training.log"

echo ""
echo "============================================================================"
echo "训练完成！"
echo "输出目录: ${OUTPUT_DIR}/${EXP_NAME}"
echo "============================================================================"
