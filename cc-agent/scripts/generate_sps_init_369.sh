#!/bin/bash
# ============================================================================
# 生成 SPS 初始化点云（data/369-sps）
# ============================================================================
# 说明:
# - 读取 data/369 下的 15 个场景 pickle（5 器官 × 3 视角）
# - 输出到 data/369-sps/init_<organ>_50_<views>views.npy
# - 使用 initialize_pcd.py 的 --enable_sps（density-weighted）
#
# 用法:
#   ./cc-agent/scripts/generate_sps_init_369.sh [GPU] [N_POINTS]
#   例:
#   ./cc-agent/scripts/generate_sps_init_369.sh 0 50000
# ============================================================================

set -e

# 取消代理设置（与训练脚本保持一致）
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU=${1:-0}
N_POINTS=${2:-50000}

# GPU 检查：TIGRE(FDK) 需要 CUDA，CPU 环境下可能直接崩溃/segfault
CUDA_VISIBLE_DEVICES=$GPU python - <<'PY'
import sys
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
    visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"错误: 未检测到可用的 CUDA GPU (CUDA_VISIBLE_DEVICES={visible}).")
    print("SPS 初始化依赖 TIGRE(FDK) 需要 GPU。请确认：")
    print("  1) nvidia-smi 正常")
    print("  2) 该 conda 环境里 torch 为 CUDA 版本")
    print("  3) CUDA_VISIBLE_DEVICES 指向存在的 GPU")
    sys.exit(1)

print(f"✓ 检测到 GPU: {torch.cuda.get_device_name(0)}")
PY

# 5 个器官 × 3 种视角
ORGANS=("foot" "chest" "head" "abdomen" "pancreas")
VIEWS=(3 6 9)

OUT_DIR="data/369-sps"
mkdir -p "$OUT_DIR"

echo "============================================================================"
echo "生成 SPS 初始化点云（369-sps）"
echo "GPU: $GPU"
echo "N_POINTS: $N_POINTS"
echo "输出目录: $OUT_DIR"
echo "============================================================================"

for ORGAN in "${ORGANS[@]}"; do
  for VIEW in "${VIEWS[@]}"; do
    DATA_PATH="data/369/${ORGAN}_50_${VIEW}views.pickle"
    OUT_PATH="${OUT_DIR}/init_${ORGAN}_50_${VIEW}views.npy"

    if [ ! -f "$DATA_PATH" ]; then
      echo "[跳过] 数据集不存在: $DATA_PATH"
      continue
    fi

    if [ -f "$OUT_PATH" ]; then
      echo "[跳过] 已存在: $OUT_PATH"
      continue
    fi

    echo ""
    echo ">>> 生成: ${ORGAN} ${VIEW}views"
    echo "    data: $DATA_PATH"
    echo "    out : $OUT_PATH"

    CUDA_VISIBLE_DEVICES=$GPU python initialize_pcd.py \
      --data "$DATA_PATH" \
      --output "$OUT_PATH" \
      --enable_sps \
      --sps_strategy density_weighted \
      --n_points "$N_POINTS"
  done
done

echo ""
echo "============================================================================"
echo "SPS 初始化点云生成完成"
echo "输出目录: $OUT_DIR"
echo "============================================================================"
