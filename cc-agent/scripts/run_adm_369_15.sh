#!/usr/bin/env bash
set -euo pipefail

# 跑 369 数据集的 ADM 15 个场景（5 organs × 3 views）。
#
# 用法：
#   bash cc-agent/scripts/run_adm_369_15.sh [gpu_id] [output_prefix]
#
# 可选环境变量（透传给 run_spags_ablation.sh）：
#   SPAGS_ITERS       总迭代数（默认 30000）
#   SPAGS_TEST_ITERS  测试迭代点（默认 "10000 20000 30000"）
#
# 示例（快速验证，每个场景 1000 iters）：
#   SPAGS_ITERS=1000 SPAGS_TEST_ITERS="1000" bash cc-agent/scripts/run_adm_369_15.sh 0 verify_adm_1k

GPU_ID="${1:-0}"
OUTPUT_PREFIX="${2:-adm_369_15_$(date +%Y_%m_%d_%H_%M)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "============================================="
echo "[ADM-15] GPU: ${GPU_ID}"
echo "[ADM-15] OUTPUT_PREFIX: ${OUTPUT_PREFIX}"
echo "[ADM-15] SPAGS_ITERS: ${SPAGS_ITERS:-30000}"
echo "[ADM-15] SPAGS_TEST_ITERS: ${SPAGS_TEST_ITERS:-10000 20000 30000}"
echo "============================================="

# 先做一次 CUDA 可用性检查（避免跑到一半才发现环境问题）
if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate r2_gaussian_new >/dev/null 2>&1 || true
fi

if python - <<'PY'
import sys

try:
    import torch
except Exception as e:
    print("[ADM-15] torch import failed:", repr(e))
    sys.exit(2)

print("[ADM-15] torch:", torch.__version__)
print("[ADM-15] cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[ADM-15] cuda_device_count:", torch.cuda.device_count())
    sys.exit(0)
sys.exit(1)
PY
then
  :  # CUDA OK
else
  CUDA_CHECK_RC=$?
  echo ""
  if [ "${CUDA_CHECK_RC}" -eq 2 ]; then
    echo "错误: torch 导入失败。请确认已正确激活 conda 环境 `r2_gaussian_new`。"
  else
    echo "错误: CUDA 不可用：torch.cuda.is_available()=False。"
    echo "请先确保驱动/容器 GPU 挂载正常（nvidia-smi 能正常输出），再重试。"
  fi
  exit 1
fi

cd "${REPO_DIR}"

organs=(foot chest head abdomen pancreas)
views=(3 6 9)

for organ in "${organs[@]}"; do
  for v in "${views[@]}"; do
    echo ""
    echo ">>> [ADM-15] ${organ} ${v}views"
    bash "${SCRIPT_DIR}/run_spags_ablation.sh" adm "${organ}" "${v}" "${GPU_ID}" "${OUTPUT_PREFIX}"
  done
done

echo ""
echo "[ADM-15] 全部完成：output/${OUTPUT_PREFIX}_<organ>_<views>views_adm"
