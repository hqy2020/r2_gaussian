#!/bin/bash
# ============================================================================
# SPAGS 完整消融实验批处理
# ============================================================================
# 运行所有 8 种配置的消融实验
#
# 用法:
#   ./cc-agent/scripts/run_spags_ablation_all.sh <器官> <视角数> [GPU]
#
# 示例:
#   ./cc-agent/scripts/run_spags_ablation_all.sh foot 3 0
# ============================================================================

set -e

ORGAN=$1
VIEWS=$2
GPU=${3:-0}

if [ -z "$ORGAN" ] || [ -z "$VIEWS" ]; then
    echo "用法: $0 <器官> <视角数> [GPU]"
    echo "示例: $0 foot 3 0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_SCRIPT="${SCRIPT_DIR}/run_spags_ablation.sh"

echo "============================================================================"
echo "SPAGS 完整消融实验"
echo "============================================================================"
echo "器官: $ORGAN"
echo "视角: $VIEWS"
echo "GPU: $GPU"
echo ""
echo "将运行以下 8 种配置:"
echo "  1. baseline  - Baseline"
echo "  2. sps       - SPS only"
echo "  3. gar       - GAR only"
echo "  4. adm       - ADM only"
echo "  5. sps_gar   - SPS + GAR"
echo "  6. sps_adm   - SPS + ADM"
echo "  7. gar_adm   - GAR + ADM"
echo "  8. spags     - Full SPAGS"
echo "============================================================================"
echo ""

# 配置列表
CONFIGS=("baseline" "sps" "gar" "adm" "sps_gar" "sps_adm" "gar_adm" "spags")

# 逐个运行
for config in "${CONFIGS[@]}"; do
    echo ""
    echo ">>> 开始实验: $config"
    echo ">>> 时间: $(date)"
    echo ""

    bash "$ABLATION_SCRIPT" "$config" "$ORGAN" "$VIEWS" "$GPU"

    echo ""
    echo ">>> 完成实验: $config"
    echo ">>> 时间: $(date)"
    echo ""
done

echo "============================================================================"
echo "所有消融实验完成！"
echo "时间: $(date)"
echo "============================================================================"
