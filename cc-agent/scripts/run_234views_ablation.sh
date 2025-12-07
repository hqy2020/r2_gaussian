#!/bin/bash
# ============================================================================
# 234 视角消融实验批量脚本
# ============================================================================
# 支持配置: baseline, sps, gar, adm, spags
# 数据目录: data/234/
# ============================================================================

set -e

cd /home/qyhu/Documents/r2_ours/r2_gaussian
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_DIR="data/234"
SPS_DATA_DIR="data/density-234"

# GAR 参数
GAR_FLAGS="--enable_fsgs_proximity --gar_adaptive_threshold --gar_adaptive_percentile 95 --gar_progressive_decay --gar_decay_start_ratio 0.5 --gar_final_strength 0.3"

# ADM 参数
ADM_FLAGS="--enable_kplanes"

run_experiment() {
    local organ=$1
    local views=$2
    local config=$3
    local gpu=$4

    local data_path="${DATA_DIR}/${organ}_50_${views}views.pickle"
    local output_name="_${TIMESTAMP}_${organ}_${views}views_${config}"
    local output_dir="output/${output_name}"
    local log_file="output/log_${config}_${organ}_${views}views.log"
    local sps_pcd_path="${SPS_DATA_DIR}/init_${organ}_50_${views}views.npy"

    echo "[GPU $gpu] Starting $config on $organ ${views}views..."

    local config_flags=""
    local ply_flag=""

    case $config in
        baseline)
            config_flags=""
            ;;
        sps)
            ply_flag="--ply_path $sps_pcd_path"
            ;;
        gar)
            config_flags="$GAR_FLAGS"
            ;;
        adm)
            config_flags="$ADM_FLAGS"
            ;;
        spags)
            config_flags="$GAR_FLAGS $ADM_FLAGS"
            ply_flag="--ply_path $sps_pcd_path"
            ;;
    esac

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        -s "$data_path" \
        -m "$output_dir" \
        --eval \
        $config_flags \
        $ply_flag \
        > "$log_file" 2>&1

    echo "[GPU $gpu] Finished $config on $organ ${views}views"
}

# ============================================================================
# 用法提示
# ============================================================================
show_usage() {
    echo "============================================================================"
    echo "234 视角消融实验批量脚本"
    echo "============================================================================"
    echo ""
    echo "用法: $0 <配置> [视角]"
    echo ""
    echo "配置选项:"
    echo "  baseline  - Baseline (无任何技术)"
    echo "  sps       - 仅 SPS (空间先验播种)"
    echo "  gar       - 仅 GAR (几何感知细化)"
    echo "  adm       - 仅 ADM (自适应密度调制)"
    echo "  spags     - Full SPAGS (SPS + GAR + ADM)"
    echo "  all       - 运行所有配置"
    echo ""
    echo "视角选项 (可选):"
    echo "  2, 3, 4   - 运行指定视角"
    echo "  all       - 运行所有视角 (默认)"
    echo ""
    echo "示例:"
    echo "  $0 adm           # 运行所有视角的 ADM 实验"
    echo "  $0 spags 3       # 仅运行 3 视角的 SPAGS 实验"
    echo "  $0 all           # 运行所有配置和视角"
    echo "  $0 baseline all  # 运行所有视角的 baseline"
    exit 1
}

# ============================================================================
# 参数解析
# ============================================================================
CONFIG=${1:-""}
VIEWS_FILTER=${2:-"all"}

if [ -z "$CONFIG" ]; then
    show_usage
fi

# 确定要运行的配置
if [ "$CONFIG" == "all" ]; then
    CONFIGS=("baseline" "sps" "gar" "adm" "spags")
else
    CONFIGS=("$CONFIG")
fi

# 确定要运行的视角
if [ "$VIEWS_FILTER" == "all" ]; then
    VIEWS_LIST=("2" "3" "4")
else
    VIEWS_LIST=("$VIEWS_FILTER")
fi

ORGANS=("foot" "chest" "head" "abdomen" "pancreas")

# 计算总实验数
TOTAL=$((${#CONFIGS[@]} * ${#ORGANS[@]} * ${#VIEWS_LIST[@]}))

echo "=========================================="
echo "234 视角消融实验"
echo "=========================================="
echo "配置: ${CONFIGS[*]}"
echo "视角: ${VIEWS_LIST[*]}"
echo "器官: ${ORGANS[*]}"
echo "总实验数: $TOTAL"
echo "时间戳: $TIMESTAMP"
echo "=========================================="

# ============================================================================
# 运行实验
# ============================================================================

for views in "${VIEWS_LIST[@]}"; do
    echo ""
    echo "========== ${views}views 实验 =========="

    for config in "${CONFIGS[@]}"; do
        echo "--- 配置: $config ---"

        # GPU 0 运行前半部分器官
        run_experiment foot $views $config 0 &
        run_experiment chest $views $config 0 &
        run_experiment head $views $config 0 &

        # GPU 1 运行后半部分器官
        run_experiment abdomen $views $config 1 &
        run_experiment pancreas $views $config 1 &

        wait
        echo "配置 $config ${views}views 完成!"
    done
done

echo ""
echo "=========================================="
echo "所有 $TOTAL 个实验完成!"
echo "结果在: output/_${TIMESTAMP}_*"
echo "=========================================="
