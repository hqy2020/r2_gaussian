#!/bin/bash
# FSGS 超参数搜索脚本
# 用法: ./scripts/fsgs_hyperparam_search.sh <CONFIG_ID> <SCENE_ID> <GPU>
#       ./scripts/fsgs_hyperparam_search.sh scene <SCENE_ID> <GPU>  # 运行某场景所有配置

set -e

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# ============== 场景定义 ==============
# 场景 ID: 0=Foot-3, 1=Abdomen-9
declare -A SCENES
SCENES[0]="foot,3,data/369/foot_50_3views.pickle"
SCENES[1]="abdomen,9,data/369/abdomen_50_9views.pickle"

# ============== 超参数配置定义 ==============
# 格式: "名称,depth_pseudo_weight,fsgs_depth_weight,proximity_threshold,proximity_k_neighbors,start_sample_pseudo"
declare -A CONFIGS
CONFIGS[0]="baseline,0.03,0.05,5.0,5,5000"      # 当前配置（对照组）
CONFIGS[1]="dpw_001,0.01,0.05,5.0,5,5000"       # 降低伪深度权重
CONFIGS[2]="dpw_005,0.05,0.05,5.0,5,5000"       # 增加伪深度权重
CONFIGS[3]="dpw_008,0.08,0.05,5.0,5,5000"       # 大幅增加伪深度
CONFIGS[4]="fdw_003,0.03,0.03,5.0,5,5000"       # 降低 MiDaS 权重
CONFIGS[5]="fdw_008,0.03,0.08,5.0,5,5000"       # 增加 MiDaS 权重
CONFIGS[6]="pt_3,0.03,0.05,3.0,5,5000"          # 收紧邻近阈值
CONFIGS[7]="pt_7,0.03,0.05,7.0,5,5000"          # 放松邻近阈值
CONFIGS[8]="pk_3,0.03,0.05,5.0,3,5000"          # 收紧邻居数
CONFIGS[9]="pk_7,0.03,0.05,5.0,7,5000"          # 放松邻居数
CONFIGS[10]="start_3k,0.03,0.05,5.0,5,3000"     # 提前启动
CONFIGS[11]="start_7k,0.03,0.05,5.0,5,7000"     # 延后启动
CONFIGS[12]="tight,0.03,0.05,3.0,3,5000"        # 全面收紧

# ============== 公共参数 ==============
ITERATIONS=${ITERATIONS:-10000}  # 默认 10k，可通过环境变量覆盖
COMMON_FLAGS="--iterations $ITERATIONS --test_iterations $ITERATIONS --gaussiansN 1"

# ============== 函数定义 ==============
run_experiment() {
    local CONFIG_ID=$1
    local SCENE_ID=$2
    local GPU=$3

    # 解析场景
    IFS=',' read -r ORGAN VIEWS DATA_PATH <<< "${SCENES[$SCENE_ID]}"

    # 解析配置
    IFS=',' read -r CONFIG_NAME DPW FDW PT PK START <<< "${CONFIGS[$CONFIG_ID]}"

    # 检查数据文件
    if [ ! -f "$DATA_PATH" ]; then
        echo "错误: 数据文件不存在: $DATA_PATH"
        return 1
    fi

    # 生成输出目录
    TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
    OUTPUT="output/fsgs_search/${TIMESTAMP}_${ORGAN}_${VIEWS}views_F_${CONFIG_NAME}"

    echo "=============================================="
    echo "FSGS 超参数搜索实验"
    echo "场景: ${ORGAN}-${VIEWS}views"
    echo "配置: ${CONFIG_NAME}"
    echo "参数: depth_pseudo_weight=${DPW}, fsgs_depth_weight=${FDW}"
    echo "      proximity_threshold=${PT}, proximity_k_neighbors=${PK}"
    echo "      start_sample_pseudo=${START}"
    echo "迭代: ${ITERATIONS}"
    echo "输出: $OUTPUT"
    echo "GPU: $GPU"
    echo "=============================================="

    mkdir -p "$OUTPUT"

    # 运行训练 - 纯 FSGS 配置
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s "$DATA_PATH" \
        -m "$OUTPUT" \
        $COMMON_FLAGS \
        --enable_fsgs_depth \
        --enable_medical_constraints \
        --depth_pseudo_weight $DPW \
        --fsgs_depth_weight $FDW \
        --proximity_threshold $PT \
        --proximity_k_neighbors $PK \
        --start_sample_pseudo $START \
        2>&1 | tee "${OUTPUT}/training.log"

    # 提取结果
    RESULT_FILE="$OUTPUT/eval/iter_$(printf '%06d' $ITERATIONS)/eval2d_render_test.yml"
    if [ -f "$RESULT_FILE" ]; then
        PSNR=$(grep "^psnr:" "$RESULT_FILE" | awk '{print $2}')
        SSIM=$(grep "^ssim:" "$RESULT_FILE" | awk '{print $2}')
        echo "结果: PSNR=${PSNR}, SSIM=${SSIM}"
        echo "${ORGAN},${VIEWS},${CONFIG_NAME},${DPW},${FDW},${PT},${PK},${START},${PSNR},${SSIM}" >> "output/fsgs_search/results.csv"
    fi

    echo "完成: ${CONFIG_NAME} @ ${ORGAN}-${VIEWS}views"
    echo ""
}

# ============== 主逻辑 ==============
# 创建输出目录和结果文件
mkdir -p output/fsgs_search
if [ ! -f "output/fsgs_search/results.csv" ]; then
    echo "organ,views,config,depth_pseudo_weight,fsgs_depth_weight,proximity_threshold,proximity_k_neighbors,start_sample_pseudo,psnr,ssim" > output/fsgs_search/results.csv
fi

if [ "$1" = "scene" ]; then
    SCENE_ID=$2
    GPU=${3:-0}
    echo "运行场景 $SCENE_ID 的所有 13 个配置 (GPU: $GPU)"
    for CONFIG_ID in {0..12}; do
        run_experiment $CONFIG_ID $SCENE_ID $GPU
    done
elif [ "$1" = "all" ]; then
    echo "错误: 请使用 fsgs_search_parallel.sh 并行运行所有场景"
    exit 1
elif [ -n "$1" ] && [ -n "$2" ]; then
    CONFIG_ID=$1
    SCENE_ID=$2
    GPU=${3:-0}
    run_experiment $CONFIG_ID $SCENE_ID $GPU
else
    echo "FSGS 超参数搜索"
    echo ""
    echo "用法:"
    echo "  $0 <CONFIG_ID> <SCENE_ID> <GPU>  # 运行单个实验"
    echo "  $0 scene <SCENE_ID> <GPU>        # 运行某场景的所有配置"
    echo ""
    echo "场景 ID:"
    echo "  0: Foot-3views"
    echo "  1: Abdomen-9views"
    echo ""
    echo "配置 ID:"
    echo "  0: baseline (对照组)"
    echo "  1: dpw_001 (depth_pseudo_weight=0.01)"
    echo "  2: dpw_005 (depth_pseudo_weight=0.05)"
    echo "  3: dpw_008 (depth_pseudo_weight=0.08)"
    echo "  4: fdw_003 (fsgs_depth_weight=0.03)"
    echo "  5: fdw_008 (fsgs_depth_weight=0.08)"
    echo "  6: pt_3 (proximity_threshold=3.0)"
    echo "  7: pt_7 (proximity_threshold=7.0)"
    echo "  8: pk_3 (proximity_k_neighbors=3)"
    echo "  9: pk_7 (proximity_k_neighbors=7)"
    echo "  10: start_3k (start_sample_pseudo=3000)"
    echo "  11: start_7k (start_sample_pseudo=7000)"
    echo "  12: tight (全面收紧: pt=3.0, pk=3)"
    echo ""
    echo "环境变量:"
    echo "  ITERATIONS=10000  # 迭代次数（默认 10000）"
    exit 1
fi

echo "实验完成!"
