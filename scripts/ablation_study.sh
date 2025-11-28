#!/bin/bash
# R²-Gaussian 消融实验脚本
# 16 种技术配置 × 3 场景 = 48 个实验
# 用法: ./scripts/ablation_study.sh <CONFIG_ID> <SCENE_ID> <GPU>
#       ./scripts/ablation_study.sh all <GPU>  # 运行所有实验

set -e

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# ============== 场景定义 ==============
# 场景 ID: 0=Foot-3, 1=Chest-6, 2=Abdomen-9
declare -A SCENES
SCENES[0]="foot,3"
SCENES[1]="chest,6"
SCENES[2]="abdomen,9"

# ============== 技术配置定义 ==============
# 配置格式: "名称,I,X,F,B" (1=启用, 0=禁用)
# I=Init-PCD, X=X²-Gaussian, F=FSGS, B=Bino
declare -A CONFIGS
CONFIGS[0]="baseline,0,0,0,0"
CONFIGS[1]="I,1,0,0,0"
CONFIGS[2]="X,0,1,0,0"
CONFIGS[3]="F,0,0,1,0"
CONFIGS[4]="B,0,0,0,1"
CONFIGS[5]="IX,1,1,0,0"
CONFIGS[6]="IF,1,0,1,0"
CONFIGS[7]="IB,1,0,0,1"
CONFIGS[8]="XF,0,1,1,0"
CONFIGS[9]="XB,0,1,0,1"
CONFIGS[10]="FB,0,0,1,1"
CONFIGS[11]="IXF,1,1,1,0"
CONFIGS[12]="IXB,1,1,0,1"
CONFIGS[13]="IFB,1,0,1,1"
CONFIGS[14]="XFB,0,1,1,1"
CONFIGS[15]="IXFB,1,1,1,1"

# ============== 公共参数 ==============
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000 --gaussiansN 1"

# ============== X²-Gaussian 参数 ==============
X_FLAGS="--enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --lambda_plane_tv 0.002 \
    --tv_loss_type l2"

# ============== FSGS 参数 ==============
F_FLAGS="--enable_fsgs_depth \
    --enable_medical_constraints \
    --depth_pseudo_weight 0.03 \
    --proximity_threshold 5.0 \
    --proximity_k_neighbors 5 \
    --start_sample_pseudo 5000"

# ============== Bino 参数 ==============
B_FLAGS="--enable_binocular_consistency \
    --binocular_max_angle_offset 0.06 \
    --binocular_start_iter 7000 \
    --binocular_warmup_iters 3000 \
    --binocular_loss_weight 0.15"

# ============== 函数定义 ==============
run_experiment() {
    local CONFIG_ID=$1
    local SCENE_ID=$2
    local GPU=$3

    # 解析场景
    IFS=',' read -r ORGAN VIEWS <<< "${SCENES[$SCENE_ID]}"

    # 解析配置
    IFS=',' read -r CONFIG_NAME USE_I USE_X USE_F USE_B <<< "${CONFIGS[$CONFIG_ID]}"

    # 选择数据路径 (I=1 使用 density-369，否则使用 369)
    if [ "$USE_I" = "1" ]; then
        DATA_DIR="data/density-369"
    else
        DATA_DIR="data/369"
    fi
    DATA_PATH="${DATA_DIR}/${ORGAN}_50_${VIEWS}views.pickle"

    # 检查数据文件
    if [ ! -f "$DATA_PATH" ]; then
        echo "错误: 数据文件不存在: $DATA_PATH"
        return 1
    fi

    # 生成输出目录
    TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
    OUTPUT="output/ablation/${TIMESTAMP}_${ORGAN}_${VIEWS}views_${CONFIG_NAME}"

    # 构建参数
    TECH_FLAGS=""
    [ "$USE_X" = "1" ] && TECH_FLAGS="$TECH_FLAGS $X_FLAGS"
    [ "$USE_F" = "1" ] && TECH_FLAGS="$TECH_FLAGS $F_FLAGS"
    [ "$USE_B" = "1" ] && TECH_FLAGS="$TECH_FLAGS $B_FLAGS"

    echo "=============================================="
    echo "实验: ${CONFIG_NAME} @ ${ORGAN}-${VIEWS}views"
    echo "配置: I=$USE_I, X=$USE_X, F=$USE_F, B=$USE_B"
    echo "数据: $DATA_PATH"
    echo "输出: $OUTPUT"
    echo "GPU: $GPU"
    echo "=============================================="

    mkdir -p "$OUTPUT"

    # 运行训练
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s "$DATA_PATH" \
        -m "$OUTPUT" \
        $COMMON_FLAGS \
        $TECH_FLAGS \
        2>&1 | tee "${OUTPUT}/training.log"

    echo "完成: ${CONFIG_NAME} @ ${ORGAN}-${VIEWS}views"
    echo ""
}

# ============== 主逻辑 ==============
if [ "$1" = "all" ]; then
    GPU=${2:-0}
    echo "运行所有 48 个消融实验 (GPU: $GPU)"
    for SCENE_ID in 0 1 2; do
        for CONFIG_ID in {0..15}; do
            run_experiment $CONFIG_ID $SCENE_ID $GPU
        done
    done
elif [ "$1" = "scene" ]; then
    SCENE_ID=$2
    GPU=${3:-0}
    echo "运行场景 $SCENE_ID 的所有 16 个配置 (GPU: $GPU)"
    for CONFIG_ID in {0..15}; do
        run_experiment $CONFIG_ID $SCENE_ID $GPU
    done
elif [ "$1" = "config" ]; then
    CONFIG_ID=$2
    GPU=${3:-0}
    echo "运行配置 $CONFIG_ID 在所有 3 个场景 (GPU: $GPU)"
    for SCENE_ID in 0 1 2; do
        run_experiment $CONFIG_ID $SCENE_ID $GPU
    done
elif [ -n "$1" ] && [ -n "$2" ]; then
    CONFIG_ID=$1
    SCENE_ID=$2
    GPU=${3:-0}
    run_experiment $CONFIG_ID $SCENE_ID $GPU
else
    echo "用法:"
    echo "  $0 <CONFIG_ID> <SCENE_ID> <GPU>  # 运行单个实验"
    echo "  $0 all <GPU>                      # 运行所有 48 个实验"
    echo "  $0 scene <SCENE_ID> <GPU>         # 运行某场景的所有配置"
    echo "  $0 config <CONFIG_ID> <GPU>       # 运行某配置在所有场景"
    echo ""
    echo "场景 ID:"
    echo "  0: Foot-3views"
    echo "  1: Chest-6views"
    echo "  2: Abdomen-9views"
    echo ""
    echo "配置 ID:"
    echo "  0: Baseline      8: X+F"
    echo "  1: I             9: X+B"
    echo "  2: X            10: F+B"
    echo "  3: F            11: I+X+F"
    echo "  4: B            12: I+X+B"
    echo "  5: I+X          13: I+F+B"
    echo "  6: I+F          14: X+F+B"
    echo "  7: I+B          15: I+X+F+B (Full)"
    exit 1
fi

echo "所有实验完成!"
