
set -e

# 取消代理设置
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 参数解析
CONFIG=$1    # baseline/sps/gar/adm/sps_gar/sps_adm/gar_adm/spags
ORGAN=$2     # foot/chest/head/abdomen/pancreas
VIEWS=$3     # 3/6/9
GPU=${4:-0}  # 默认 GPU 0
OUTPUT_PREFIX=${5:-}  # 可选的输出前缀（如 lucky_gar）

if [ -z "$CONFIG" ] || [ -z "$ORGAN" ] || [ -z "$VIEWS" ]; then
    echo "============================================================================"
    echo "SPAGS 消融实验脚本"
    echo "============================================================================"
    echo ""
    echo "用法: $0 <配置> <器官> <视角数> [GPU]"
    echo ""
    echo "配置选项:"
    echo "  baseline  - Baseline (无任何技术)"
    echo "  sps       - 仅 SPS (空间先验播种)"
    echo "  gar       - 仅 GAR (几何感知细化)"
    echo "  adm       - 仅 ADM (自适应密度调制)"
    echo "  sps_gar   - SPS + GAR"
    echo "  sps_adm   - SPS + ADM"
    echo "  gar_adm   - GAR + ADM"
    echo "  spags     - Full SPAGS (SPS + GAR + ADM)"
    echo ""
    echo "Baseline 方法:"
    echo "  xgaussian - X-Gaussian baseline"
    echo "  naf       - NAF (Neural Attenuation Fields)"
    echo "  tensorf   - TensoRF"
    echo "  saxnerf   - SAX-NeRF with Lineformer"
    echo ""
    echo "器官: foot, chest, head, abdomen, pancreas"
    echo "视角: 3, 6, 9"
    echo ""
    echo "示例:"
    echo "  $0 spags foot 3 0"
    echo "  $0 baseline chest 6 1"
    echo "  $0 xgaussian foot 3 0"
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

# 初始化点云路径
# - baseline / gar / adm / gar_adm: 使用 data/369 下的 init_*.npy（与数据同目录，作为 baseline init）
# - sps / sps_* / spags / xgaussian: 使用 data/369-sps 下的 init_*.npy（SPS 生成，避免覆盖 baseline init）
BASE_INIT_PCD_PATH="data/369/init_${ORGAN}_50_${VIEWS}views.npy"
SPS_INIT_PCD_DIR="data/369-sps"
SPS_INIT_PCD_PATH="${SPS_INIT_PCD_DIR}/init_${ORGAN}_50_${VIEWS}views.npy"

# 公共参数
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000"

# ============================================================================
# 配置定义
# ============================================================================

# GAR 参数（Proximity-guided Densification）- 🔧 优化版本 v5
# 关键优化（基于诊断分析）：
#   - 📊 percentile 85→90：只密化最稀疏的 10%（原来 15% 仍然偏多）
#   - 📊 decay_start_ratio 0.7→0.6：更早开始衰减，给新点更多优化时间
#   - 📊 max_candidates 1000→2000：增加每次密化的点数上限
#   - 🔧 下限保护已在代码中从 P50→P25（允许密化更多点）
#   - 🔧 候选选择已从随机改为按分数排序（优先最稀疏的点）
# 注意: 布尔参数使用 flag 格式（不带 true/false 值）
GAR_FLAGS_COMPAT="--enable_fsgs_proximity \
    --gar_proximity_threshold 0.05 \
    --gar_proximity_k 5 \
    --gar_adaptive_threshold \
    --gar_adaptive_percentile 90 \
    --gar_progressive_decay \
    --gar_decay_start_ratio 0.6 \
    --gar_final_strength 0.5 \
    --gar_max_candidates 2000"

# ADM 参数（K-Planes Density Modulation）- 🔧 补全完整参数
# 关键修复：
#   - 🔴 添加所有关键参数（之前只传了 --enable_kplanes）
#   - 分辨率、特征维度、MLP 配置、TV 正则化
# 注意: 布尔参数使用 flag 格式（不带 true/false 值）
ADM_FLAGS_COMPAT="--enable_kplanes \
    --adm_resolution 64 \
    --adm_feature_dim 32 \
    --adm_decoder_hidden 128 \
    --adm_decoder_layers 3 \
    --lambda_plane_tv 0.002 \
    --adm_max_range 0.3 \
    --adm_view_adaptive"

# ============================================================================
# 根据配置选择参数
# ============================================================================

case $CONFIG in
    baseline)
        echo "=== Baseline (无技术) ==="
        CONFIG_FLAGS=""
        USE_SPS=false
        METHOD="r2_gaussian"
        ;;
    sps)
        echo "=== SPS (空间先验播种) ==="
        CONFIG_FLAGS=""
        USE_SPS=true
        METHOD="r2_gaussian"
        ;;
    gar)
        echo "=== GAR (几何感知细化) ==="
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT"
        USE_SPS=false
        METHOD="r2_gaussian"
        ;;
    adm)
        echo "=== ADM (自适应密度调制) ==="
        CONFIG_FLAGS="$ADM_FLAGS_COMPAT"
        USE_SPS=false
        METHOD="r2_gaussian"
        ;;
    sps_gar)
        echo "=== SPS + GAR ==="
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT"
        USE_SPS=true
        METHOD="r2_gaussian"
        ;;
    sps_adm)
        echo "=== SPS + ADM ==="
        CONFIG_FLAGS="$ADM_FLAGS_COMPAT"
        USE_SPS=true
        METHOD="r2_gaussian"
        ;;
    gar_adm)
        echo "=== GAR + ADM ==="
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT $ADM_FLAGS_COMPAT"
        USE_SPS=false
        METHOD="r2_gaussian"
        ;;
    spags)
        echo "=== Full SPAGS (SPS + GAR + ADM) ==="
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT $ADM_FLAGS_COMPAT"
        USE_SPS=true
        METHOD="r2_gaussian"
        ;;
    xgaussian)
        echo "=== X-Gaussian Baseline ==="
        CONFIG_FLAGS=""
        USE_SPS=true
        METHOD="xgaussian"
        ;;
    naf)
        echo "=== NAF (Neural Attenuation Fields) Baseline ==="
        CONFIG_FLAGS=""
        USE_SPS=false
        METHOD="naf"
        ;;
    tensorf)
        echo "=== TensoRF Baseline ==="
        CONFIG_FLAGS=""
        USE_SPS=false
        METHOD="tensorf"
        ;;
    saxnerf)
        echo "=== SAX-NeRF Baseline ==="
        CONFIG_FLAGS=""
        USE_SPS=false
        METHOD="saxnerf"
        ;;
    *)
        echo "错误: 未知配置 '$CONFIG'"
        echo "可用配置: baseline, sps, gar, adm, sps_gar, sps_adm, gar_adm, spags"
        echo "Baseline 方法: xgaussian, naf, tensorf, saxnerf"
        exit 1
        ;;
esac

# 生成输出目录（支持自定义前缀）
if [ -n "$OUTPUT_PREFIX" ]; then
    OUTPUT="output/${OUTPUT_PREFIX}_${ORGAN}_${VIEWS}views_${CONFIG}"
else
    OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_${CONFIG}"
fi

# ============================================================================
# 初始化点云检查
# - SPS: data/369-sps（由 generate_sps_init_369.sh 生成）
# - 非 SPS: data/369（baseline init，与数据同目录）
# ============================================================================

if [ "$USE_SPS" = true ]; then
    if [ ! -f "$SPS_INIT_PCD_PATH" ]; then
        echo "错误: SPS 初始化点云不存在: $SPS_INIT_PCD_PATH"
        echo "请先生成 SPS init（推荐一次性生成 15 个场景）:"
        echo "  bash ./cc-agent/scripts/generate_sps_init_369.sh $GPU"
        exit 1
    fi
    echo ">>> 使用 SPS 点云: $SPS_INIT_PCD_PATH"
    PLY_FLAG="--ply_path $SPS_INIT_PCD_PATH"
else
    PLY_FLAG=""
    # R²-Gaussian 系列（含 baseline/gar/adm/gar_adm）需要 init_*.npy
    if [ "$METHOD" = "r2_gaussian" ]; then
        if [ ! -f "$BASE_INIT_PCD_PATH" ]; then
            echo "错误: Baseline 初始化点云不存在: $BASE_INIT_PCD_PATH"
            echo "请先生成 baseline init（不要加 --enable_sps）:"
            echo "  python initialize_pcd.py --data $DATA_PATH --output $BASE_INIT_PCD_PATH"
            exit 1
        fi
        echo ">>> 使用 Baseline 点云: $BASE_INIT_PCD_PATH (自动加载；等价于不传 --ply_path)"
    fi
fi

# ============================================================================
# 训练
# ============================================================================

echo ""
echo "============================================================================"
echo "配置: $CONFIG"
echo "方法: $METHOD"
echo "器官: $ORGAN"
echo "视角: $VIEWS"
echo "GPU: $GPU"
echo "数据: $DATA_PATH"
echo "输出: $OUTPUT"
echo "SPS: $USE_SPS"
echo "============================================================================"
echo ""

mkdir -p "$OUTPUT"

# 记录配置
cat > "${OUTPUT}/spags_config.txt" << EOF
SPAGS 消融实验配置
==================
配置: $CONFIG
方法: $METHOD
器官: $ORGAN
视角: $VIEWS
GPU: $GPU
时间: $(date)

SPS 启用: $USE_SPS
Baseline init: $BASE_INIT_PCD_PATH
SPS init: $SPS_INIT_PCD_PATH
GAR 启用: $(echo "$CONFIG_FLAGS" | grep -q "proximity" && echo "true" || echo "false")
ADM 启用: $(echo "$CONFIG_FLAGS" | grep -q "kplanes" && echo "true" || echo "false")

完整命令:
CUDA_VISIBLE_DEVICES=$GPU python train.py \\
    --method $METHOD \\
    -s $DATA_PATH \\
    -m $OUTPUT \\
    $COMMON_FLAGS \\
    $CONFIG_FLAGS \\
    $PLY_FLAG
EOF

# 执行训练
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --method "$METHOD" \
    -s "$DATA_PATH" \
    -m "$OUTPUT" \
    $COMMON_FLAGS \
    $CONFIG_FLAGS \
    $PLY_FLAG \
    2>&1 | tee "${OUTPUT}/training.log"

echo ""
echo "============================================================================"
echo "训练完成！"
echo "输出目录: $OUTPUT"
echo "============================================================================"
