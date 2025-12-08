#!/bin/bash
# ============================================================================
# SPAGS 消融实验脚本
# ============================================================================
# 用法:
#   ./cc-agent/scripts/run_spags_ablation.sh <配置> <器官> <视角数> [GPU]
#
# 配置选项:
#   baseline  - Baseline (无任何技术)
#   sps       - 仅 SPS (空间先验播种)
#   gar       - 仅 GAR (几何感知细化) - 🆕 已优化版本
#   adm       - 仅 ADM (自适应密度调制)
#   sps_gar   - SPS + GAR
#   sps_adm   - SPS + ADM
#   gar_adm   - GAR + ADM
#   spags     - Full SPAGS (SPS + GAR + ADM)
#
# 示例:
#   ./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0
#   ./cc-agent/scripts/run_spags_ablation.sh baseline chest 6 1
# ============================================================================

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
    echo "器官: foot, chest, head, abdomen, pancreas"
    echo "视角: 3, 6, 9"
    echo ""
    echo "示例:"
    echo "  $0 spags foot 3 0"
    echo "  $0 baseline chest 6 1"
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

# SPS 点云路径（使用 density-369 目录的密度加权点云）
SPS_PCD_PATH="data/density-369/init_${ORGAN}_50_${VIEWS}views.npy"

# 公共参数
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000"

# ============================================================================
# 配置定义
# ============================================================================

# GAR 参数（Proximity-guided Densification）- 🆕 优化版本 v2
# 基于日志分析优化（iter 9000 后几乎停止密化问题）：
#   - 自适应阈值 (percentile=85)：密化最稀疏的 15% 点（之前 5% 太保守）
#   - 渐进衰减 (start=0.7, final=0.5)：延迟衰减开始，减少衰减幅度
# 注意: 布尔参数使用 flag 格式（不带 true/false 值）
GAR_FLAGS_COMPAT="--enable_fsgs_proximity --gar_adaptive_threshold --gar_adaptive_percentile 85 --gar_progressive_decay --gar_decay_start_ratio 0.7 --gar_final_strength 0.5"

# ADM 参数（K-Planes Density Modulation）
# 注意: 布尔参数使用 flag 格式（不带 true/false 值）
# 默认值: resolution=64, dim=32, lambda_plane_tv=0.002, tv_type=l2
ADM_FLAGS_COMPAT="--enable_kplanes"

# ============================================================================
# 根据配置选择参数
# ============================================================================

case $CONFIG in
    baseline)
        echo "=== Baseline (无技术) ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_baseline"
        CONFIG_FLAGS=""
        USE_SPS=false
        ;;
    sps)
        echo "=== SPS (空间先验播种) ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_sps"
        CONFIG_FLAGS=""
        USE_SPS=true
        ;;
    gar)
        echo "=== GAR (几何感知细化) ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_gar"
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT"
        USE_SPS=false
        ;;
    adm)
        echo "=== ADM (自适应密度调制) ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_adm"
        CONFIG_FLAGS="$ADM_FLAGS_COMPAT"
        USE_SPS=false
        ;;
    sps_gar)
        echo "=== SPS + GAR ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_sps_gar"
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT"
        USE_SPS=true
        ;;
    sps_adm)
        echo "=== SPS + ADM ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_sps_adm"
        CONFIG_FLAGS="$ADM_FLAGS_COMPAT"
        USE_SPS=true
        ;;
    gar_adm)
        echo "=== GAR + ADM ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_gar_adm"
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT $ADM_FLAGS_COMPAT"
        USE_SPS=false
        ;;
    spags)
        echo "=== Full SPAGS (SPS + GAR + ADM) ==="
        OUTPUT="output/_${TIMESTAMP}_${ORGAN}_${VIEWS}views_spags"
        CONFIG_FLAGS="$GAR_FLAGS_COMPAT $ADM_FLAGS_COMPAT"
        USE_SPS=true
        ;;
    *)
        echo "错误: 未知配置 '$CONFIG'"
        echo "可用配置: baseline, sps, gar, adm, sps_gar, sps_adm, gar_adm, spags"
        exit 1
        ;;
esac

# ============================================================================
# SPS 点云检查（使用 density-369 目录已有的密度加权点云）
# ============================================================================

if [ "$USE_SPS" = true ]; then
    if [ ! -f "$SPS_PCD_PATH" ]; then
        echo "错误: SPS 点云不存在: $SPS_PCD_PATH"
        echo "请先在 data/density-369/ 目录生成密度加权点云"
        exit 1
    fi
    echo ">>> 使用 SPS 点云: $SPS_PCD_PATH"
    PLY_FLAG="--ply_path $SPS_PCD_PATH"
else
    PLY_FLAG=""
fi

# ============================================================================
# 训练
# ============================================================================

echo ""
echo "============================================================================"
echo "配置: $CONFIG"
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
器官: $ORGAN
视角: $VIEWS
GPU: $GPU
时间: $(date)

SPS 启用: $USE_SPS
GAR 启用: $(echo "$CONFIG_FLAGS" | grep -q "proximity" && echo "true" || echo "false")
ADM 启用: $(echo "$CONFIG_FLAGS" | grep -q "kplanes" && echo "true" || echo "false")

完整命令:
CUDA_VISIBLE_DEVICES=$GPU python train.py \\
    -s $DATA_PATH \\
    -m $OUTPUT \\
    $COMMON_FLAGS \\
    $CONFIG_FLAGS \\
    $PLY_FLAG
EOF

# 执行训练
CUDA_VISIBLE_DEVICES=$GPU python train.py \
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
