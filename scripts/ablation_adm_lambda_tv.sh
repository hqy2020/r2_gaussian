#!/bin/bash
# ============================================================================
# ADM lambda_tv 消融实验脚本
# ============================================================================
# 测试 5 种不同的 TV 正则化权重，并行运行
# 用法:
#   bash scripts/ablation_adm_lambda_tv.sh
# ============================================================================

set -e

# 取消代理设置
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 时间戳
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# 数据集配置
ORGAN="foot"
VIEWS="3"
DATA_PATH="data/369/${ORGAN}_50_${VIEWS}views.pickle"

# SPS 点云路径（3k 点）
SPS_PCD_PATH="data/density-369/init_${ORGAN}_50_${VIEWS}views.npy"

# 检查数据文件
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据集不存在: $DATA_PATH"
    exit 1
fi

if [ ! -f "$SPS_PCD_PATH" ]; then
    echo "错误: SPS 点云不存在: $SPS_PCD_PATH"
    exit 1
fi

# 公共参数
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000"

# GAR 参数（邻居点数 k=5）
GAR_FLAGS="--enable_binocular_consistency \
    --binocular_loss_weight 0.08 \
    --binocular_max_angle_offset 0.04 \
    --binocular_start_iter 5000 \
    --binocular_warmup_iters 3000 \
    --enable_fsgs_proximity \
    --proximity_threshold 5.0 \
    --proximity_k_neighbors 5 \
    --enable_medical_constraints"

# ADM 基础参数（lambda_tv 由循环设置）
ADM_BASE_FLAGS="--enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --tv_loss_type l2"

# ============================================================================
# 定义 5 种 lambda_tv 配置（分布在 2 张 GPU 上）
# ============================================================================
LAMBDA_TV_VALUES=("0.0005" "0.001" "0.002" "0.005" "0.01")
GPU_IDS=("0" "1" "0" "1" "0")  # 轮流分配到 GPU 0 和 1

echo "============================================================================"
echo "ADM lambda_tv 消融实验"
echo "============================================================================"
echo "时间戳: $TIMESTAMP"
echo "数据集: $DATA_PATH"
echo "SPS 点云: $SPS_PCD_PATH"
echo "测试配置: ${LAMBDA_TV_VALUES[*]}"
echo "============================================================================"

# 创建输出目录
OUTPUT_BASE="output/ablation_adm_lambda_tv_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE"

# 记录实验配置
cat > "${OUTPUT_BASE}/experiment_config.txt" << EOF
ADM lambda_tv 消融实验配置
==========================
时间: $(date)
数据集: $DATA_PATH
SPS 点云: $SPS_PCD_PATH (3k 点)
GAR 邻居点数: 5

测试配置:
EOF

for i in "${!LAMBDA_TV_VALUES[@]}"; do
    echo "  - lambda_tv=${LAMBDA_TV_VALUES[$i]} (GPU ${GPU_IDS[$i]})" >> "${OUTPUT_BASE}/experiment_config.txt"
done

# ============================================================================
# 并行启动 5 个实验
# ============================================================================

PIDS=()

for i in "${!LAMBDA_TV_VALUES[@]}"; do
    LAMBDA_TV=${LAMBDA_TV_VALUES[$i]}
    GPU=${GPU_IDS[$i]}

    # 输出目录（使用下划线替换小数点）
    LAMBDA_TV_SAFE=$(echo "$LAMBDA_TV" | tr '.' '_')
    OUTPUT="${OUTPUT_BASE}/${ORGAN}_${VIEWS}views_adm_tv${LAMBDA_TV_SAFE}"

    echo ""
    echo ">>> 启动实验: lambda_tv=$LAMBDA_TV (GPU $GPU)"
    echo "    输出目录: $OUTPUT"

    mkdir -p "$OUTPUT"

    # 记录单个实验配置
    cat > "${OUTPUT}/config.txt" << EOF
配置: ADM lambda_tv=$LAMBDA_TV
器官: $ORGAN
视角: $VIEWS
GPU: $GPU
时间: $(date)

完整命令:
CUDA_VISIBLE_DEVICES=$GPU python train.py \\
    -s $DATA_PATH \\
    -m $OUTPUT \\
    $COMMON_FLAGS \\
    $GAR_FLAGS \\
    $ADM_BASE_FLAGS \\
    --lambda_plane_tv $LAMBDA_TV \\
    --ply_path $SPS_PCD_PATH
EOF

    # 后台启动训练
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s "$DATA_PATH" \
        -m "$OUTPUT" \
        $COMMON_FLAGS \
        $GAR_FLAGS \
        $ADM_BASE_FLAGS \
        --lambda_plane_tv "$LAMBDA_TV" \
        --ply_path "$SPS_PCD_PATH" \
        > "${OUTPUT}/training.log" 2>&1 &

    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
done

echo ""
echo "============================================================================"
echo "所有实验已启动！"
echo "============================================================================"
echo "进程 PIDs: ${PIDS[*]}"
echo "输出目录: $OUTPUT_BASE"
echo ""
echo "监控命令:"
echo "  tail -f ${OUTPUT_BASE}/*/training.log"
echo ""
echo "查看 GPU 使用:"
echo "  watch -n 1 nvidia-smi"
echo "============================================================================"

# 等待所有进程完成
echo ""
echo "等待所有实验完成..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "  进程 $pid 已完成"
done

echo ""
echo "============================================================================"
echo "所有实验完成！"
echo "============================================================================"

# 提取结果
echo ""
echo "提取实验结果..."
echo ""
echo "| lambda_tv | PSNR | SSIM |"
echo "|-----------|------|------|"

for i in "${!LAMBDA_TV_VALUES[@]}"; do
    LAMBDA_TV=${LAMBDA_TV_VALUES[$i]}
    LAMBDA_TV_SAFE=$(echo "$LAMBDA_TV" | tr '.' '_')
    OUTPUT="${OUTPUT_BASE}/${ORGAN}_${VIEWS}views_adm_tv${LAMBDA_TV_SAFE}"

    # 从日志中提取最终 PSNR 和 SSIM
    if [ -f "${OUTPUT}/training.log" ]; then
        PSNR=$(grep -oP 'PSNR:\s*\K[\d.]+' "${OUTPUT}/training.log" | tail -1)
        SSIM=$(grep -oP 'SSIM:\s*\K[\d.]+' "${OUTPUT}/training.log" | tail -1)
        echo "| $LAMBDA_TV | ${PSNR:-N/A} | ${SSIM:-N/A} |"
    else
        echo "| $LAMBDA_TV | 日志未找到 | - |"
    fi
done

echo ""
echo "详细结果请查看: ${OUTPUT_BASE}/"
