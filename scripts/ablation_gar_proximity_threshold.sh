#!/bin/bash
# ============================================================================
# GAR proximity_threshold 超参数消融实验
# ============================================================================
# 目的: 测试 gar_proximity_threshold 参数对 GAR 性能的影响
#
# 用法:
#   ./scripts/ablation_gar_proximity_threshold.sh           # 运行所有实验（后台并行）
#   ./scripts/ablation_gar_proximity_threshold.sh 1         # 只运行 thresh=1
#   ./scripts/ablation_gar_proximity_threshold.sh 1 3       # 运行 thresh=1,3
#
# 配置:
#   - 数据集: foot_50_3views
#   - 点云: 3k SPS 密度加权采样
#   - 测试值: [1, 3, 5, 7, 10]
#   - 2 GPU 并行执行
# ============================================================================

set -e

# 取消代理设置
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# ============================================================================
# 配置
# ============================================================================
THRESHOLDS=(1 3 5 7 10)
DATA_PATH="data/369/foot_50_3views.pickle"
PCD_PATH="ablation-foot3/init_foot_3000.npy"
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
OUTPUT_BASE="output/ablation_gar_prox_thresh"

# GAR 固定参数（除了 proximity_threshold）
GAR_FIXED_FLAGS="--enable_binocular_consistency \
    --binocular_loss_weight 0.08 \
    --binocular_max_angle_offset 0.04 \
    --binocular_start_iter 5000 \
    --binocular_warmup_iters 3000 \
    --enable_fsgs_proximity \
    --proximity_k_neighbors 5 \
    --enable_medical_constraints"

# 训练参数
TRAIN_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000"

# ============================================================================
# 确定要运行的阈值
# ============================================================================
if [ $# -gt 0 ]; then
    # 使用命令行指定的阈值
    THRESHOLDS=("$@")
    echo ">>> 运行指定的阈值: ${THRESHOLDS[*]}"
else
    echo ">>> 运行所有阈值: ${THRESHOLDS[*]}"
fi

# ============================================================================
# 检查点云文件
# ============================================================================
if [ ! -f "$PCD_PATH" ]; then
    echo "错误: 3k SPS 点云不存在: $PCD_PATH"
    exit 1
fi

echo ""
echo "============================================================================"
echo "GAR proximity_threshold 消融实验"
echo "============================================================================"
echo "数据集: $DATA_PATH"
echo "点云: $PCD_PATH"
echo "测试阈值: ${THRESHOLDS[*]}"
echo "时间戳: $TIMESTAMP"
echo "============================================================================"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE"
mkdir -p "${OUTPUT_BASE}/logs"

# ============================================================================
# 定义训练函数
# ============================================================================
run_experiment() {
    local THRESH=$1
    local GPU=$2

    local OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_thresh_${THRESH}"
    local LOG_FILE="${OUTPUT_BASE}/logs/train_thresh_${THRESH}.log"

    echo "[GPU $GPU] 启动 proximity_threshold=$THRESH"
    echo "  输出: $OUTPUT_DIR"
    echo "  日志: $LOG_FILE"

    mkdir -p "$OUTPUT_DIR"

    # 记录配置
    cat > "${OUTPUT_DIR}/ablation_config.txt" << EOF
GAR proximity_threshold 消融实验
================================
proximity_threshold: $THRESH
数据集: $DATA_PATH
点云: $PCD_PATH (3k SPS)
GPU: $GPU
时间: $(date)

完整命令:
CUDA_VISIBLE_DEVICES=$GPU python train.py \\
    -s $DATA_PATH \\
    -m $OUTPUT_DIR \\
    $TRAIN_FLAGS \\
    $GAR_FIXED_FLAGS \\
    --proximity_threshold $THRESH \\
    --ply_path $PCD_PATH
EOF

    # 执行训练
    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -s "$DATA_PATH" \
        -m "$OUTPUT_DIR" \
        $TRAIN_FLAGS \
        $GAR_FIXED_FLAGS \
        --proximity_threshold "$THRESH" \
        --ply_path "$PCD_PATH" \
        2>&1 | tee "$LOG_FILE"

    echo "[GPU $GPU] 完成 proximity_threshold=$THRESH"
}

# ============================================================================
# 并行执行（2 GPU）
# ============================================================================
# GPU 分配策略: 奇数索引用 GPU 0，偶数索引用 GPU 1

PIDS=()

for i in "${!THRESHOLDS[@]}"; do
    THRESH=${THRESHOLDS[$i]}
    GPU=$((i % 2))  # 0, 1, 0, 1, 0 ...

    # 后台运行
    run_experiment "$THRESH" "$GPU" &
    PIDS+=($!)

    # 如果两个 GPU 都启动了一个任务，等待它们完成再启动下一批
    if [ ${#PIDS[@]} -eq 2 ]; then
        echo ""
        echo ">>> 等待当前批次完成 (PID: ${PIDS[*]})..."
        wait "${PIDS[@]}"
        PIDS=()
        echo ">>> 当前批次完成"
        echo ""
    fi
done

# 等待剩余任务
if [ ${#PIDS[@]} -gt 0 ]; then
    echo ">>> 等待最后一批完成 (PID: ${PIDS[*]})..."
    wait "${PIDS[@]}"
fi

echo ""
echo "============================================================================"
echo "所有实验完成！"
echo "结果目录: $OUTPUT_BASE"
echo "============================================================================"

# ============================================================================
# 收集结果
# ============================================================================
echo ""
echo ">>> 收集实验结果..."
echo ""
echo "| proximity_threshold | PSNR (dB) | SSIM |"
echo "|---------------------|-----------|------|"

for THRESH in "${THRESHOLDS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_thresh_${THRESH}"
    EVAL_FILE="${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"

    if [ -f "$EVAL_FILE" ]; then
        PSNR=$(grep "psnr:" "$EVAL_FILE" | head -1 | awk '{print $2}')
        SSIM=$(grep "ssim:" "$EVAL_FILE" | head -1 | awk '{print $2}')
        printf "| %19s | %9s | %4s |\n" "$THRESH" "$PSNR" "$SSIM"
    else
        printf "| %19s | %9s | %4s |\n" "$THRESH" "N/A" "N/A"
    fi
done

echo ""
echo "Baseline 参考: PSNR 28.4873, SSIM 0.9005"
