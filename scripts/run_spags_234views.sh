#!/bin/bash
# ============================================================================
# SPAGS 实验：2/3/4 视角测试
# ============================================================================
# 15 个实验：5 器官 x 3 视角 (2/3/4)
# 三个创新点：SPS-50k + GAR (threshold=7) + ADM (lambda_tv=0.002)
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
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
OUTPUT_BASE="output/spags_234views_${TIMESTAMP}"
DATA_DIR="data/234"

ORGANS=(chest foot head abdomen pancreas)
VIEWS=(2 3 4)

# 完整 SPAGS 参数（三个创新点）
# - SPS: 使用 50k 密度加权初始化点云 (已在 init_*.npy 中)
# - GAR: binocular consistency + proximity densification (threshold=7)
# - ADM: K-Planes (lambda_tv=0.002)
SPAGS_FLAGS="--enable_binocular_consistency \
    --binocular_loss_weight 0.08 \
    --binocular_max_angle_offset 0.04 \
    --binocular_start_iter 5000 \
    --binocular_warmup_iters 3000 \
    --enable_fsgs_proximity \
    --proximity_threshold 5 \
    --proximity_k_neighbors 5 \
    --enable_medical_constraints \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --lambda_plane_tv 0.002"

TRAIN_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000"

# ============================================================================
# 创建输出目录
# ============================================================================
mkdir -p "$OUTPUT_BASE/logs"

echo "============================================================================"
echo "SPAGS 实验 (2/3/4 视角)"
echo "============================================================================"
echo "时间戳: $TIMESTAMP"
echo "输出目录: $OUTPUT_BASE"
echo "数据目录: $DATA_DIR"
echo "实验数量: 15 (5 器官 x 3 视角)"
echo "并发模式: 全并发 (2 GPU)"
echo ""
echo "创新点配置:"
echo "  - SPS: 50k 密度加权初始化"
echo "  - GAR: proximity_threshold=5, binocular_loss_weight=0.08"
echo "  - ADM: K-Planes (lambda_tv=0.002)"
echo "============================================================================"
echo ""

# ============================================================================
# 启动所有 15 个任务
# ============================================================================
idx=0
PIDS=()

for organ in "${ORGANS[@]}"; do
    for views in "${VIEWS[@]}"; do
        GPU=$((idx % 2))  # 0,1,0,1...

        DATA_FILE="${DATA_DIR}/${organ}_50_${views}views.pickle"
        PCD_FILE="${DATA_DIR}/init_${organ}_50_${views}views.npy"
        OUTPUT_DIR="${OUTPUT_BASE}/${organ}_${views}views"
        LOG_FILE="${OUTPUT_BASE}/logs/${organ}_${views}views.log"

        # 检查数据文件
        if [ ! -f "$DATA_FILE" ]; then
            echo "[警告] 数据文件不存在: $DATA_FILE"
            continue
        fi

        if [ ! -f "$PCD_FILE" ]; then
            echo "[警告] 点云文件不存在: $PCD_FILE"
            continue
        fi

        mkdir -p "$OUTPUT_DIR"

        # 启动训练
        CUDA_VISIBLE_DEVICES=$GPU python train.py \
            -s "$DATA_FILE" \
            -m "$OUTPUT_DIR" \
            --ply_path "$PCD_FILE" \
            --proximity_organ_type "$organ" \
            $SPAGS_FLAGS \
            $TRAIN_FLAGS \
            > "$LOG_FILE" 2>&1 &

        PID=$!
        PIDS+=($PID)
        echo "[GPU $GPU] 启动 ${organ}_${views}views (PID: $PID)"

        idx=$((idx + 1))
    done
done

echo ""
echo "============================================================================"
echo "已启动 ${#PIDS[@]} 个训练任务"
echo "日志目录: ${OUTPUT_BASE}/logs/"
echo "监控命令: tail -f ${OUTPUT_BASE}/logs/*.log"
echo "GPU 监控: watch -n 5 nvidia-smi"
echo "============================================================================"

# 等待所有任务完成
wait "${PIDS[@]}"

echo ""
echo "============================================================================"
echo "所有任务完成！"
echo "============================================================================"

# ============================================================================
# 收集结果
# ============================================================================
echo ""
echo ">>> 收集实验结果..."
echo ""

RESULT_FILE="${OUTPUT_BASE}/results_summary.md"

cat > "$RESULT_FILE" << EOF
# SPAGS 2/3/4 视角实验结果汇总

## 配置
- **SPS**: 50k 密度加权初始化
- **GAR**: proximity_threshold=5, binocular_loss_weight=0.08
- **ADM**: K-Planes (lambda_tv=0.002)
- 时间戳: $TIMESTAMP

## 2D 渲染指标 (render_test)

| 器官 | 2views PSNR | 2views SSIM | 3views PSNR | 3views SSIM | 4views PSNR | 4views SSIM |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
EOF

# 收集 2D 结果
for organ in "${ORGANS[@]}"; do
    printf "| %-8s" "$organ" >> "$RESULT_FILE"
    for views in "${VIEWS[@]}"; do
        OUTPUT_DIR="${OUTPUT_BASE}/${organ}_${views}views"
        EVAL_FILE="${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"

        if [ -f "$EVAL_FILE" ]; then
            PSNR=$(grep "psnr:" "$EVAL_FILE" | head -1 | awk '{print $2}')
            SSIM=$(grep "ssim:" "$EVAL_FILE" | head -1 | awk '{print $2}')
        else
            PSNR="N/A"
            SSIM="N/A"
        fi

        printf " | %s | %s" "$PSNR" "$SSIM" >> "$RESULT_FILE"
    done
    echo " |" >> "$RESULT_FILE"
done

cat >> "$RESULT_FILE" << EOF

## 3D 重建指标 (reconstruction)

| 器官 | 2views PSNR | 2views SSIM | 3views PSNR | 3views SSIM | 4views PSNR | 4views SSIM |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
EOF

# 收集 3D 结果
for organ in "${ORGANS[@]}"; do
    printf "| %-8s" "$organ" >> "$RESULT_FILE"
    for views in "${VIEWS[@]}"; do
        OUTPUT_DIR="${OUTPUT_BASE}/${organ}_${views}views"
        EVAL_FILE="${OUTPUT_DIR}/eval/iter_030000/eval3d_reconstruction.yml"

        if [ -f "$EVAL_FILE" ]; then
            PSNR=$(grep "psnr:" "$EVAL_FILE" | head -1 | awk '{print $2}')
            SSIM=$(grep "ssim:" "$EVAL_FILE" | head -1 | awk '{print $2}')
        else
            PSNR="N/A"
            SSIM="N/A"
        fi

        printf " | %s | %s" "$PSNR" "$SSIM" >> "$RESULT_FILE"
    done
    echo " |" >> "$RESULT_FILE"
done

echo ""
echo "结果已保存到: $RESULT_FILE"
cat "$RESULT_FILE"
