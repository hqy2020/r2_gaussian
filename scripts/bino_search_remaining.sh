#!/bin/bash
# Bino 超参数搜索 - 补充剩余实验
# 用法:
#   ./scripts/bino_search_remaining.sh foot 0    # GPU 0 跑 Foot-3
#   ./scripts/bino_search_remaining.sh abdomen 1 # GPU 1 跑 Abdomen-9

set -e

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# ============== 参数解析 ==============
ORGAN=${1:-foot}
GPU=${2:-0}

if [ "$ORGAN" = "foot" ]; then
    VIEWS=3
    DATA_PATH="data/369/foot_50_3views.pickle"
    BASELINE_PSNR=28.487
elif [ "$ORGAN" = "abdomen" ]; then
    VIEWS=9
    DATA_PATH="data/369/abdomen_50_9views.pickle"
    BASELINE_PSNR=36.948
else
    echo "错误: 未知器官 $ORGAN (支持: foot, abdomen)"
    exit 1
fi

# ============== 搜索空间定义 ==============
LOSS_WEIGHTS=(0.08 0.15 0.25)
ANGLE_OFFSETS=(0.04 0.06 0.10)
START_ITERS=(5000 7000)

# 固定参数
WARMUP=3000
SMOOTH_WEIGHT=0.05
ITERATIONS=30000

# ============== 输出目录 ==============
SEARCH_DIR="output/bino_search_20251128"
mkdir -p "$SEARCH_DIR"

LOG_FILE="$SEARCH_DIR/${ORGAN}_${VIEWS}views.log"
RESULT_FILE="$SEARCH_DIR/${ORGAN}_${VIEWS}views_results.csv"

# 初始化结果文件（如果不存在）
if [ ! -f "$RESULT_FILE" ]; then
    echo "exp_id,loss_weight,angle_offset,start_iter,psnr,ssim,delta_psnr" > "$RESULT_FILE"
fi

echo "==========================================" | tee -a "$LOG_FILE"
echo "Bino 超参数搜索 (补充) - ${ORGAN}-${VIEWS}views" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "GPU: $GPU" | tee -a "$LOG_FILE"
echo "数据: $DATA_PATH" | tee -a "$LOG_FILE"
echo "Baseline PSNR: $BASELINE_PSNR" | tee -a "$LOG_FILE"
echo "搜索空间: 3×3×2 = 18 组参数" | tee -a "$LOG_FILE"
echo "输出目录: $SEARCH_DIR" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============== 开始搜索 ==============
EXP_ID=0
COMPLETED=0
SKIPPED=0
TOTAL_EXPS=$((${#LOSS_WEIGHTS[@]} * ${#ANGLE_OFFSETS[@]} * ${#START_ITERS[@]}))

for LW in "${LOSS_WEIGHTS[@]}"; do
    for AO in "${ANGLE_OFFSETS[@]}"; do
        for SI in "${START_ITERS[@]}"; do
            EXP_ID=$((EXP_ID + 1))

            # 生成实验名称
            EXP_NAME="lw${LW}_ao${AO}_si${SI}"
            OUTPUT_PATH="$SEARCH_DIR/${ORGAN}_${VIEWS}views_${EXP_NAME}"

            # 检查是否已完成
            EVAL_FILE="$OUTPUT_PATH/eval/iter_030000/eval2d_render_test.yml"
            if [ -f "$EVAL_FILE" ]; then
                PSNR=$(grep "^psnr_2d:" "$EVAL_FILE" | awk '{print $2}')
                SSIM=$(grep "^ssim_2d:" "$EVAL_FILE" | awk '{print $2}')
                DELTA=$(echo "$PSNR - $BASELINE_PSNR" | bc -l)
                echo "[$EXP_ID/$TOTAL_EXPS] $EXP_NAME [跳过] 已完成: PSNR=$PSNR (Δ=$DELTA)" | tee -a "$LOG_FILE"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo "[$EXP_ID/$TOTAL_EXPS] $EXP_NAME" | tee -a "$LOG_FILE"
            echo "  loss_weight=$LW, angle_offset=$AO, start_iter=$SI" | tee -a "$LOG_FILE"

            # 运行训练
            START_TIME=$(date +%s)

            CUDA_VISIBLE_DEVICES=$GPU python train.py \
                -s "$DATA_PATH" \
                -m "$OUTPUT_PATH" \
                --iterations $ITERATIONS \
                --gaussiansN 1 \
                --enable_binocular_consistency \
                --binocular_loss_weight $LW \
                --binocular_max_angle_offset $AO \
                --binocular_start_iter $SI \
                --binocular_warmup_iters $WARMUP \
                --binocular_smooth_weight $SMOOTH_WEIGHT \
                --test_iterations 10000 20000 30000 \
                2>&1 | tee "$OUTPUT_PATH/training.log"

            END_TIME=$(date +%s)
            ELAPSED=$((END_TIME - START_TIME))

            # 读取结果
            if [ -f "$EVAL_FILE" ]; then
                PSNR=$(grep "^psnr_2d:" "$EVAL_FILE" | awk '{print $2}')
                SSIM=$(grep "^ssim_2d:" "$EVAL_FILE" | awk '{print $2}')
                DELTA=$(echo "$PSNR - $BASELINE_PSNR" | bc -l)
                echo "  完成: PSNR=$PSNR, SSIM=$SSIM (Δ=$DELTA) [${ELAPSED}s]" | tee -a "$LOG_FILE"
                echo "$EXP_ID,$LW,$AO,$SI,$PSNR,$SSIM,$DELTA" >> "$RESULT_FILE"
                COMPLETED=$((COMPLETED + 1))
            else
                echo "  [错误] 评估文件未生成" | tee -a "$LOG_FILE"
                echo "$EXP_ID,$LW,$AO,$SI,ERROR,ERROR,ERROR" >> "$RESULT_FILE"
            fi

            echo "" | tee -a "$LOG_FILE"
        done
    done
done

echo "==========================================" | tee -a "$LOG_FILE"
echo "搜索完成！" | tee -a "$LOG_FILE"
echo "跳过: $SKIPPED, 新完成: $COMPLETED" | tee -a "$LOG_FILE"
echo "结果文件: $RESULT_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 显示最佳结果
echo "" | tee -a "$LOG_FILE"
echo "Top 5 结果 (按 delta_psnr 排序):" | tee -a "$LOG_FILE"
tail -n +2 "$RESULT_FILE" | grep -v ERROR | sort -t',' -k7 -n -r | head -5 | tee -a "$LOG_FILE"
