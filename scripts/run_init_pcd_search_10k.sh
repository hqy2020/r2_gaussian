#!/bin/bash
# Init-PCD 超参数搜索 - Phase 1: 10k 粗筛
# 使用两个 GPU 并行训练 18 个实验 (9 配置 × 2 场景)

# 配置
PROJECT_ROOT="/home/qyhu/Documents/r2_ours/r2_gaussian"
SEARCH_DIR="$PROJECT_ROOT/data/init-pcd-search"
OUTPUT_BASE="$PROJECT_ROOT/output/init_pcd_search"
ITERATIONS=10000
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

echo "============================================"
echo "Init-PCD Hyperparameter Search - Phase 1"
echo "============================================"
echo "Timestamp: $TIMESTAMP"
echo "Iterations: $ITERATIONS"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 定义所有任务
declare -a TASKS=(
    "exp01:foot_3:foot_50_3views"
    "exp01:abdomen_9:abdomen_50_9views"
    "exp02:foot_3:foot_50_3views"
    "exp02:abdomen_9:abdomen_50_9views"
    "exp03:foot_3:foot_50_3views"
    "exp03:abdomen_9:abdomen_50_9views"
    "exp04:foot_3:foot_50_3views"
    "exp04:abdomen_9:abdomen_50_9views"
    "exp05:foot_3:foot_50_3views"
    "exp05:abdomen_9:abdomen_50_9views"
    "exp06:foot_3:foot_50_3views"
    "exp06:abdomen_9:abdomen_50_9views"
    "exp07:foot_3:foot_50_3views"
    "exp07:abdomen_9:abdomen_50_9views"
    "exp08:foot_3:foot_50_3views"
    "exp08:abdomen_9:abdomen_50_9views"
    "exp09:foot_3:foot_50_3views"
    "exp09:abdomen_9:abdomen_50_9views"
)

echo "Total tasks: ${#TASKS[@]}"
echo ""

# 运行单个任务
run_task() {
    local task=$1
    local gpu=$2

    IFS=':' read -r exp scene_id scene_name <<< "$task"

    local data_path="$SEARCH_DIR/$exp/${scene_name}.pickle"
    local ply_path="$SEARCH_DIR/$exp/init_${scene_name}.npy"
    local output_dir="$OUTPUT_BASE/${TIMESTAMP}_${exp}_${scene_id}_10k"
    local log_file="${output_dir}.log"

    mkdir -p "$output_dir"

    echo "[GPU $gpu] Starting: $exp $scene_id"

    CUDA_VISIBLE_DEVICES=$gpu python "$PROJECT_ROOT/train.py" \
        -s "$data_path" \
        -m "$output_dir" \
        --ply_path "$ply_path" \
        --gaussiansN 1 \
        --iterations $ITERATIONS \
        --test_iterations 5000 10000 \
        --save_iterations 10000 \
        > "$log_file" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU $gpu] Completed: $exp $scene_id"
    else
        echo "[GPU $gpu] FAILED: $exp $scene_id (exit code: $status)"
    fi
    return $status
}

# 并行调度：每个 GPU 运行一个任务队列
run_gpu_queue() {
    local gpu=$1
    shift
    local tasks=("$@")

    for task in "${tasks[@]}"; do
        run_task "$task" "$gpu"
    done
}

# 将任务分配到两个 GPU
GPU0_TASKS=()
GPU1_TASKS=()
for i in "${!TASKS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_TASKS+=("${TASKS[$i]}")
    else
        GPU1_TASKS+=("${TASKS[$i]}")
    fi
done

echo "GPU 0 tasks: ${#GPU0_TASKS[@]}"
echo "GPU 1 tasks: ${#GPU1_TASKS[@]}"
echo ""
echo "Starting parallel training..."
echo ""

# 并行运行两个 GPU 队列
run_gpu_queue 0 "${GPU0_TASKS[@]}" &
PID0=$!

run_gpu_queue 1 "${GPU1_TASKS[@]}" &
PID1=$!

# 等待两个队列完成
wait $PID0
wait $PID1

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Timestamp: $TIMESTAMP"
echo "Results saved in: $OUTPUT_BASE"
echo ""
echo "Run analysis:"
echo "  python scripts/analyze_init_pcd_search.py --timestamp $TIMESTAMP"
echo "============================================"
