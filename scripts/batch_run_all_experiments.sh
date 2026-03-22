#!/bin/bash
# 75 组对比实验批量运行脚本
# 5 方法 (R2GS, XGS, DNGS, CORGS, FSGS) × 5 器官 × 3 视角
# 用法: bash batch_run_all_experiments.sh [--method METHOD] [--organ ORGAN] [--views VIEWS] [--skip-existing]

set -euo pipefail

# === 配置 ===
PROJECT_DIR="/root/r2_gaussian"
DATA_DIR="${PROJECT_DIR}/data"
EXP_BASE="/root/experiments/results"
CONDA_ENV="r2gs"
LOG_DIR="/root/experiments/logs"

METHODS=("R2GS" "XGS" "DNGS" "CORGS" "FSGS")
ORGANS=("chest" "foot" "head" "abdomen" "pancreas")
VIEWS=("3" "6" "9")

SKIP_EXISTING=false
FILTER_METHOD=""
FILTER_ORGAN=""
FILTER_VIEWS=""

# === 参数解析 ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)        FILTER_METHOD="$2"; shift 2 ;;
        --organ)         FILTER_ORGAN="$2"; shift 2 ;;
        --views)         FILTER_VIEWS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--method M] [--organ O] [--views V] [--skip-existing]"
            exit 0 ;;
        *) shift ;;
    esac
done

mkdir -p "$LOG_DIR"

# === 构建训练命令 ===
get_train_cmd() {
    local method="$1" organ="$2" views="$3"
    local data_file="${DATA_DIR}/${organ}_50_${views}views.pickle"
    local output_dir="${EXP_BASE}/${method}/${organ}_${views}views"

    case "$method" in
        R2GS)
            echo "python ${PROJECT_DIR}/train.py -s ${data_file} -m ${output_dir} --gaussiansN 1"
            ;;
        XGS)
            # X-Gaussian: X 射线方法，检查其 conda 环境
            echo "python /root/comparison_methods/X-Gaussian/train.py -s ${data_file} -m ${output_dir}"
            ;;
        DNGS)
            # DNGaussian 适配: 深度正则化集成到 R2GS
            echo "python ${PROJECT_DIR}/train.py -s ${data_file} -m ${output_dir} --gaussiansN 1 --enable_depth --depth_loss_weight 0.1 --depth_loss_type pearson"
            ;;
        CORGS)
            # CoR-GS 适配: 协同正则化（多高斯模型）
            echo "python ${PROJECT_DIR}/train.py -s ${data_file} -m ${output_dir} --gaussiansN 2 --multi_gaussian_weight 0.1"
            ;;
        FSGS)
            # FSGS: 使用已有模块
            echo "python ${PROJECT_DIR}/train.py -s ${data_file} -m ${output_dir} --gaussiansN 1 --enable_fsgs_pseudo_labels --enable_fsgs_proximity_guided --enable_fsgs_complete_system"
            ;;
    esac
}

# === 检查是否已完成 ===
is_completed() {
    local method="$1" organ="$2" views="$3"
    local eval_file="${EXP_BASE}/${method}/${organ}_${views}views/eval/iter_030000/eval2d_render_test.yml"
    [[ -f "$eval_file" ]]
}

# === 主循环 ===
total=0
skipped=0
completed=0
failed=0

echo "============================================"
echo "  批量实验启动: $(date)"
echo "============================================"

for method in "${METHODS[@]}"; do
    [[ -n "$FILTER_METHOD" && "$method" != "$FILTER_METHOD" ]] && continue

    for organ in "${ORGANS[@]}"; do
        [[ -n "$FILTER_ORGAN" && "$organ" != "$FILTER_ORGAN" ]] && continue

        for views in "${VIEWS[@]}"; do
            [[ -n "$FILTER_VIEWS" && "$views" != "$FILTER_VIEWS" ]] && continue

            total=$((total + 1))
            exp_name="${method}_${organ}_${views}views"
            log_file="${LOG_DIR}/${exp_name}.log"

            echo ""
            echo "--- [$total] $exp_name ---"

            # 跳过已完成
            if $SKIP_EXISTING && is_completed "$method" "$organ" "$views"; then
                echo "  [SKIP] Already completed"
                skipped=$((skipped + 1))
                continue
            fi

            # 检查数据文件
            data_file="${DATA_DIR}/${organ}_50_${views}views.pickle"
            if [[ ! -f "$data_file" ]]; then
                echo "  [SKIP] Data file missing: $data_file"
                skipped=$((skipped + 1))
                continue
            fi

            # 创建输出目录
            output_dir="${EXP_BASE}/${method}/${organ}_${views}views"
            mkdir -p "$output_dir"

            # 获取训练命令
            train_cmd=$(get_train_cmd "$method" "$organ" "$views")

            echo "  Start: $(date '+%H:%M:%S')"
            echo "  CMD: $train_cmd"

            # 执行训练
            if eval "conda run -n ${CONDA_ENV} ${train_cmd}" > "$log_file" 2>&1; then
                echo "  [DONE] $(date '+%H:%M:%S')"
                completed=$((completed + 1))

                # 自动运行测试
                if [[ "$method" == "R2GS" || "$method" == "DNGS" || "$method" == "CORGS" || "$method" == "FSGS" ]]; then
                    echo "  Running test.py..."
                    conda run -n "${CONDA_ENV}" python "${PROJECT_DIR}/test.py" -m "$output_dir" >> "$log_file" 2>&1 || true
                fi
            else
                echo "  [FAIL] Check log: $log_file"
                failed=$((failed + 1))
            fi
        done
    done
done

echo ""
echo "============================================"
echo "  实验完成: $(date)"
echo "  Total: $total | Done: $completed | Skipped: $skipped | Failed: $failed"
echo "============================================"

# 生成汇总
if [[ $completed -gt 0 ]]; then
    echo ""
    echo "Generating metrics summary..."
    conda run -n "${CONDA_ENV}" python "${PROJECT_DIR}/scripts/extract_metrics_all.py" "$EXP_BASE" --markdown 2>/dev/null || echo "Metrics extraction failed (install tabulate if needed)"
fi
