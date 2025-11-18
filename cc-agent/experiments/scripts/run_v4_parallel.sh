#!/bin/bash
# V4+ FSGS 优化实验 - 多 GPU 并行执行脚本
# 生成时间: 2025-11-18
# 基于 v2 baseline (PSNR 28.50 dB) 进行优化
# 决策: ABAAA (全部8个实验 + 多GPU并行 + 修复bug + Early Stopping + 失败后转其他器官)

set -e  # 遇到错误立即退出

# ============================================================================
# 配置区域
# ============================================================================

# CUDA 环境
CONDA_ENV="r2_gaussian_new"

# 基础路径
DATA_PATH="data/369/foot_50_3views.pickle"
OUTPUT_BASE="output"
DATE_PREFIX="2025_11_19_foot_3views"

# 基础参数（继承自 v2，确保无干扰项）
BASE_ARGS="
  -s ${DATA_PATH} \
  --iterations 30000 \
  --eval \
  --enable_fsgs_proximity \
  --enable_medical_constraints \
  --proximity_organ_type foot \
  --fsgs_start_iter 2000 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000
"

# ⚠️ 确保干扰项禁用（虽然默认已经是 false，但明确指定以防万一）
# 注意：这些参数在代码中默认就是 false，无需额外传递
# 如果需要明确禁用，可以在 train.py 中添加对应的命令行参数

# Early Stopping 参数
EARLY_STOP_ITER=10000
EARLY_STOP_PSNR=28.0

# ============================================================================
# GPU 分配（仅使用 GPU 1）
# ============================================================================

# 8 个实验全部在 GPU 1 上顺序执行
# 预计时间：32-64 小时（每个实验约 4-8 小时）
declare -A GPU_ASSIGNMENT
GPU_ASSIGNMENT["v4_tv_0.10"]=1
GPU_ASSIGNMENT["v4_tv_0.12"]=1
GPU_ASSIGNMENT["v4_k_5"]=1
GPU_ASSIGNMENT["v4_tau_7.0"]=1
GPU_ASSIGNMENT["v4_densify_10k"]=1
GPU_ASSIGNMENT["v4_grad_3e-4"]=1
GPU_ASSIGNMENT["v4_dssim_0.30"]=1
GPU_ASSIGNMENT["v4_cap_180k"]=1

# ============================================================================
# 实验定义
# ============================================================================

declare -A EXPERIMENTS

# 实验 1: TV 正则化 0.10
EXPERIMENTS["v4_tv_0.10"]="--lambda_tv 0.10"

# 实验 2: TV 正则化 0.12
EXPERIMENTS["v4_tv_0.12"]="--lambda_tv 0.12"

# 实验 3: 邻居数减少到 5
EXPERIMENTS["v4_k_5"]="--proximity_k_neighbors 5"

# 实验 4: 邻近阈值收紧到 7.0
EXPERIMENTS["v4_tau_7.0"]="--proximity_threshold 7.0"

# 实验 5: 密集化提前停止到 10k
EXPERIMENTS["v4_densify_10k"]="--densify_until_iter 10000"

# 实验 6: 密集化梯度阈值提高到 3e-4
EXPERIMENTS["v4_grad_3e-4"]="--densify_grad_threshold 3e-4"

# 实验 7: DSSIM 权重提高到 0.30
EXPERIMENTS["v4_dssim_0.30"]="--lambda_dssim 0.30"

# 实验 8: 高斯点数上限降低到 180k
EXPERIMENTS["v4_cap_180k"]="--max_num_gaussians 180000"

# ============================================================================
# Early Stopping 检查函数
# ============================================================================

check_early_stop() {
    local exp_name=$1
    local model_path="${OUTPUT_BASE}/${DATE_PREFIX}_${exp_name}"
    local eval_file="${model_path}/eval/iter_${EARLY_STOP_ITER}/eval2d_render_test.yml"

    if [ -f "${eval_file}" ]; then
        local psnr=$(grep "^psnr_2d:" "${eval_file}" | awk '{print $2}')
        if [ ! -z "${psnr}" ]; then
            # 使用 bc 进行浮点数比较
            if (( $(echo "${psnr} < ${EARLY_STOP_PSNR}" | bc -l) )); then
                echo "⚠️ Early Stopping: ${exp_name} PSNR ${psnr} < ${EARLY_STOP_PSNR} at iter ${EARLY_STOP_ITER}"
                return 1  # 触发 Early Stopping
            fi
        fi
    fi
    return 0  # 继续训练
}

# ============================================================================
# 单个实验执行函数
# ============================================================================

run_experiment() {
    local exp_name=$1
    local exp_args=$2
    local gpu_id=${GPU_ASSIGNMENT[$exp_name]}
    local model_path="${OUTPUT_BASE}/${DATE_PREFIX}_${exp_name}"
    local log_file="${OUTPUT_BASE}/${DATE_PREFIX}_${exp_name}_train.log"

    echo "========================================"
    echo "开始实验: ${exp_name}"
    echo "GPU: ${gpu_id}"
    echo "输出路径: ${model_path}"
    echo "========================================"

    # 设置 CUDA 可见设备
    export CUDA_VISIBLE_DEVICES=${gpu_id}

    # 执行训练（后台运行）
    conda run -n ${CONDA_ENV} python train.py \
        ${BASE_ARGS} \
        ${exp_args} \
        -m ${model_path} \
        > ${log_file} 2>&1 &

    local pid=$!
    echo "实验 ${exp_name} 启动，PID: ${pid}, GPU: ${gpu_id}"

    # 等待进程完成（带 Early Stopping 检查）
    while kill -0 ${pid} 2>/dev/null; do
        sleep 600  # 每 10 分钟检查一次

        # 检查是否到达 Early Stopping 检查点
        if check_early_stop "${exp_name}"; then
            :  # 继续等待
        else
            # 触发 Early Stopping，终止训练
            echo "执行 Early Stopping，终止实验 ${exp_name}"
            kill -9 ${pid} 2>/dev/null || true
            break
        fi
    done

    wait ${pid} 2>/dev/null || true
    echo "✅ 实验 ${exp_name} 完成"
}

# ============================================================================
# 主执行流程
# ============================================================================

echo "============================================"
echo "V4+ FSGS 优化实验 - GPU 1 顺序执行"
echo "============================================"
echo "基于 baseline: 2025_11_18_foot_3views_fsgs_fixed_v2"
echo "PSNR: 28.50 dB, SSIM: 0.9015"
echo "目标: PSNR ≥ 28.60 dB, 泛化差距 < 20 dB"
echo ""
echo "实验配置:"
echo "  - 总实验数: 8"
echo "  - GPU 分配: 仅使用 GPU 1（顺序执行）"
echo "  - 预计时间: 32-64 小时"
echo "  - Early Stopping: iter ${EARLY_STOP_ITER}, PSNR < ${EARLY_STOP_PSNR}"
echo "  - 干扰项: CoR-GS/SSS/GraphLaplacian 已禁用"
echo "============================================"
echo ""

# 注意：每个实验已在 run_experiment() 中使用 conda run，无需手动激活环境

# 所有实验在 GPU 1 上顺序执行
run_experiment "v4_tv_0.10" "${EXPERIMENTS[v4_tv_0.10]}"
run_experiment "v4_tv_0.12" "${EXPERIMENTS[v4_tv_0.12]}"
run_experiment "v4_k_5" "${EXPERIMENTS[v4_k_5]}"
run_experiment "v4_tau_7.0" "${EXPERIMENTS[v4_tau_7.0]}"
run_experiment "v4_densify_10k" "${EXPERIMENTS[v4_densify_10k]}"
run_experiment "v4_grad_3e-4" "${EXPERIMENTS[v4_grad_3e-4]}"
run_experiment "v4_dssim_0.30" "${EXPERIMENTS[v4_dssim_0.30]}"
run_experiment "v4_cap_180k" "${EXPERIMENTS[v4_cap_180k]}"

echo ""
echo "============================================"
echo "✅ 所有实验完成！"
echo "============================================"
echo ""
echo "下一步操作:"
echo "1. 查看结果汇总:"
echo "   python cc-agent/experiments/scripts/summarize_v4_results.py"
echo ""
echo "2. 查看单个实验日志:"
echo "   tail -f output/2025_11_19_foot_3views_v4_*/train.log"
echo ""
echo "3. 比较实验结果:"
echo "   cat cc-agent/experiments/v4_results_summary.md"
echo "============================================"
