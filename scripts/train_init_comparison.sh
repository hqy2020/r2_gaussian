#!/bin/bash
# 对比四种初始化方法的完整训练实验
# 数据集: Foot-3 views
# 目标: 验证初始化对最终 PSNR/SSIM 的影响

set -e  # 遇到错误立即退出

echo "=========================================="
echo "R²-Gaussian 初始化方法训练对比实验"
echo "=========================================="
echo "数据集: Foot-3 views"
echo "迭代次数: 30,000"
echo "初始化方法: baseline, de-init, smart, combined"
echo ""

# 基础参数
DATA_PATH="data/369/foot_50_3views.pickle"
ITERATIONS=30000
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)

# 训练函数
run_training() {
    local method=$1
    local init_file=$2
    local output_dir="output/${TIMESTAMP}_foot_3views_init_${method}"

    echo ""
    echo "=========================================="
    echo "训练: ${method}"
    echo "=========================================="
    echo "初始化文件: ${init_file}"
    echo "输出目录: ${output_dir}"
    echo "开始时间: $(date)"
    echo ""

    # 运行训练
    conda run -n r2_gaussian_new python train.py \
        -s "$DATA_PATH" \
        --ply_path "$init_file" \
        --iterations "$ITERATIONS" \
        --save_iterations 7000 15000 30000 \
        --test_iterations 1000 7000 15000 30000 \
        --checkpoint_iterations 7000 15000 30000 \
        -m "$output_dir" \
        2>&1 | tee "${output_dir}_train.log"

    echo ""
    echo "✅ ${method} 训练完成"
    echo "完成时间: $(date)"
    echo "日志文件: ${output_dir}_train.log"
    echo ""

    # 提取最终结果
    if [ -f "${output_dir}/eval/iter_030000/eval2d_render_test.yml" ]; then
        echo "最终评估结果:"
        grep -E "psnr|ssim" "${output_dir}/eval/iter_030000/eval2d_render_test.yml" | head -2
    fi
    echo ""
}

# 1. Baseline 初始化
echo "========================================"
echo "步骤 1/4: Baseline 训练"
echo "========================================"
run_training "baseline" "init_comparison_test/init_baseline.npy"

# 2. De-Init 降噪
echo "========================================"
echo "步骤 2/4: De-Init 训练"
echo "========================================"
run_training "denoise" "init_comparison_test/init_denoise.npy"

# 3. Smart Sampling
echo "========================================"
echo "步骤 3/4: Smart Sampling 训练"
echo "========================================"
run_training "smart" "init_comparison_test/init_smart.npy"

# 4. Combined
echo "========================================"
echo "步骤 4/4: Combined 训练"
echo "========================================"
run_training "combined" "init_comparison_test/init_combined.npy"

# 汇总结果
echo ""
echo "=========================================="
echo "📊 所有训练完成！汇总结果"
echo "=========================================="
echo ""

for method in baseline denoise smart combined; do
    output_dir="output/${TIMESTAMP}_foot_3views_init_${method}"
    eval_file="${output_dir}/eval/iter_030000/eval2d_render_test.yml"

    echo "方法: ${method}"
    if [ -f "$eval_file" ]; then
        psnr=$(grep "psnr:" "$eval_file" | head -1 | awk '{print $2}')
        ssim=$(grep "ssim:" "$eval_file" | head -1 | awk '{print $2}')
        echo "  PSNR: ${psnr} dB"
        echo "  SSIM: ${ssim}"
        echo "  输出: ${output_dir}"
    else
        echo "  ⚠️  评估文件不存在"
    fi
    echo ""
done

echo "=========================================="
echo "实验标识: ${TIMESTAMP}"
echo "=========================================="
