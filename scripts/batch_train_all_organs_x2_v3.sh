#!/bin/bash

# X²-Gaussian v3 批量训练脚本
# 目标：使用成功的 Foot 配置训练所有 5 个器官（3 views）
# 执行方式：串行执行（避免 GPU 内存冲突）
#
# 使用方法：
#   bash scripts/batch_train_all_organs_x2_v3.sh
#
# 预计时间：
#   每个器官 ~35 分钟 × 4 = ~2.5 小时

echo "========================================"
echo "X²-Gaussian v3 批量训练 - 所有器官"
echo "========================================"
echo "开始时间: $(date)"
echo ""
echo "训练计划："
echo "  1. Chest   (baseline: 26.506 PSNR)"
echo "  2. Head    (baseline: 26.692 PSNR)"
echo "  3. Abdomen (baseline: 29.290 PSNR)"
echo "  4. Pancreas (baseline: 28.767 PSNR)"
echo ""
echo "⚠️  注意：Foot 已完成（28.696 PSNR ✅）"
echo "========================================"

# 配置参数（统一使用 v3 成功配置）
KPLANES_RESOLUTION=64
KPLANES_DIM=32
KPLANES_DECODER_HIDDEN=128
KPLANES_DECODER_LAYERS=3
KPLANES_LR_INIT=0.002
KPLANES_LR_FINAL=0.0002
LAMBDA_PLANE_TV=0.002
TV_LOSS_TYPE="l2"
ITERATIONS=30000
POSITION_LR_INIT=0.0002
DENSIFY_UNTIL_ITER=15000
DENSIFY_GRAD_THRESHOLD=0.00005

# 器官列表（Foot 已完成，跳过）
ORGANS=("chest" "head" "abdomen" "pancreas")
ORGAN_NAMES=("Chest" "Head" "Abdomen" "Pancreas")
BASELINE_PSNR=("26.506" "26.692" "29.290" "28.767")

# 训练函数
train_organ() {
    local organ=$1
    local organ_name=$2
    local baseline=$3
    local timestamp=$(date +"%Y_%m_%d_%H_%M")
    local output_dir="output/${timestamp}_${organ}_3views_x2_v3"

    echo ""
    echo "========================================"
    echo "开始训练: ${organ_name}"
    echo "========================================"
    echo "时间: $(date)"
    echo "输出: ${output_dir}"
    echo "Baseline PSNR: ${baseline} dB"
    echo "========================================"

    conda run -n r2_gaussian_new python train.py \
        --source_path data/369/${organ}_50_3views.pickle \
        --model_path ${output_dir} \
        --gaussiansN 1 \
        --enable_kplanes \
        --kplanes_resolution ${KPLANES_RESOLUTION} \
        --kplanes_dim ${KPLANES_DIM} \
        --kplanes_decoder_hidden ${KPLANES_DECODER_HIDDEN} \
        --kplanes_decoder_layers ${KPLANES_DECODER_LAYERS} \
        --kplanes_lr_init ${KPLANES_LR_INIT} \
        --kplanes_lr_final ${KPLANES_LR_FINAL} \
        --lambda_plane_tv ${LAMBDA_PLANE_TV} \
        --tv_loss_type ${TV_LOSS_TYPE} \
        --iterations ${ITERATIONS} \
        --position_lr_init ${POSITION_LR_INIT} \
        --densify_until_iter ${DENSIFY_UNTIL_ITER} \
        --densify_grad_threshold ${DENSIFY_GRAD_THRESHOLD} \
        --test_iterations 5000 10000 20000 30000 \
        --save_iterations 20000 30000 \
        --eval

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ ${organ_name} 训练完成！"
        echo "结果文件: ${output_dir}/eval/iter_030000/eval2d_render_test.yml"

        # 提取 PSNR 结果
        if [ -f "${output_dir}/eval/iter_020000/eval2d_render_test.yml" ]; then
            local psnr_20k=$(grep "psnr_2d:" "${output_dir}/eval/iter_020000/eval2d_render_test.yml" | awk '{print $2}')
            echo "20k 迭代 PSNR: ${psnr_20k} dB"
        fi

        if [ -f "${output_dir}/eval/iter_030000/eval2d_render_test.yml" ]; then
            local psnr_30k=$(grep "psnr_2d:" "${output_dir}/eval/iter_030000/eval2d_render_test.yml" | awk '{print $2}')
            echo "30k 迭代 PSNR: ${psnr_30k} dB"
        fi
    else
        echo ""
        echo "❌ ${organ_name} 训练失败（退出码: ${exit_code}）"
        echo "请检查日志: ${output_dir}/"
        return 1
    fi
}

# 串行训练所有器官
for i in "${!ORGANS[@]}"; do
    train_organ "${ORGANS[$i]}" "${ORGAN_NAMES[$i]}" "${BASELINE_PSNR[$i]}"

    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠️  警告: ${ORGAN_NAMES[$i]} 训练失败，但继续训练下一个器官"
    fi

    # 短暂休息（避免 GPU 热重启问题）
    if [ $i -lt $((${#ORGANS[@]} - 1)) ]; then
        echo ""
        echo "休息 10 秒后继续..."
        sleep 10
    fi
done

echo ""
echo "========================================"
echo "所有训练任务完成"
echo "========================================"
echo "结束时间: $(date)"
echo ""
echo "结果汇总："
echo "  Foot:     28.696 PSNR ✅ (已完成)"
echo "  Chest:    检查 output/*chest*x2_v3/eval/"
echo "  Head:     检查 output/*head*x2_v3/eval/"
echo "  Abdomen:  检查 output/*abdomen*x2_v3/eval/"
echo "  Pancreas: 检查 output/*pancreas*x2_v3/eval/"
echo ""
echo "快速查看所有结果："
echo "  grep -h \"psnr_2d:\" output/*_3views_x2_v3/eval/iter_020000/eval2d_render_test.yml"
echo "========================================"
