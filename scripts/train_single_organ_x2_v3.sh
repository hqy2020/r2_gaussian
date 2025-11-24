#!/bin/bash

# X²-Gaussian v3 单器官训练脚本模板
# 使用方法：
#   bash scripts/train_single_organ_x2_v3.sh <organ_name>
#
# 示例：
#   bash scripts/train_single_organ_x2_v3.sh chest
#   bash scripts/train_single_organ_x2_v3.sh head
#   bash scripts/train_single_organ_x2_v3.sh abdomen
#   bash scripts/train_single_organ_x2_v3.sh pancreas

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <organ_name>"
    echo ""
    echo "可用器官："
    echo "  chest    - 胸部 (baseline: 26.506 PSNR)"
    echo "  head     - 头部 (baseline: 26.692 PSNR)"
    echo "  abdomen  - 腹部 (baseline: 29.290 PSNR)"
    echo "  pancreas - 胰腺 (baseline: 28.767 PSNR)"
    echo "  foot     - 脚部 (baseline: 28.487 PSNR, ✅ 已达成 28.696)"
    exit 1
fi

ORGAN=$1
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
OUTPUT_DIR="output/${TIMESTAMP}_${ORGAN}_3views_x2_v3"

# 验证器官名称
case $ORGAN in
    chest|head|abdomen|pancreas|foot)
        ;;
    *)
        echo "❌ 错误：未知器官 '${ORGAN}'"
        echo "请使用: chest, head, abdomen, pancreas, foot"
        exit 1
        ;;
esac

echo "========================================"
echo "X²-Gaussian v3 训练"
echo "========================================"
echo "器官: ${ORGAN}"
echo "启动时间: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "核心配置："
echo "  - 调制方式: sigmoid [0.7, 1.3]"
echo "  - TV 正则化: λ = 0.002"
echo "  - K-Planes: 64×64 分辨率, 32 维度"
echo "  - 迭代次数: 30000 (最佳点通常在 20k)"
echo "========================================"

conda run -n r2_gaussian_new python train.py \
    --source_path data/369/${ORGAN}_50_3views.pickle \
    --model_path ${OUTPUT_DIR} \
    --gaussiansN 1 \
    --enable_kplanes \
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --kplanes_decoder_hidden 128 \
    --kplanes_decoder_layers 3 \
    --kplanes_lr_init 0.002 \
    --kplanes_lr_final 0.0002 \
    --lambda_plane_tv 0.002 \
    --tv_loss_type l2 \
    --iterations 30000 \
    --position_lr_init 0.0002 \
    --densify_until_iter 15000 \
    --densify_grad_threshold 0.00005 \
    --test_iterations 5000 10000 20000 30000 \
    --save_iterations 20000 30000 \
    --eval

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成"
    echo ""
    echo "查看结果："
    echo "  20k: cat ${OUTPUT_DIR}/eval/iter_020000/eval2d_render_test.yml | grep psnr_2d"
    echo "  30k: cat ${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml | grep psnr_2d"
    echo ""

    # 自动提取并显示结果
    if [ -f "${OUTPUT_DIR}/eval/iter_020000/eval2d_render_test.yml" ]; then
        echo "20k 迭代结果："
        grep "psnr_2d:\|ssim_2d:" "${OUTPUT_DIR}/eval/iter_020000/eval2d_render_test.yml"
        echo ""
    fi

    if [ -f "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml" ]; then
        echo "30k 迭代结果："
        grep "psnr_2d:\|ssim_2d:" "${OUTPUT_DIR}/eval/iter_030000/eval2d_render_test.yml"
    fi
else
    echo "❌ 训练失败（退出码: ${EXIT_CODE}）"
    echo "检查日志: ${OUTPUT_DIR}/"
fi
echo "========================================"
echo "完成时间: $(date)"
echo "========================================"
