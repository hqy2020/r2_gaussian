#!/bin/bash

# 批量运行2视角实验脚本
# 用途：为 chest, abdomen, head, pancreas 四个器官运行与 foot 相同的2视角实验

# 设置工作目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 设置GPU
export CUDA_VISIBLE_DEVICES=1

# 获取当前日期（格式：MMDD）
DATE=$(date +%m%d)

# 需要运行的器官列表（排除已完成的 foot）
ORGANS=("chest" "abdomen" "head" "pancreas")

# 数据目录和输出目录
DATA_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/data/24"
OUTPUT_BASE="/home/qyhu/Documents/r2_ours/r2_gaussian/output/24"

# 遍历每个器官
for organ in "${ORGANS[@]}"; do
    echo "=========================================="
    echo "开始运行: ${organ} 2视角实验"
    echo "=========================================="
    
    # 输入文件路径
    INPUT_FILE="${DATA_DIR}/${organ}_50_2views.pickle"
    
    # 输出目录路径
    OUTPUT_DIR="${OUTPUT_BASE}/${organ}ddg500_2_${DATE}"
    
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ 错误: 输入文件不存在: $INPUT_FILE"
        echo "跳过 ${organ}"
        continue
    fi
    
    # 检查输出目录是否已存在
    if [ -d "$OUTPUT_DIR" ]; then
        echo "⚠️  警告: 输出目录已存在: $OUTPUT_DIR"
        echo "是否继续？(y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            echo "跳过 ${organ}"
            continue
        fi
    fi
    
    # 运行训练命令
    echo "输入文件: $INPUT_FILE"
    echo "输出目录: $OUTPUT_DIR"
    echo "开始时间: $(date)"
    
    python train.py \
        -s "$INPUT_FILE" \
        -m "$OUTPUT_DIR" \
        --gaussiansN 1 \
        --enable_depth \
        --depth_loss_weight 0.05 \
        --depth_loss_type pearson \
        --pseudo_labels \
        --pseudo_label_weight 0.02 \
        --multi_gaussian_weight 0 \
        --num_additional_views 50
    
    # 检查运行结果
    if [ $? -eq 0 ]; then
        echo "✅ ${organ} 2视角实验完成"
        echo "完成时间: $(date)"
    else
        echo "❌ ${organ} 2视角实验失败"
        echo "失败时间: $(date)"
    fi
    
    echo ""
    echo "等待5秒后继续下一个..."
    sleep 5
done

echo "=========================================="
echo "所有实验运行完成！"
echo "=========================================="