#!/bin/bash

# FSGS 30k 训练监控脚本
# 每 30 分钟检查一次训练进度

TRAINING_LOG="/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_fsgs_30k/training.log"
MONITOR_LOG="/home/qyhu/Documents/r2_ours/r2_gaussian/fsgs_30k_monitor.log"
OUTPUT_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_fsgs_30k"

echo "=========================================" >> "$MONITOR_LOG"
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$MONITOR_LOG"
echo "=========================================" >> "$MONITOR_LOG"

# 检查进程是否还在运行
if pgrep -f "train.py.*fsgs_30k" > /dev/null; then
    echo "✅ FSGS 30k 训练仍在运行" >> "$MONITOR_LOG"

    # 提取最新进度 (最后一行包含 Train: 的)
    if [ -f "$TRAINING_LOG" ]; then
        LATEST_PROGRESS=$(tail -5 "$TRAINING_LOG" | grep "Train:" | tail -1)
        if [ -n "$LATEST_PROGRESS" ]; then
            echo "   $LATEST_PROGRESS" >> "$MONITOR_LOG"
        else
            echo "   无法读取训练进度" >> "$MONITOR_LOG"
        fi

        # 提取最新的 loss 值
        LATEST_LOSS=$(tail -100 "$TRAINING_LOG" | grep -o "Loss: [0-9.]*" | tail -1)
        if [ -n "$LATEST_LOSS" ]; then
            echo "   $LATEST_LOSS" >> "$MONITOR_LOG"
        fi
    fi

    # 检查评估点是否已生成
    echo "" >> "$MONITOR_LOG"
    echo "📊 已完成的评估点:" >> "$MONITOR_LOG"
    for iter in 5000 10000 15000 20000 25000 30000; do
        ITER_DIR=$(printf "%06d" $iter)
        EVAL_FILE="$OUTPUT_DIR/eval/iter_${ITER_DIR}/eval2d_render_test.yml"

        if [ -f "$EVAL_FILE" ] && [ ! -f "$OUTPUT_DIR/eval/iter_${ITER_DIR}/.reported_30k" ]; then
            echo "   🎯 iter $iter (新)" >> "$MONITOR_LOG"
            PSNR=$(grep "^psnr_2d:" "$EVAL_FILE" | head -1 | awk '{print $2}')
            SSIM=$(grep "^ssim_2d:" "$EVAL_FILE" | head -1 | awk '{print $2}')
            echo "      PSNR: $PSNR dB" >> "$MONITOR_LOG"
            echo "      SSIM: $SSIM" >> "$MONITOR_LOG"

            # 标记已报告
            touch "$OUTPUT_DIR/eval/iter_${ITER_DIR}/.reported_30k"

            # 如果是 30000，额外高亮显示
            if [ "$iter" == "30000" ]; then
                echo "" >> "$MONITOR_LOG"
                echo "🏁 最终结果:" >> "$MONITOR_LOG"
                echo "   PSNR: $PSNR dB (baseline: 28.547 dB)" >> "$MONITOR_LOG"
                echo "   SSIM: $SSIM (baseline: 0.9008)" >> "$MONITOR_LOG"

                # 计算差异
                DIFF=$(echo "$PSNR - 28.547" | bc)
                if (( $(echo "$PSNR >= 28.547" | bc -l) )); then
                    echo "   ✅ 超越 baseline (+$DIFF dB)" >> "$MONITOR_LOG"
                else
                    echo "   ❌ 低于 baseline ($DIFF dB)" >> "$MONITOR_LOG"
                fi
            fi
        elif [ -f "$EVAL_FILE" ]; then
            echo "   ✓ iter $iter (已记录)" >> "$MONITOR_LOG"
        fi
    done
else
    echo "❌ FSGS 30k 训练已结束或未运行" >> "$MONITOR_LOG"

    # 提取最终结果
    echo "" >> "$MONITOR_LOG"
    echo "📊 最终评估结果:" >> "$MONITOR_LOG"

    FINAL_EVAL="$OUTPUT_DIR/eval/iter_030000/eval2d_render_test.yml"
    if [ -f "$FINAL_EVAL" ]; then
        PSNR=$(grep "^psnr_2d:" "$FINAL_EVAL" | head -1 | awk '{print $2}')
        SSIM=$(grep "^ssim_2d:" "$FINAL_EVAL" | head -1 | awk '{print $2}')

        echo "   iter 30000:" >> "$MONITOR_LOG"
        echo "   PSNR: $PSNR dB (baseline: 28.547 dB)" >> "$MONITOR_LOG"
        echo "   SSIM: $SSIM (baseline: 0.9008)" >> "$MONITOR_LOG"

        # 判断是否成功
        DIFF=$(echo "$PSNR - 28.547" | bc)
        if (( $(echo "$PSNR >= 28.547" | bc -l) )); then
            echo "   ✅ 成功超越 baseline (+$DIFF dB)" >> "$MONITOR_LOG"
        else
            echo "   ❌ 未能超越 baseline ($DIFF dB)" >> "$MONITOR_LOG"
        fi
    else
        echo "   未找到最终评估文件" >> "$MONITOR_LOG"
    fi
fi

echo "" >> "$MONITOR_LOG"
echo "" >> "$MONITOR_LOG"
