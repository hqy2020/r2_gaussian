#!/bin/bash

# FSGS 训练监控脚本
# 每 30 分钟检查一次训练进度

FSGS_LOG="/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_fsgs_corrected/training.log"
MONITOR_LOG="/home/qyhu/Documents/r2_ours/r2_gaussian/fsgs_monitor.log"

echo "=========================================" >> "$MONITOR_LOG"
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$MONITOR_LOG"
echo "=========================================" >> "$MONITOR_LOG"

# 检查进程是否还在运行
if ps -p 4023750 > /dev/null 2>&1; then
    echo "✅ FSGS 训练仍在运行 (PID 4023750)" >> "$MONITOR_LOG"

    # 提取最新进度
    tail -1 "$FSGS_LOG" 2>/dev/null | grep -o "Train:.*" >> "$MONITOR_LOG" || echo "无法读取进度" >> "$MONITOR_LOG"

    # 检查是否已到达评估点
    for iter in 5000 10000 15000; do
        EVAL_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_fsgs_corrected/eval/iter_$(printf "%06d" $iter)"
        if [ -d "$EVAL_DIR" ] && [ ! -f "$EVAL_DIR/.reported" ]; then
            echo "" >> "$MONITOR_LOG"
            echo "🎯 发现新的评估结果: iter $iter" >> "$MONITOR_LOG"

            if [ -f "$EVAL_DIR/eval2d_render_test.yml" ]; then
                PSNR=$(grep "psnr_2d:" "$EVAL_DIR/eval2d_render_test.yml" | head -1)
                SSIM=$(grep "ssim_2d:" "$EVAL_DIR/eval2d_render_test.yml" | head -1)
                echo "   $PSNR" >> "$MONITOR_LOG"
                echo "   $SSIM" >> "$MONITOR_LOG"

                # 标记已报告
                touch "$EVAL_DIR/.reported"
            fi
        fi
    done
else
    echo "❌ FSGS 训练已结束" >> "$MONITOR_LOG"

    # 提取最终结果
    echo "" >> "$MONITOR_LOG"
    echo "📊 最终结果:" >> "$MONITOR_LOG"

    FINAL_EVAL="/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_fsgs_corrected/eval/iter_015000/eval2d_render_test.yml"
    if [ -f "$FINAL_EVAL" ]; then
        grep "psnr_2d:" "$FINAL_EVAL" | head -1 >> "$MONITOR_LOG"
        grep "ssim_2d:" "$FINAL_EVAL" | head -1 >> "$MONITOR_LOG"
    else
        echo "   未找到最终评估文件" >> "$MONITOR_LOG"
    fi
fi

echo "" >> "$MONITOR_LOG"
echo "" >> "$MONITOR_LOG"
