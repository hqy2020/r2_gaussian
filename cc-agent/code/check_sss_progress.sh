#!/bin/bash
# SSS 训练进度快速检查脚本

OUTPUT_DIR="output/2025_11_17_foot_3views_sss"
BASELINE_DIR="output/foot_3_1013"

echo "=========================================="
echo "🔍 SSS Training Progress Check"
echo "=========================================="
echo ""

# 检查进程状态
echo "📊 训练进程状态:"
ps aux | grep -E "train.py.*2025_11_17_foot_3views_sss" | grep -v grep | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Memory:", $6/1024/1024"GB", "| Time:", $10}'
echo ""

# 检查已完成的评估
echo "✅ 已完成的评估 iterations:"
find $OUTPUT_DIR/eval -name "iter_*" -type d | sed 's/.*iter_/  - iter_/' | sort
echo ""

# 显示最新评估结果
LATEST_EVAL=$(find $OUTPUT_DIR/eval -name "iter_*" -type d | sort | tail -1)
if [ -n "$LATEST_EVAL" ]; then
    ITER_NUM=$(basename $LATEST_EVAL)
    echo "📈 最新评估结果 ($ITER_NUM):"
    if [ -f "$LATEST_EVAL/eval2d_render_test.yml" ]; then
        PSNR=$(grep "^psnr_2d:" "$LATEST_EVAL/eval2d_render_test.yml" | awk '{print $2}')
        SSIM=$(grep "^ssim_2d:" "$LATEST_EVAL/eval2d_render_test.yml" | awk '{print $2}')
        echo "  PSNR: $PSNR dB"
        echo "  SSIM: $SSIM"
    fi
fi
echo ""

# 对比 baseline
echo "🎯 与 Baseline 对比:"
if [ -f "$BASELINE_DIR/eval/iter_010000/eval2d_render_test.yml" ]; then
    BASELINE_PSNR=$(grep "^psnr_2d:" "$BASELINE_DIR/eval/iter_010000/eval2d_render_test.yml" | awk '{print $2}')
    BASELINE_SSIM=$(grep "^ssim_2d:" "$BASELINE_DIR/eval/iter_010000/eval2d_render_test.yml" | awk '{print $2}')
    echo "  Baseline (iter 10000):"
    echo "    PSNR: $BASELINE_PSNR dB"
    echo "    SSIM: $BASELINE_SSIM"

    if [ -n "$PSNR" ]; then
        DIFF=$(awk "BEGIN {print $PSNR - $BASELINE_PSNR}")
        echo ""
        echo "  当前差距: $DIFF dB"
        if (( $(echo "$DIFF > 0" | bc -l) )); then
            echo "  状态: ✅ 已超越 baseline!"
        else
            echo "  状态: ⏳ 继续训练中..."
        fi
    fi
fi
echo ""

# 估算完成时间
echo "⏱️  训练进度:"
TOTAL_ITERS=10000
if [ -n "$ITER_NUM" ]; then
    CURRENT_ITER=$(echo $ITER_NUM | sed 's/iter_0*//')
    PROGRESS=$(awk "BEGIN {print ($CURRENT_ITER / $TOTAL_ITERS) * 100}")
    echo "  当前进度: $CURRENT_ITER / $TOTAL_ITERS ($PROGRESS%)"

    REMAINING=$(awk "BEGIN {print $TOTAL_ITERS - $CURRENT_ITER}")
    echo "  剩余 iterations: $REMAINING"
fi
echo ""
echo "=========================================="
