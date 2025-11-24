#!/bin/bash
# 监控四个初始化方法的训练进度

echo "=========================================="
echo "训练进度监控"
echo "=========================================="
echo "更新时间: $(date)"
echo ""

# 查找所有相关训练输出
PATTERN="output/*_foot_3views_init_*"

for dir in $PATTERN; do
    if [ -d "$dir" ]; then
        method=$(basename "$dir" | sed 's/.*init_//')
        echo "方法: ${method}"
        echo "目录: ${dir}"

        # 检查训练是否在运行
        log_file="${dir}_train.log"
        if [ -f "$log_file" ]; then
            # 最新迭代
            last_iter=$(grep -oP 'Iteration \K[0-9]+' "$log_file" | tail -1)
            if [ -n "$last_iter" ]; then
                echo "  当前迭代: ${last_iter} / 30000"
                progress=$(awk "BEGIN {printf \"%.1f\", $last_iter / 30000 * 100}")
                echo "  进度: ${progress}%"

                # 最新 loss
                last_loss=$(grep "Loss:" "$log_file" | tail -1 | grep -oP 'Loss: \K[0-9.]+')
                if [ -n "$last_loss" ]; then
                    echo "  当前 Loss: ${last_loss}"
                fi

                # 最新评估结果
                if [ -f "${dir}/eval/iter_030000/eval2d_render_test.yml" ]; then
                    psnr=$(grep "psnr:" "${dir}/eval/iter_030000/eval2d_render_test.yml" | head -1 | awk '{print $2}')
                    ssim=$(grep "ssim:" "${dir}/eval/iter_030000/eval2d_render_test.yml" | head -1 | awk '{print $2}')
                    echo "  ✅ 训练完成!"
                    echo "  最终 PSNR: ${psnr} dB"
                    echo "  最终 SSIM: ${ssim}"
                else
                    # 检查中间评估
                    latest_eval=$(find "${dir}/eval" -name "eval2d_render_test.yml" 2>/dev/null | sort | tail -1)
                    if [ -n "$latest_eval" ]; then
                        iter=$(echo "$latest_eval" | grep -oP 'iter_\K[0-9]+')
                        psnr=$(grep "psnr:" "$latest_eval" | head -1 | awk '{print $2}')
                        ssim=$(grep "ssim:" "$latest_eval" | head -1 | awk '{print $2}')
                        echo "  最近评估 (iter ${iter}):"
                        echo "    PSNR: ${psnr} dB"
                        echo "    SSIM: ${ssim}"
                    fi
                fi

                # GPU 使用情况
                echo "  日志: ${log_file}"
            else
                echo "  ⏳ 等待开始..."
            fi
        else
            echo "  ❌ 日志文件不存在"
        fi
        echo ""
    fi
done

echo "=========================================="
echo "快速命令:"
echo "  查看实时日志: tail -f output/*_foot_3views_init_baseline_train.log"
echo "  查看 GPU: watch -n 1 nvidia-smi"
echo "  停止所有训练: pkill -f 'train.py.*foot_3views'"
echo "=========================================="
