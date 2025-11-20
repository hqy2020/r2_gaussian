#!/bin/bash

# 监控5个器官的训练进度

echo "========================================"
echo "监控 5 个器官训练进度"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

organs=("chest" "foot" "head" "abdomen" "pancreas")

# 定义SOTA基准
declare -A SOTA_PSNR SOTA_SSIM
SOTA_PSNR[chest]=26.506;  SOTA_SSIM[chest]=0.8413
SOTA_PSNR[foot]=28.4873;  SOTA_SSIM[foot]=0.9005
SOTA_PSNR[head]=26.6915;  SOTA_SSIM[head]=0.9247
SOTA_PSNR[abdomen]=29.2896; SOTA_SSIM[abdomen]=0.9366
SOTA_PSNR[pancreas]=28.7669; SOTA_SSIM[pancreas]=0.9247

echo "训练进程状态:"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    pid_file="output/2025_11_20_16_16_${organ}_3views_bino.pid"

    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            runtime=$(ps -p "$pid" -o etime= | tr -d ' ')
            echo "✅ $organ (PID: $pid, 运行时间: $runtime)"
        else
            echo "❌ $organ (进程已结束)"
        fi
    else
        echo "⚠️  $organ (无PID文件)"
    fi
done
echo ""

echo "训练进度 (从日志):"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    log_file="output/2025_11_20_16_16_${organ}_3views_bino_train.log"

    if [ -f "$log_file" ]; then
        # 提取最新的迭代进度
        latest_iter=$(tail -5 "$log_file" | grep -oP 'Train:\s+\d+%\|\S+\|\s+\K\d+(?=/30000)' | tail -1)
        latest_percent=$(tail -5 "$log_file" | grep -oP 'Train:\s+\K\d+(?=%)' | tail -1)
        latest_loss=$(tail -5 "$log_file" | grep -oP 'loss=\K[0-9.e+-]+' | tail -1)
        latest_pts=$(tail -5 "$log_file" | grep -oP 'pts=\K[0-9.e+]+' | tail -1)

        if [ -n "$latest_iter" ]; then
            echo "$organ: 迭代 $latest_iter/30000 ($latest_percent%) | loss=$latest_loss | pts=$latest_pts"
        else
            echo "$organ: 无法读取进度"
        fi
    else
        echo "$organ: 日志文件不存在"
    fi
done
echo ""

echo "已保存的Checkpoint:"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    model_path="output/2025_11_20_16_16_${organ}_3views_bino"

    if [ -d "$model_path/point_cloud" ]; then
        checkpoints=$(ls "$model_path/point_cloud" 2>/dev/null | grep -oP 'iteration_\K\d+' | sort -n | tail -5 | tr '\n' ', ')
        echo "$organ: $checkpoints"
    else
        echo "$organ: 无checkpoint目录"
    fi
done
echo ""

echo "预估完成时间 (基于当前速度):"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    log_file="output/2025_11_20_16_16_${organ}_3views_bino_train.log"
    pid_file="output/2025_11_20_16_16_${organ}_3views_bino.pid"

    if [ -f "$log_file" ] && [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            # 获取运行时间(秒)
            runtime_sec=$(ps -p "$pid" -o etimes= | tr -d ' ')
            # 获取当前迭代数
            current_iter=$(tail -5 "$log_file" | grep -oP 'Train:\s+\d+%\|\S+\|\s+\K\d+(?=/30000)' | tail -1)

            if [ -n "$current_iter" ] && [ "$current_iter" -gt 0 ] && [ "$runtime_sec" -gt 0 ]; then
                # 计算速度(秒/迭代)
                speed=$(echo "scale=2; $runtime_sec / $current_iter" | bc)
                # 计算剩余时间
                remaining_iter=$((30000 - current_iter))
                remaining_sec=$(echo "$speed * $remaining_iter" | bc | cut -d. -f1)

                # 转换为小时:分钟
                remaining_hours=$((remaining_sec / 3600))
                remaining_mins=$(( (remaining_sec % 3600) / 60 ))

                echo "$organ: 约 ${remaining_hours}小时${remaining_mins}分钟 (速度: ${speed}秒/iter)"
            fi
        fi
    fi
done
echo ""

echo "========================================"
echo "使用 './test_all_30k.sh' 测试完成的模型"
echo "========================================"
