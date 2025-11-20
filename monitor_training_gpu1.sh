#!/bin/bash

# 监控GPU 1上5个器官的训练进度

echo "========================================"
echo "监控 GPU 1 上 5 个器官训练进度"
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

# 查找最新批次
latest_batch=$(ls -d output/*_chest_3views_bino_gpu1 2>/dev/null | sort | tail -1 | grep -oP '\d{4}_\d{2}_\d{2}_\d{2}_\d{2}')

if [ -z "$latest_batch" ]; then
    echo "❌ 未找到GPU 1的训练"
    exit 1
fi

echo "训练批次: $latest_batch"
echo ""

echo "GPU 状态:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | grep "^1,"
echo ""

echo "训练进程状态:"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    pid_file="output/${latest_batch}_${organ}_3views_bino_gpu1.pid"

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
    log_file="output/${latest_batch}_${organ}_3views_bino_gpu1_train.log"

    if [ -f "$log_file" ]; then
        # 提取最新的迭代进度
        latest_line=$(tail -10 "$log_file" | grep "Train:" | tail -1)
        if [ -n "$latest_line" ]; then
            # 提取迭代数和百分比
            iter=$(echo "$latest_line" | grep -oP '\|\s+\K\d+(?=/30000)' || echo "N/A")
            percent=$(echo "$latest_line" | grep -oP 'Train:\s+\K\d+(?=%)' || echo "N/A")
            loss=$(echo "$latest_line" | grep -oP 'loss=\K[0-9.e+-]+' || echo "N/A")

            echo "$organ: $iter/30000 ($percent%) | loss=$loss"
        else
            echo "$organ: 等待训练开始..."
        fi
    else
        echo "$organ: 日志文件不存在"
    fi
done
echo ""

echo "已保存的Checkpoint:"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    model_path="output/${latest_batch}_${organ}_3views_bino_gpu1"

    if [ -d "$model_path/point_cloud" ]; then
        checkpoints=$(ls "$model_path/point_cloud" 2>/dev/null | grep -oP 'iteration_\K\d+' | sort -n | tail -5 | tr '\n' ', ' | sed 's/,$//')
        if [ -n "$checkpoints" ]; then
            echo "$organ: $checkpoints"
        else
            echo "$organ: 无checkpoint"
        fi
    else
        echo "$organ: 无point_cloud目录"
    fi
done
echo ""

echo "========================================"
echo "使用 './test_all_30k_gpu1.sh' 测试完成的模型"
echo "========================================"
