#!/bin/bash

# 测试5个器官当前最新checkpoint的结果

echo "========================================"
echo "测试 5 个器官当前训练进度"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# 定义器官列表和SOTA基准值
declare -A SOTA_PSNR SOTA_SSIM
SOTA_PSNR[chest]=26.506;  SOTA_SSIM[chest]=0.8413
SOTA_PSNR[foot]=28.4873;  SOTA_SSIM[foot]=0.9005
SOTA_PSNR[head]=26.6915;  SOTA_SSIM[head]=0.9247
SOTA_PSNR[abdomen]=29.2896; SOTA_SSIM[abdomen]=0.9366
SOTA_PSNR[pancreas]=28.7669; SOTA_SSIM[pancreas]=0.9247

organs=("chest" "foot" "head" "abdomen" "pancreas")

echo "查找各器官最新的checkpoint..."
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    model_path="output/2025_11_20_16_16_${organ}_3views_bino"

    if [ -d "$model_path/eval" ]; then
        # 查找最新的iteration
        latest_iter=$(ls "$model_path/eval" 2>/dev/null | grep -oP 'iter_\K\d+' | sort -n | tail -1)

        if [ -n "$latest_iter" ]; then
            echo "✅ $organ: 最新checkpoint iter_${latest_iter}"
        else
            echo "❌ $organ: 无可用checkpoint"
        fi
    else
        echo "❌ $organ: eval目录不存在"
    fi
done
echo ""

# 测试所有器官的最新checkpoint
echo "开始测试..."
echo "========================================"
echo ""

for organ in "${organs[@]}"; do
    model_path="output/2025_11_20_16_16_${organ}_3views_bino"

    if [ -d "$model_path/eval" ]; then
        # 获取最新的iteration
        latest_iter=$(ls "$model_path/eval" 2>/dev/null | grep -oP 'iter_\K\d+' | sort -n | tail -1)

        if [ -n "$latest_iter" ]; then
            echo "----------------------------------------"
            echo "测试器官: $organ"
            echo "Checkpoint: iteration $latest_iter"
            echo "SOTA基准: PSNR=${SOTA_PSNR[$organ]}, SSIM=${SOTA_SSIM[$organ]}"
            echo "----------------------------------------"

            # 运行测试
            python test.py \
                -m "$model_path" \
                -s "data/369/${organ}_50_3views.pickle" \
                --iteration $latest_iter \
                --eval

            if [ $? -eq 0 ]; then
                echo "✅ $organ (iter_$latest_iter) 测试完成"

                # 查找结果文件
                result_file="${model_path}/results_${latest_iter}.json"
                if [ -f "$result_file" ]; then
                    echo "📊 测试结果:"
                    cat "$result_file"
                    echo ""
                fi
            else
                echo "❌ $organ 测试失败"
            fi
            echo ""
        fi
    fi
done

echo "========================================"
echo "测试完成!"
echo "========================================"
echo ""
echo "结果总结 (与SOTA对比):"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    model_path="output/2025_11_20_16_16_${organ}_3views_bino"

    if [ -d "$model_path/eval" ]; then
        latest_iter=$(ls "$model_path/eval" 2>/dev/null | grep -oP 'iter_\K\d+' | sort -n | tail -1)
        result_file="${model_path}/results_${latest_iter}.json"

        if [ -f "$result_file" ]; then
            echo ""
            echo "器官: $organ (iter_${latest_iter})"
            echo "  SOTA基准: PSNR=${SOTA_PSNR[$organ]}, SSIM=${SOTA_SSIM[$organ]}"
            echo "  当前结果:"
            cat "$result_file" | python3 -m json.tool 2>/dev/null || cat "$result_file"
        fi
    fi
done
echo ""
echo "========================================"
