#!/bin/bash

# 测试5个器官30000轮的BINO训练结果
# 对比SOTA基准值

echo "========================================"
echo "测试 5 个器官 30000 轮 BINO 训练结果"
echo "========================================"
echo ""

# 定义器官列表和对应的SOTA基准值
declare -A SOTA_PSNR
declare -A SOTA_SSIM

SOTA_PSNR[chest]=26.506
SOTA_SSIM[chest]=0.8413

SOTA_PSNR[foot]=28.4873
SOTA_SSIM[foot]=0.9005

SOTA_PSNR[head]=26.6915
SOTA_SSIM[head]=0.9247

SOTA_PSNR[abdomen]=29.2896
SOTA_SSIM[abdomen]=0.9366

SOTA_PSNR[pancreas]=28.7669
SOTA_SSIM[pancreas]=0.9247

# 器官列表
organs=("chest" "foot" "head" "abdomen" "pancreas")

# 检查30000轮checkpoint是否存在
echo "检查训练checkpoint状态:"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    model_path="output/2025_11_20_16_16_${organ}_3views_bino"
    checkpoint_path="${model_path}/point_cloud/iteration_30000/point_cloud.ply"

    if [ -f "$checkpoint_path" ]; then
        echo "✅ $organ: 30000轮checkpoint存在"
    else
        echo "❌ $organ: 30000轮checkpoint不存在 (训练可能尚未完成)"
    fi
done
echo ""

# 测试已完成的模型
echo "开始测试已完成的模型..."
echo "========================================"
echo ""

for organ in "${organs[@]}"; do
    model_path="output/2025_11_20_16_16_${organ}_3views_bino"
    checkpoint_path="${model_path}/point_cloud/iteration_30000/point_cloud.ply"

    # 只测试已经完成训练的模型
    if [ -f "$checkpoint_path" ]; then
        echo "----------------------------------------"
        echo "测试器官: $organ"
        echo "模型路径: $model_path"
        echo "SOTA基准: PSNR=${SOTA_PSNR[$organ]}, SSIM=${SOTA_SSIM[$organ]}"
        echo "----------------------------------------"

        # 运行测试
        python test.py \
            -m "$model_path" \
            -s "data/369/${organ}_50_3views.pickle" \
            --iteration 30000 \
            --eval

        # 检查测试是否成功
        if [ $? -eq 0 ]; then
            echo "✅ $organ 测试完成"

            # 查找并显示结果
            result_file="${model_path}/results_30000.json"
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
done

echo "========================================"
echo "所有测试完成!"
echo "========================================"
echo ""
echo "结果总结:"
echo "----------------------------------------"
for organ in "${organs[@]}"; do
    model_path="output/2025_11_20_16_16_${organ}_3views_bino"
    result_file="${model_path}/results_30000.json"

    if [ -f "$result_file" ]; then
        echo ""
        echo "器官: $organ (SOTA: PSNR=${SOTA_PSNR[$organ]}, SSIM=${SOTA_SSIM[$organ]})"
        cat "$result_file" | grep -E "PSNR|SSIM" || echo "  结果文件格式异常"
    fi
done
echo ""
echo "========================================"
