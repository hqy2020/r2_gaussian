#!/bin/bash
# R²-Gaussian 组合实验结果分析脚本
# 用法: ./scripts/analyze_combo_results.sh

echo "=== R²-Gaussian 组合实验结果 ==="
echo ""

# Baseline SOTA 值
declare -A BASELINE_PSNR
BASELINE_PSNR["foot"]=28.487
BASELINE_PSNR["chest"]=26.506
BASELINE_PSNR["head"]=26.692
BASELINE_PSNR["abdomen"]=29.290
BASELINE_PSNR["pancreas"]=28.767

# 遍历所有组合实验结果
for dir in output/*combo*; do
    if [ -d "$dir" ]; then
        # 解析目录名
        dirname=$(basename "$dir")

        # 提取器官和组合信息
        organ=$(echo "$dirname" | grep -oP '(?<=_)(foot|chest|head|abdomen|pancreas)(?=_)')
        combo=$(echo "$dirname" | grep -oP 'combo_[A-D]' | cut -d'_' -f2)
        views=$(echo "$dirname" | grep -oP '\d+(?=views)')

        echo "=== $dirname ==="

        # 读取评估结果
        eval_file="$dir/eval/iter_030000/eval2d_render_test.yml"
        eval3d_file="$dir/eval/iter_030000/eval3d.yml"

        if [ -f "$eval_file" ]; then
            psnr=$(grep "psnr_2d" "$eval_file" 2>/dev/null | head -1 | awk '{print $2}')
            ssim=$(grep "ssim_2d" "$eval_file" 2>/dev/null | head -1 | awk '{print $2}')

            if [ -n "$psnr" ] && [ -n "$organ" ]; then
                baseline=${BASELINE_PSNR[$organ]}
                diff=$(echo "$psnr - $baseline" | bc -l 2>/dev/null || echo "N/A")

                echo "  组合: $combo"
                echo "  器官: $organ"
                echo "  视角: ${views}v"
                echo "  PSNR: $psnr (baseline: $baseline, Δ: $diff)"
                echo "  SSIM: $ssim"
            fi
        else
            # 尝试读取 10k 结果
            eval_file_10k="$dir/eval/iter_010000/eval2d_render_test.yml"
            if [ -f "$eval_file_10k" ]; then
                psnr=$(grep "psnr_2d" "$eval_file_10k" 2>/dev/null | head -1 | awk '{print $2}')
                echo "  [10k] PSNR: $psnr"
            else
                echo "  评估结果不存在"
            fi
        fi
        echo ""
    fi
done

echo "=== 分析完成 ==="
