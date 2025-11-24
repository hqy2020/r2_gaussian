#!/bin/bash

###############################################################################
# 点云数量对比实验 - 结果检查脚本
# 用法: bash scripts/check_npoints_results.sh <TIMESTAMP>
###############################################################################

if [ -z "$1" ]; then
    echo "用法: bash scripts/check_npoints_results.sh <TIMESTAMP>"
    echo "例如: bash scripts/check_npoints_results.sh 2025_11_24_12_30"
    exit 1
fi

TIMESTAMP=$1

echo "======================================================================"
echo "📊 点云数量对比实验 - 结果汇总"
echo "======================================================================"
echo ""

# 检查训练进程
echo "🔍 训练进程状态："
ps aux | grep train.py | grep npoints | grep -v grep | awk '{print "  PID", $2, "-", $NF}' || echo "  ❌ 没有运行中的训练进程"
echo ""

# 检查 GPU 使用情况
echo "🎮 GPU 使用情况："
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s (%s): %s%% GPU, %s/%s MB\n", $1, $2, $3, $4, $5}'
echo ""

# 结果对比表格
echo "📈 10k 迭代结果对比："
echo "======================================================================"
printf "%-15s %-15s %-15s %-15s\n" "点云数量" "PSNR (dB)" "SSIM" "状态"
echo "----------------------------------------------------------------------"

for n_points in 25000 50000 75000 100000; do
    output_dir="output/${TIMESTAMP}_foot_3views_npoints_${n_points}_10k"
    eval_file="$output_dir/eval/iter_010000/eval2d_render_test.yml"

    if [ -f "$eval_file" ]; then
        psnr=$(grep "^psnr_2d:" "$eval_file" | awk '{print $2}')
        ssim=$(grep "^ssim_2d:" "$eval_file" | awk '{print $2}')
        status="✅ 完成"
    else
        psnr="-"
        ssim="-"
        if [ -d "$output_dir" ]; then
            # 检查最新的 checkpoint
            latest_iter=$(ls -d $output_dir/point_cloud/iteration_* 2>/dev/null | sort -V | tail -1 | grep -oP '\d+$')
            if [ -n "$latest_iter" ]; then
                status="⏳ 训练中 (iter $latest_iter)"
            else
                status="🚧 初始化中"
            fi
        else
            status="❌ 未启动"
        fi
    fi

    printf "%-15s %-15s %-15s %-15s\n" "${n_points}" "$psnr" "$ssim" "$status"
done

echo "======================================================================"
echo ""

# Baseline 对比
echo "📊 与 Baseline 对比："
echo "  Baseline (50k): PSNR 28.48 dB, SSIM 0.9008"
echo ""

# 日志文件位置
echo "📁 日志文件："
for n_points in 25000 50000 75000 100000; do
    log_file="output/${TIMESTAMP}_foot_3views_npoints_${n_points}_10k.log"
    if [ -f "$log_file" ]; then
        lines=$(wc -l < "$log_file")
        size=$(du -h "$log_file" | cut -f1)
        echo "  ${n_points} 点: $log_file ($lines 行, $size)"
    fi
done

echo ""
echo "======================================================================"
echo "💡 提示："
echo "  - 实时监控: tail -f output/${TIMESTAMP}_foot_3views_npoints_50000_10k.log"
echo "  - GPU 监控: watch -n 2 nvidia-smi"
echo "  - 完整结果: 等待所有训练完成后查看 eval/iter_010000/ 目录"
echo "======================================================================"
