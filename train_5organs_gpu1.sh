#!/bin/bash

# 在GPU 1上训练5个器官的BINO方法
# 保存中间checkpoint用于测试

export CUDA_VISIBLE_DEVICES=1

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 定义器官列表
organs=("chest" "foot" "head" "abdomen" "pancreas")

# 获取当前时间戳
timestamp=$(date '+%Y_%m_%d_%H_%M')

echo "========================================"
echo "开始在 GPU 1 上训练 5 个器官"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "方法: BINO"
echo "迭代数: 30000"
echo "========================================"
echo ""

# 为每个器官启动训练
for organ in "${organs[@]}"; do
    output_dir="output/${timestamp}_${organ}_3views_bino_gpu1"
    log_file="${output_dir}_train.log"
    pid_file="${output_dir}.pid"

    echo "----------------------------------------"
    echo "启动器官: $organ"
    echo "输出目录: $output_dir"
    echo "日志文件: $log_file"
    echo "----------------------------------------"

    # 启动训练 (后台运行)
    nohup python train.py \
        -s "data/369/${organ}_50_3views.pickle" \
        -m "$output_dir" \
        --eval \
        --iterations 30000 \
        --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
        --save_iterations 30000 \
        --checkpoint_iterations 1000 5000 10000 15000 20000 25000 \
        --densify_until_iter 15000 \
        > "$log_file" 2>&1 &

    # 保存进程ID
    echo $! > "$pid_file"

    echo "✅ $organ 训练已启动 (PID: $!)"
    echo ""

    # 等待2秒再启动下一个,避免GPU初始化冲突
    sleep 2
done

echo "========================================"
echo "所有训练已启动完成!"
echo "========================================"
echo ""
echo "监控命令:"
echo "  查看所有进程: ps aux | grep train.py"
echo "  查看GPU使用: nvidia-smi"
echo "  运行监控脚本: ./monitor_training_gpu1.sh"
echo ""
echo "测试命令 (训练完成后):"
echo "  ./test_all_30k_gpu1.sh"
echo ""
