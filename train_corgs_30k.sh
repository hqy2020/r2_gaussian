#!/bin/bash
# CoR-GS 完整训练脚本 - 30k iterations
# 修复所有关键 Bug 后的版本

echo "========================================"
echo "CoR-GS 完整训练 (30k iterations)"
echo "修复版本: Bug 1/2/3/4 已修复"
echo "========================================"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 训练参数
MODEL_PATH="output/2025_11_18_foot_3views_corgs_fixed_v3_mem_opt"
ITERATIONS=30000
DATA_PATH="/home/qyhu/Documents/r2_ours/r2_gaussian/data/foot_3views"

# CoR-GS 参数（基于官方配置）
LAMBDA_PSEUDO=1.0          # Pseudo-view 权重（官方未明确，默认 1.0）
PSEUDO_START_ITER=2000     # 官方: 2000
DENSIFY_UNTIL_ITER=15000   # 官方: 15000

echo "训练参数："
echo "  - 模型路径: $MODEL_PATH"
echo "  - 总迭代次数: $ITERATIONS"
echo "  - Pseudo-view 启动: $PSEUDO_START_ITER"
echo "  - Densification 结束: $DENSIFY_UNTIL_ITER"
echo "  - Lambda pseudo: $LAMBDA_PSEUDO"
echo ""

# 检查数据路径
if [ ! -d "$DATA_PATH" ]; then
    echo "❌ 错误：数据路径不存在: $DATA_PATH"
    echo "请修改脚本中的 DATA_PATH 变量"
    exit 1
fi

# 运行训练（后台运行，输出重定向到日志）
nohup python train.py \
    --source_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --iterations $ITERATIONS \
    --densify_until_iter $DENSIFY_UNTIL_ITER \
    --test_iterations 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --enable_pseudo_coreg \
    --lambda_pseudo $LAMBDA_PSEUDO \
    --pseudo_start_iter $PSEUDO_START_ITER \
    --gaussiansN 2 \
    --coreg \
    > train_corgs_30k.log 2>&1 &

# 获取进程 PID
TRAIN_PID=$!

echo ""
echo "========================================"
echo "训练已启动！"
echo "========================================"
echo "进程 PID: $TRAIN_PID"
echo "日志文件: train_corgs_30k.log"
echo "输出目录: $MODEL_PATH"
echo ""
echo "监控命令："
echo "  tail -f train_corgs_30k.log         # 实时查看日志"
echo "  ps aux | grep $TRAIN_PID           # 检查进程状态"
echo "  kill $TRAIN_PID                     # 停止训练"
echo ""
echo "预计训练时间: 6-8 小时（取决于硬件）"
echo ""

# 保存 PID 到文件
echo $TRAIN_PID > train_corgs_30k.pid
echo "PID 已保存到: train_corgs_30k.pid"
