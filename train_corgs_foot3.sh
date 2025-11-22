#!/bin/bash
# CoR-GS Stage 3: Pseudo-view Co-regularization 完整训练脚本
# 数据集: Foot-3 views
# 目标: 超越 baseline (PSNR 28.4873, SSIM 0.9005)

echo "========================================"
echo "CoR-GS Stage 3 - Foot 3 views 训练"
echo "日期: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 激活环境
echo "激活 Conda 环境..."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 训练参数
TIMESTAMP=$(date '+%Y_%m_%d_%H_%M')
MODEL_PATH="output/${TIMESTAMP}_foot_3views_corgs_stage3"
ITERATIONS=30000
DATA_PATH="data/369/foot_50_3views.pickle"

# CoR-GS 参数（基于论文和已有测试）
GAUSSIANS_N=2                    # 双模型
ENABLE_CORGS=true                # 启用 CoR-GS
LAMBDA_PSEUDO=1.0                # Pseudo-view 权重（论文默认）
PSEUDO_START_ITER=2000           # 从 2000 iter 开始 pseudo-view co-reg
PSEUDO_NOISE_STD=0.02            # 位置噪声标准差（约 ±0.4mm）
DENSIFY_UNTIL_ITER=15000         # Densification 结束迭代

echo ""
echo "训练配置："
echo "  模型路径: $MODEL_PATH"
echo "  总迭代数: $ITERATIONS"
echo "  数据集: $DATA_PATH"
echo "  模型数量: $GAUSSIANS_N"
echo "  Pseudo-view 启动: $PSEUDO_START_ITER iter"
echo "  Pseudo-view 权重: $LAMBDA_PSEUDO"
echo "  位置噪声标准差: $PSEUDO_NOISE_STD"
echo "  Densify 结束: $DENSIFY_UNTIL_ITER iter"
echo ""

# 检查数据文件
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 错误：数据文件不存在: $DATA_PATH"
    echo "请确认数据路径正确"
    exit 1
fi

echo "✓ 数据文件验证通过"
echo ""

# 启动训练（后台运行）
echo "启动训练..."
nohup python3 train.py \
    --source_path "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --iterations $ITERATIONS \
    --gaussiansN $GAUSSIANS_N \
    --enable_corgs \
    --corgs_pseudo_weight $LAMBDA_PSEUDO \
    --corgs_pseudo_start_iter $PSEUDO_START_ITER \
    --corgs_pseudo_noise_std $PSEUDO_NOISE_STD \
    --densify_until_iter $DENSIFY_UNTIL_ITER \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 5000 10000 15000 20000 25000 30000 \
    --checkpoint_iterations 5000 10000 15000 20000 25000 30000 \
    > train_corgs_foot3.log 2>&1 &

# 获取进程 PID
TRAIN_PID=$!

echo ""
echo "========================================"
echo "训练已启动！"
echo "========================================"
echo "进程 PID: $TRAIN_PID"
echo "日志文件: train_corgs_foot3.log"
echo "输出目录: $MODEL_PATH"
echo ""
echo "监控命令："
echo "  tail -f train_corgs_foot3.log       # 实时查看日志"
echo "  watch -n 5 'tail -20 train_corgs_foot3.log'  # 每5秒刷新"
echo "  ps aux | grep $TRAIN_PID            # 检查进程状态"
echo "  kill $TRAIN_PID                      # 停止训练"
echo ""
echo "TensorBoard 监控："
echo "  tensorboard --logdir $MODEL_PATH --port 6006"
echo ""
echo "预计训练时间: 6-8 小时"
echo "预期性能: PSNR > 28.8 dB, SSIM > 0.908"
echo ""

# 保存 PID 到文件
echo $TRAIN_PID > train_corgs_foot3.pid
echo "PID 已保存到: train_corgs_foot3.pid"
echo ""
echo "训练已在后台运行，请使用上述命令监控进度。"
echo "========================================"
