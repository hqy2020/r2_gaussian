#!/bin/bash

###############################################################################
# 点云数量对比实验 - 并行训练脚本
#
# 同时启动 4 个训练（10k iterations 快速验证）：
# - 25k 点 (GPU 0)
# - 50k 点 (GPU 0) - baseline 验证
# - 75k 点 (GPU 1)
# - 100k 点 (GPU 1)
#
# 预计时间：2-3 小时
# 日期：2025-11-24
###############################################################################

source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

echo "======================================================================"
echo "🚀 点云数量对比实验 - 并行训练"
echo "======================================================================"
echo ""
echo "实验配置："
echo "  - 25k 点  (GPU 0)"
echo "  - 50k 点  (GPU 0) - baseline 验证"
echo "  - 75k 点  (GPU 1)"
echo "  - 100k 点 (GPU 1)"
echo ""
echo "训练参数："
echo "  - 迭代次数: 10000 (快速验证)"
echo "  - 数据集: Foot-3 views"
echo "  - Baseline: PSNR 28.48 dB, SSIM 0.9008"
echo ""
echo "预计时间: 2-3 小时"
echo "======================================================================"
echo ""

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")
DATASET="data/369/foot_50_3views.pickle"
ITERATIONS=10000
DENSIFY_UNTIL=5000

# 检查初始化文件是否存在
INIT_DIR="data/init_points"
if [ ! -d "$INIT_DIR" ]; then
    echo "❌ 错误: 初始化文件目录不存在"
    echo "请先运行: bash scripts/init_npoints_comparison.sh"
    exit 1
fi

# 训练函数
train_model() {
    local n_points=$1
    local gpu_id=$2
    local init_file="${INIT_DIR}/init_foot_50_3views_${n_points}.npy"
    local output_dir="output/${TIMESTAMP}_foot_3views_npoints_${n_points}_10k"
    local log_file="${output_dir}.log"

    echo "⚡ 启动训练: ${n_points} 点 (GPU ${gpu_id})"

    CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py \
        --source_path $DATASET \
        --model_path "$output_dir" \
        --ply_path "$init_file" \
        --eval \
        --gaussiansN 1 \
        --iterations $ITERATIONS \
        --position_lr_init 0.0002 \
        --density_lr_init 0.01 \
        --scaling_lr_init 0.005 \
        --rotation_lr_init 0.001 \
        --densify_until_iter $DENSIFY_UNTIL \
        2>&1 | tee "$log_file" &

    echo "  输出目录: $output_dir"
    echo "  日志文件: $log_file"
    echo "  进程 PID: $!"
    echo ""
}

# 启动并行训练
echo "开始启动训练..."
echo ""

train_model 25000 0
train_model 50000 0
train_model 75000 1
train_model 100000 1

echo "======================================================================"
echo "✅ 所有训练已启动！"
echo "======================================================================"
echo ""
echo "监控命令："
echo "  # 查看所有训练进程"
echo "  ps aux | grep train.py | grep npoints"
echo ""
echo "  # 查看 GPU 使用情况"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "  # 实时查看日志"
echo "  tail -f output/${TIMESTAMP}_foot_3views_npoints_25000_10k.log"
echo "  tail -f output/${TIMESTAMP}_foot_3views_npoints_50000_10k.log"
echo "  tail -f output/${TIMESTAMP}_foot_3views_npoints_75000_10k.log"
echo "  tail -f output/${TIMESTAMP}_foot_3views_npoints_100000_10k.log"
echo ""
echo "  # 查看中期结果（约 30 分钟后）"
echo "  bash scripts/check_npoints_results.sh $TIMESTAMP"
echo ""
echo "预计完成时间: $(date -d '+3 hours' '+%Y-%m-%d %H:%M')"
echo "======================================================================"
