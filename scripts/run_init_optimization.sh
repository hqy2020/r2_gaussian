#!/bin/bash
# R²-Gaussian 点云初始化系统性优化实验脚本
# 阶段一：快速筛选 6 个核心配置（10k iterations）

# 设置环境
# 注意：假设当前已在正确的 conda 环境中，或使用系统 python3
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 如果需要激活 conda 环境，取消下面的注释并运行
# eval "$(conda shell.bash hook)"
# conda activate r2_gaussian_new

# 实验配置
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_PATH="data/369/foot_50_3views.pickle"
OUTPUT_BASE="output/init_optim_${TIMESTAMP}"
ITERATIONS=10000
DEVICE=0

echo "=========================================="
echo "点云初始化优化实验 - 阶段一（10k快速筛选）"
echo "时间戳: ${TIMESTAMP}"
echo "数据集: ${DATA_PATH}"
echo "输出目录: ${OUTPUT_BASE}"
echo "训练迭代: ${ITERATIONS}"
echo "=========================================="

# 创建输出目录
mkdir -p ${OUTPUT_BASE}
mkdir -p init_files

# =========================================
# 实验 1: Baseline (50k, random, no denoise)
# =========================================
echo ""
echo "[1/6] 实验 1: Baseline"
echo "配置: n_points=50000, sampling=random, denoise=False"

EXP_NAME="exp1_baseline"
INIT_FILE="init_files/${TIMESTAMP}_${EXP_NAME}.npy"

# 使用现有的 baseline 初始化文件（如果存在）
if [ -f "data/369/init_foot_50_3views.npy" ]; then
    echo "使用现有 baseline 初始化文件"
    INIT_FILE="data/369/init_foot_50_3views.npy"
else
    python3 initialize_pcd.py \
        --data ${DATA_PATH} \
        --output ${INIT_FILE} \
        --n_points 50000 \
        --density_thresh 0.05 \
        --density_rescale 0.15 \
        --sampling_strategy random \
        --enable_denoise false
fi

python3 train.py \
    --source_path ${DATA_PATH} \
    --ply_path ${INIT_FILE} \
    --model_path ${OUTPUT_BASE}/${EXP_NAME} \
    --iterations ${ITERATIONS} \
    --device cuda:${DEVICE} \
    --eval \
    > ${OUTPUT_BASE}/${EXP_NAME}.log 2>&1 &

echo "进程启动: $(jobs -p | tail -1)"
sleep 5

# =========================================
# 实验 2: De-Init 降噪 (50k, random, denoise σ=3)
# =========================================
echo ""
echo "[2/6] 实验 2: De-Init 降噪"
echo "配置: n_points=50000, sampling=random, denoise=True (σ=3)"

EXP_NAME="exp2_denoise"
INIT_FILE="init_files/${TIMESTAMP}_${EXP_NAME}.npy"

python3 initialize_pcd.py \
    --data ${DATA_PATH} \
    --output ${INIT_FILE} \
    --n_points 50000 \
    --density_thresh 0.05 \
    --density_rescale 0.15 \
    --sampling_strategy random \
    --enable_denoise true \
    --denoise_sigma 3.0

python3 train.py \
    --source_path ${DATA_PATH} \
    --ply_path ${INIT_FILE} \
    --model_path ${OUTPUT_BASE}/${EXP_NAME} \
    --iterations ${ITERATIONS} \
    --device cuda:${DEVICE} \
    --eval \
    > ${OUTPUT_BASE}/${EXP_NAME}.log 2>&1 &

echo "进程启动: $(jobs -p | tail -1)"
sleep 5

# =========================================
# 实验 3: 密度加权采样 (50k, density_weighted, no denoise)
# =========================================
echo ""
echo "[3/6] 实验 3: 密度加权采样"
echo "配置: n_points=50000, sampling=density_weighted, denoise=False"

EXP_NAME="exp3_weighted"
INIT_FILE="init_files/${TIMESTAMP}_${EXP_NAME}.npy"

python3 initialize_pcd.py \
    --data ${DATA_PATH} \
    --output ${INIT_FILE} \
    --n_points 50000 \
    --density_thresh 0.05 \
    --density_rescale 0.15 \
    --sampling_strategy density_weighted \
    --enable_denoise false

python3 train.py \
    --source_path ${DATA_PATH} \
    --ply_path ${INIT_FILE} \
    --model_path ${OUTPUT_BASE}/${EXP_NAME} \
    --iterations ${ITERATIONS} \
    --device cuda:${DEVICE} \
    --eval \
    > ${OUTPUT_BASE}/${EXP_NAME}.log 2>&1 &

echo "进程启动: $(jobs -p | tail -1)"
sleep 5

# =========================================
# 实验 4: 增加点数 (75k, random, no denoise)
# =========================================
echo ""
echo "[4/6] 实验 4: 增加点数"
echo "配置: n_points=75000, sampling=random, denoise=False"

EXP_NAME="exp4_more_points"
INIT_FILE="init_files/${TIMESTAMP}_${EXP_NAME}.npy"

python3 initialize_pcd.py \
    --data ${DATA_PATH} \
    --output ${INIT_FILE} \
    --n_points 75000 \
    --density_thresh 0.05 \
    --density_rescale 0.15 \
    --sampling_strategy random \
    --enable_denoise false

python3 train.py \
    --source_path ${DATA_PATH} \
    --ply_path ${INIT_FILE} \
    --model_path ${OUTPUT_BASE}/${EXP_NAME} \
    --iterations ${ITERATIONS} \
    --device cuda:${DEVICE} \
    --eval \
    > ${OUTPUT_BASE}/${EXP_NAME}.log 2>&1 &

echo "进程启动: $(jobs -p | tail -1)"
sleep 5

# =========================================
# 实验 5: 组合优化 (60k, density_weighted, denoise σ=3, rescale=0.20)
# =========================================
echo ""
echo "[5/6] 实验 5: 组合优化"
echo "配置: n_points=60000, sampling=density_weighted, denoise=True, rescale=0.20"

EXP_NAME="exp5_combined"
INIT_FILE="init_files/${TIMESTAMP}_${EXP_NAME}.npy"

python3 initialize_pcd.py \
    --data ${DATA_PATH} \
    --output ${INIT_FILE} \
    --n_points 60000 \
    --density_thresh 0.05 \
    --density_rescale 0.20 \
    --sampling_strategy density_weighted \
    --enable_denoise true \
    --denoise_sigma 3.0

python3 train.py \
    --source_path ${DATA_PATH} \
    --ply_path ${INIT_FILE} \
    --model_path ${OUTPUT_BASE}/${EXP_NAME} \
    --iterations ${ITERATIONS} \
    --device cuda:${DEVICE} \
    --eval \
    > ${OUTPUT_BASE}/${EXP_NAME}.log 2>&1 &

echo "进程启动: $(jobs -p | tail -1)"
sleep 5

# =========================================
# 实验 6: 更严格过滤 (50k, random, denoise σ=3, thresh=0.08)
# =========================================
echo ""
echo "[6/6] 实验 6: 更严格过滤"
echo "配置: n_points=50000, sampling=random, denoise=True, thresh=0.08"

EXP_NAME="exp6_high_thresh"
INIT_FILE="init_files/${TIMESTAMP}_${EXP_NAME}.npy"

python3 initialize_pcd.py \
    --data ${DATA_PATH} \
    --output ${INIT_FILE} \
    --n_points 50000 \
    --density_thresh 0.08 \
    --density_rescale 0.15 \
    --sampling_strategy random \
    --enable_denoise true \
    --denoise_sigma 3.0

python3 train.py \
    --source_path ${DATA_PATH} \
    --ply_path ${INIT_FILE} \
    --model_path ${OUTPUT_BASE}/${EXP_NAME} \
    --iterations ${ITERATIONS} \
    --device cuda:${DEVICE} \
    --eval \
    > ${OUTPUT_BASE}/${EXP_NAME}.log 2>&1 &

echo "进程启动: $(jobs -p | tail -1)"

# =========================================
# 监控信息
# =========================================
echo ""
echo "=========================================="
echo "所有实验已启动！"
echo "=========================================="
echo ""
echo "监控命令："
echo "  查看进程: ps aux | grep train.py | grep ${TIMESTAMP}"
echo "  查看 GPU: watch -n 1 nvidia-smi"
echo "  查看日志: tail -f ${OUTPUT_BASE}/exp*.log"
echo ""
echo "预计完成时间: 2-3 小时"
echo ""
echo "结果分析命令（实验完成后运行）:"
echo "  python3 scripts/analyze_init_results.py --timestamp ${TIMESTAMP}"
echo ""
