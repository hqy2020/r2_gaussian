#!/bin/bash

# DropGaussian 官方对齐实验
# 严格遵循官方实现: https://github.com/DCVL-3D/DropGaussian_release
# 修改内容:
#   1. drop_gamma: 0.1 -> 0.2 (官方默认)
#   2. 调度策略: drop_rate = 0.2 * (iter / 10000)，无延迟启动
#   3. 删除 importance-aware drop（官方无此功能）

source ~/anaconda3/bin/activate r2_gaussian_new
cd /home/qyhu/Documents/r2_ours/r2_gaussian

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M")

echo "=================================================="
echo "DropGaussian 官方对齐实验"
echo "时间戳: ${TIMESTAMP}"
echo ""
echo "对齐官方实现:"
echo "  - drop_gamma: 0.2 (官方默认)"
echo "  - drop_full_iter: 10000 (官方默认)"
echo "  - 调度公式: drop_rate = 0.2 * min(iter/10000, 1.0)"
echo "=================================================="
echo ""

# 3 views 实验
echo "[1/3] 启动 Foot 3views DropGaussian 官方对齐实验..."
python3 train.py \
    --source_path "data/369/foot_50_3views.pickle" \
    --model_path "output/${TIMESTAMP}_foot_3views_dropgaussian_official" \
    --iterations 30000 \
    --eval \
    --use_drop_gaussian \
    --drop_gamma 0.2 \
    --drop_full_iter 10000 \
    > output/${TIMESTAMP}_foot_3views_dropgaussian_official.log 2>&1 &
PID_3V=$!
echo "  PID: $PID_3V"

# 6 views 实验
echo "[2/3] 启动 Foot 6views DropGaussian 官方对齐实验..."
python3 train.py \
    --source_path "data/369/foot_50_6views.pickle" \
    --model_path "output/${TIMESTAMP}_foot_6views_dropgaussian_official" \
    --iterations 30000 \
    --eval \
    --use_drop_gaussian \
    --drop_gamma 0.2 \
    --drop_full_iter 10000 \
    > output/${TIMESTAMP}_foot_6views_dropgaussian_official.log 2>&1 &
PID_6V=$!
echo "  PID: $PID_6V"

# 9 views 实验
echo "[3/3] 启动 Foot 9views DropGaussian 官方对齐实验..."
python3 train.py \
    --source_path "data/369/foot_50_9views.pickle" \
    --model_path "output/${TIMESTAMP}_foot_9views_dropgaussian_official" \
    --iterations 30000 \
    --eval \
    --use_drop_gaussian \
    --drop_gamma 0.2 \
    --drop_full_iter 10000 \
    > output/${TIMESTAMP}_foot_9views_dropgaussian_official.log 2>&1 &
PID_9V=$!
echo "  PID: $PID_9V"

echo ""
echo "=================================================="
echo "3 个实验已启动！"
echo ""
echo "进程 PID:"
echo "  - 3views: $PID_3V"
echo "  - 6views: $PID_6V"
echo "  - 9views: $PID_9V"
echo ""
echo "监控命令:"
echo "  tail -f output/${TIMESTAMP}_foot_*views_dropgaussian_official.log"
echo ""
echo "检查结果:"
echo "  for v in 3 6 9; do"
echo "    cat output/${TIMESTAMP}_foot_\${v}views_dropgaussian_official/eval/iter_030000/eval2d_render_test.yml | head -2"
echo "  done"
echo "=================================================="
