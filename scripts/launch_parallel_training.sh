#!/bin/bash
# 并行启动四个训练（利用双 GPU）

TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_PATH="data/369/foot_50_3views.pickle"
ITERATIONS=30000

echo "=========================================="
echo "启动并行训练"
echo "=========================================="
echo "实验标识: ${TIMESTAMP}"
echo "GPU 0: baseline + smart"
echo "GPU 1: denoise + combined"
echo ""

# GPU 0: Baseline
echo "🚀 启动 baseline 训练 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 conda run -n r2_gaussian_new python train.py \
    -s "$DATA_PATH" \
    --ply_path "init_comparison_test/init_baseline.npy" \
    --iterations "$ITERATIONS" \
    --save_iterations 7000 15000 30000 \
    --test_iterations 1000 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 30000 \
    -m "output/${TIMESTAMP}_foot_3views_init_baseline" \
    > "output/${TIMESTAMP}_foot_3views_init_baseline_train.log" 2>&1 &
BASELINE_PID=$!
echo "  PID: $BASELINE_PID"
sleep 5

# GPU 0: Smart Sampling
echo "🚀 启动 smart 训练 (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 conda run -n r2_gaussian_new python train.py \
    -s "$DATA_PATH" \
    --ply_path "init_comparison_test/init_smart.npy" \
    --iterations "$ITERATIONS" \
    --save_iterations 7000 15000 30000 \
    --test_iterations 1000 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 30000 \
    -m "output/${TIMESTAMP}_foot_3views_init_smart" \
    > "output/${TIMESTAMP}_foot_3views_init_smart_train.log" 2>&1 &
SMART_PID=$!
echo "  PID: $SMART_PID"
sleep 5

# GPU 1: De-Init
echo "🚀 启动 denoise 训练 (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 conda run -n r2_gaussian_new python train.py \
    -s "$DATA_PATH" \
    --ply_path "init_comparison_test/init_denoise.npy" \
    --iterations "$ITERATIONS" \
    --save_iterations 7000 15000 30000 \
    --test_iterations 1000 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 30000 \
    -m "output/${TIMESTAMP}_foot_3views_init_denoise" \
    > "output/${TIMESTAMP}_foot_3views_init_denoise_train.log" 2>&1 &
DENOISE_PID=$!
echo "  PID: $DENOISE_PID"
sleep 5

# GPU 1: Combined
echo "🚀 启动 combined 训练 (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 conda run -n r2_gaussian_new python train.py \
    -s "$DATA_PATH" \
    --ply_path "init_comparison_test/init_combined.npy" \
    --iterations "$ITERATIONS" \
    --save_iterations 7000 15000 30000 \
    --test_iterations 1000 7000 15000 30000 \
    --checkpoint_iterations 7000 15000 30000 \
    -m "output/${TIMESTAMP}_foot_3views_init_combined" \
    > "output/${TIMESTAMP}_foot_3views_init_combined_train.log" 2>&1 &
COMBINED_PID=$!
echo "  PID: $COMBINED_PID"

echo ""
echo "=========================================="
echo "✅ 所有训练已启动"
echo "=========================================="
echo "baseline PID: $BASELINE_PID (GPU 0)"
echo "smart PID: $SMART_PID (GPU 0)"
echo "denoise PID: $DENOISE_PID (GPU 1)"
echo "combined PID: $COMBINED_PID (GPU 1)"
echo ""
echo "预计完成时间: $(date -d '+8 hours' '+%Y-%m-%d %H:%M')"
echo ""
echo "监控命令:"
echo "  查看进度: bash scripts/monitor_init_comparison.sh"
echo "  查看 GPU: watch -n 1 nvidia-smi"
echo "  实时日志: tail -f output/${TIMESTAMP}_foot_3views_init_baseline_train.log"
echo ""
echo "停止所有训练:"
echo "  kill $BASELINE_PID $SMART_PID $DENOISE_PID $COMBINED_PID"
echo "=========================================="

# 保存 PID 到文件
echo "$BASELINE_PID $SMART_PID $DENOISE_PID $COMBINED_PID" > /tmp/init_comparison_pids.txt
echo "实验标识: ${TIMESTAMP}" > /tmp/init_comparison_timestamp.txt
