#!/bin/bash
# 一键运行初始化方法对比实验

echo "=========================================="
echo "R²-Gaussian 初始化方法对比实验"
echo "=========================================="

# 默认参数
DATA_PATH="${1:-data/369/foot_50_3views.pickle}"
OUTPUT_DIR="${2:-init_comparison_$(date +%Y%m%d_%H%M%S)}"
N_POINTS="${3:-50000}"

echo "数据集: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "采样点数: $N_POINTS"
echo ""

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "1. Baseline（原始方法）"
echo "=========================================="
python initialize_pcd.py \
    --data "$DATA_PATH" \
    --n_points "$N_POINTS" \
    --output "$OUTPUT_DIR/init_baseline.npy"

echo ""
echo "=========================================="
echo "2. De-Init（降噪初始化）"
echo "=========================================="
python initialize_pcd.py \
    --data "$DATA_PATH" \
    --n_points "$N_POINTS" \
    --enable_denoise \
    --denoise_sigma 3.0 \
    --output "$OUTPUT_DIR/init_denoise.npy"

echo ""
echo "=========================================="
echo "3. Smart Sampling（智能采样）"
echo "=========================================="
python initialize_pcd.py \
    --data "$DATA_PATH" \
    --n_points "$N_POINTS" \
    --enable_smart_sampling \
    --high_density_ratio 0.7 \
    --output "$OUTPUT_DIR/init_smart.npy"

echo ""
echo "=========================================="
echo "4. Combined（组合方法）"
echo "=========================================="
python initialize_pcd.py \
    --data "$DATA_PATH" \
    --n_points "$N_POINTS" \
    --enable_denoise \
    --denoise_sigma 3.0 \
    --enable_smart_sampling \
    --high_density_ratio 0.7 \
    --output "$OUTPUT_DIR/init_combined.npy"

echo ""
echo "=========================================="
echo "5. 生成对比可视化"
echo "=========================================="
python scripts/visualize_init_pointcloud.py \
    --npy_path "$OUTPUT_DIR/init_baseline.npy" \
    --output "$OUTPUT_DIR/baseline_viz.png"

python scripts/visualize_init_pointcloud.py \
    --npy_path "$OUTPUT_DIR/init_denoise.npy" \
    --output "$OUTPUT_DIR/denoise_viz.png"

python scripts/visualize_init_pointcloud.py \
    --npy_path "$OUTPUT_DIR/init_smart.npy" \
    --output "$OUTPUT_DIR/smart_viz.png"

python scripts/visualize_init_pointcloud.py \
    --npy_path "$OUTPUT_DIR/init_combined.npy" \
    --output "$OUTPUT_DIR/combined_viz.png"

echo ""
echo "=========================================="
echo "✅ 实验完成！"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "生成的文件:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "查看可视化结果:"
echo "  baseline:  $OUTPUT_DIR/baseline_viz.png"
echo "  denoise:   $OUTPUT_DIR/denoise_viz.png"
echo "  smart:     $OUTPUT_DIR/smart_viz.png"
echo "  combined:  $OUTPUT_DIR/combined_viz.png"
