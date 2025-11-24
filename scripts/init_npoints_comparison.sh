#!/bin/bash

###############################################################################
# 点云数量对比实验 - 初始化脚本
# 生成 4 种不同点云数量的初始化文件：25k, 50k, 75k, 100k
# 日期：2025-11-24
###############################################################################

source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

echo "======================================================================"
echo "🔧 生成不同点云数量的初始化文件"
echo "======================================================================"
echo ""

DATASET="data/369/foot_50_3views.pickle"
OUTPUT_DIR="data/init_points"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 测试配置
declare -a N_POINTS=(25000 50000 75000 100000)

for n in "${N_POINTS[@]}"; do
    echo "======================================================================"
    echo "⚙️  生成 ${n} 点初始化文件..."
    echo "======================================================================"

    # 设置输出文件路径
    OUTPUT_FILE="${OUTPUT_DIR}/init_foot_50_3views_${n}.npy"

    python3 initialize_pcd.py \
        --data $DATASET \
        --output "$OUTPUT_FILE" \
        --n_points $n \
        --recon_method fdk \
        --density_thresh 0.05 \
        --density_rescale 0.15

    # 检查文件是否生成
    if [ -f "$OUTPUT_FILE" ]; then
        echo "✅ 已保存: $OUTPUT_FILE"
        echo ""
    else
        echo "❌ 错误: 未能生成初始化文件 $OUTPUT_FILE"
        exit 1
    fi
done

echo "======================================================================"
echo "✅ 所有初始化文件生成完成！"
echo "======================================================================"
echo ""
echo "生成的文件："
ls -lh ${OUTPUT_DIR}/init_foot_50_3views_*.npy
echo ""
echo "下一步：运行 scripts/train_npoints_comparison.sh 启动并行训练"
echo "======================================================================"
