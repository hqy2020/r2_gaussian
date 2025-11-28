#!/bin/bash
# ============================================================================
# INIT-PCD: 批量生成密度加权采样点云
# ============================================================================
# 为 5 个器官 × 3 种视角配置 = 15 个场景生成密度加权初始化点云
# 输出目录: data/density-369/
#
# 使用方法:
#   bash scripts/generate_density_weighted_pcd.sh
#
# 训练时使用:
#   python train.py -s data/density-369/foot_50_3views.pickle ...
#   (会自动加载 data/density-369/init_foot_50_3views.npy)
# ============================================================================

set -e  # 遇到错误立即退出

# 配置
CONDA_ENV="r2_gaussian_new"
DATA_DIR="data/369"
OUTPUT_DIR="data/density-369"
SAMPLING_STRATEGY="density_weighted"

# 器官列表
ORGANS=("foot" "chest" "head" "abdomen" "pancreas")
# 视角配置
VIEWS=("3" "6" "9")

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "=============================================="
echo "INIT-PCD: 批量生成密度加权采样点云"
echo "=============================================="
echo "采样策略: $SAMPLING_STRATEGY"
echo "输入目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "=============================================="

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 计数器
total=0
success=0
failed=0

# 遍历所有场景
for organ in "${ORGANS[@]}"; do
    for view in "${VIEWS[@]}"; do
        total=$((total + 1))

        # 构建文件路径
        pickle_file="${DATA_DIR}/${organ}_50_${view}views.pickle"
        output_file="${OUTPUT_DIR}/init_${organ}_50_${view}views.npy"

        echo ""
        echo "[$total/15] 处理: ${organ}_50_${view}views"
        echo "  输入: $pickle_file"
        echo "  输出: $output_file"

        # 检查输入文件是否存在
        if [ ! -f "$pickle_file" ]; then
            echo "  [跳过] 输入文件不存在"
            failed=$((failed + 1))
            continue
        fi

        # 如果输出文件已存在，跳过
        if [ -f "$output_file" ]; then
            echo "  [跳过] 输出文件已存在"
            success=$((success + 1))
            continue
        fi

        # 生成点云
        python initialize_pcd.py \
            --data "$pickle_file" \
            --output "$output_file" \
            --sampling_strategy "$SAMPLING_STRATEGY" \
            --n_points 50000 \
            --density_thresh 0.05 \
            --density_rescale 0.15

        if [ $? -eq 0 ]; then
            echo "  [成功]"
            success=$((success + 1))
        else
            echo "  [失败]"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "=============================================="
echo "生成完成!"
echo "  总数: $total"
echo "  成功: $success"
echo "  失败: $failed"
echo "=============================================="

# 同时复制 pickle 文件的符号链接到输出目录，方便训练时使用
echo ""
echo "创建数据文件符号链接..."
for organ in "${ORGANS[@]}"; do
    for view in "${VIEWS[@]}"; do
        pickle_file="${organ}_50_${view}views.pickle"
        src="../369/${pickle_file}"
        dst="${OUTPUT_DIR}/${pickle_file}"

        if [ ! -e "$dst" ]; then
            ln -s "$src" "$dst"
            echo "  链接: $dst -> $src"
        fi
    done
done

echo ""
echo "完成! 训练时使用:"
echo "  python train.py -s data/density-369/foot_50_3views.pickle ..."
