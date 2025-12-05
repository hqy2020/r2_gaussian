#!/bin/bash
# ============================================================================
# 生成 SPS 3k 密度加权点云
# ============================================================================
# 为所有 15 个场景 (5器官 × 3视角) 生成 3000 点的密度加权初始化点云
# ============================================================================

set -e

# 取消代理设置
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy ALL_PROXY all_proxy

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 输出目录
OUTPUT_DIR="data/369-sps-3k"
mkdir -p "$OUTPUT_DIR"

# 器官和视角
ORGANS=("foot" "chest" "head" "abdomen" "pancreas")
VIEWS=("3" "6" "9")

echo "============================================================================"
echo "生成 SPS 3k 密度加权点云"
echo "============================================================================"
echo "输出目录: $OUTPUT_DIR"
echo "器官: ${ORGANS[*]}"
echo "视角: ${VIEWS[*]}"
echo "总计: $((${#ORGANS[@]} * ${#VIEWS[@]})) 个点云"
echo "============================================================================"
echo ""

# 生成点云
for organ in "${ORGANS[@]}"; do
    for view in "${VIEWS[@]}"; do
        DATA_PATH="data/369/${organ}_50_${view}views.pickle"
        SAVE_PATH="${OUTPUT_DIR}/init_${organ}_50_${view}views.npy"

        if [ ! -f "$DATA_PATH" ]; then
            echo "警告: 数据集不存在: $DATA_PATH"
            continue
        fi

        if [ -f "$SAVE_PATH" ]; then
            echo "跳过 (已存在): $SAVE_PATH"
            continue
        fi

        echo ""
        echo ">>> 生成: ${organ}_${view}views"
        echo "    数据: $DATA_PATH"
        echo "    输出: $SAVE_PATH"

        python initialize_pcd.py \
            --data "$DATA_PATH" \
            --output "$SAVE_PATH" \
            --n_points 3000 \
            --enable_sps \
            --sps_strategy density_weighted

        echo ">>> 完成: ${organ}_${view}views"
    done
done

echo ""
echo "============================================================================"
echo "所有点云生成完成！"
echo "============================================================================"
ls -lh "$OUTPUT_DIR"
