#!/bin/bash
# NeRF 批量重新评估脚本 - 只重新评估 2D 结果
# 使用 eval_max_views=50（与 3DGS 一致）
# 跳过 3D 评估（3D 结果不受 eval_max_views 影响）

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 使用完整路径的 Python
PYTHON="/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python"

echo "============================================================"
echo "NeRF 批量重新评估 2D 结果 (eval_max_views=50)"
echo "============================================================"
echo ""

total=0
success=0

# 查找所有今天训练的 NeRF 模型
for pth_file in $(find output -name "*_iter_30000.pth" -mtime -1 2>/dev/null | grep -E "(naf|tensorf|saxnerf)" | sort); do
    model_dir=$(dirname "$pth_file")
    dir_name=$(basename "$model_dir")
    total=$((total + 1))

    # 提取方法名
    if [[ "$dir_name" == *"_naf" ]]; then
        method="naf"
    elif [[ "$dir_name" == *"_tensorf" ]]; then
        method="tensorf"
    elif [[ "$dir_name" == *"_saxnerf" ]]; then
        method="saxnerf"
    else
        echo "[SKIP] 无法识别方法: $dir_name"
        continue
    fi

    # 提取 organ 和 views
    # 格式: _2025_12_16_xx_xx_organ_Nviews_method
    organ=$(echo "$dir_name" | sed -E 's/.*_([a-z]+)_([0-9]+)views_.*/\1/')
    views=$(echo "$dir_name" | sed -E 's/.*_([a-z]+)_([0-9]+)views_.*/\2/')

    # 构建数据集路径（正确格式: organ_50_Nviews.pickle）
    source_path="data/369/${organ}_50_${views}views.pickle"

    if [ ! -f "$source_path" ]; then
        echo "[SKIP] 数据集不存在: $source_path (dir: $dir_name)"
        continue
    fi

    echo "------------------------------------------------------------"
    echo "[$success/$total] $dir_name"
    echo "    方法: $method | 数据: ${organ}_50_${views}views"
    echo "------------------------------------------------------------"

    $PYTHON test_nerf.py \
        --method "$method" \
        --model_path "$model_dir" \
        --source_path "$source_path" \
        --iteration 30000 \
        --skip_3d \
        --eval

    if [ $? -eq 0 ]; then
        success=$((success + 1))
    fi
    echo ""
done

echo "============================================================"
echo "批量重新评估完成！成功: $success / $total"
echo "============================================================"
