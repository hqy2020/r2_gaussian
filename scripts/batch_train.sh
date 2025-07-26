#!/bin/bash

DATA_DIR="/home/qyhu/Documents/r2_gaussian/data/369"
PYTHON_SCRIPT="/home/qyhu/Documents/r2_gaussian/train.py"

# 遍历所有非chest的pickle文件
for PICKLE_FILE in "$DATA_DIR"/*.pickle; do
    if [[ "$PICKLE_FILE" == *"chest"* ]]; then
        continue  # 跳过chest的文件
    fi

    echo "🚀 正在训练: $PICKLE_FILE"
    python "$PYTHON_SCRIPT" -s "$PICKLE_FILE"

    echo "✅ 完成训练: $PICKLE_FILE"
    echo "---------------------------------------------"
done
