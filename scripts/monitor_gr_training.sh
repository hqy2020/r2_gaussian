#!/bin/bash
# 📊 GR-Gaussian 训练监控工具
#
# 用法: ./scripts/monitor_gr_training.sh <output_dir>
# 示例: ./scripts/monitor_gr_training.sh output/2025_11_24_01_30_gr_foot3_30k_FIXED

if [ -z "$1" ]; then
    echo "❌ 请提供输出目录"
    echo "用法: $0 <output_dir>"
    echo "示例: $0 output/2025_11_24_01_30_gr_foot3_30k_FIXED"
    exit 1
fi

OUTPUT_DIR="$1"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ 目录不存在: $OUTPUT_DIR"
    exit 1
fi

echo "========================================="
echo "  📊 GR-Gaussian 训练监控"
echo "========================================="
echo "目录: $OUTPUT_DIR"
echo ""

# 1. 检查训练进度
echo "📈 训练进度："
if [ -f "$OUTPUT_DIR/train.log" ]; then
    LAST_ITER=$(grep -oP 'Iteration \K[0-9]+' "$OUTPUT_DIR/train.log" | tail -1)
    echo "   最新迭代: $LAST_ITER / 30000"
    echo ""
else
    echo "   ⚠️ 未找到 train.log"
    echo ""
fi

# 2. 检查 Graph Laplacian Loss
echo "🔍 Graph Laplacian Loss (最近10次)："
if [ -f "$OUTPUT_DIR/train.log" ]; then
    grep "graph_loss" "$OUTPUT_DIR/train.log" | tail -10 || echo "   ⚠️ 未找到 graph_loss 记录"
else
    echo "   ⚠️ 未找到日志文件"
fi
echo ""

# 3. 检查 De-Init 是否生效
echo "🌟 De-Init 降噪："
if [ -f "$OUTPUT_DIR/train.log" ]; then
    grep "De-Init" "$OUTPUT_DIR/train.log" || echo "   ⚠️ De-Init 未启用或无日志"
else
    echo "   ⚠️ 未找到日志文件"
fi
echo ""

# 4. 查看最新评估结果
echo "📊 最新评估结果："
for iter_dir in "$OUTPUT_DIR"/eval/iter_*; do
    if [ -d "$iter_dir" ]; then
        ITER=$(basename "$iter_dir")
        EVAL_FILE="$iter_dir/eval2d_render_test.yml"
        if [ -f "$EVAL_FILE" ]; then
            PSNR=$(grep "^psnr_2d:" "$EVAL_FILE" | awk '{print $2}')
            SSIM=$(grep "^ssim_2d:" "$EVAL_FILE" | awk '{print $2}')
            echo "   $ITER: PSNR=$PSNR dB, SSIM=$SSIM"
        fi
    fi
done | sort
echo ""

# 5. 显示实时日志（最后20行）
echo "📝 实时日志（最后20行）："
echo "========================================="
if [ -f "$OUTPUT_DIR/train.log" ]; then
    tail -20 "$OUTPUT_DIR/train.log"
else
    echo "⚠️ 未找到 train.log"
fi
echo "========================================="
echo ""

echo "💡 持续监控命令："
echo "   watch -n 10 '$0 $OUTPUT_DIR'"
echo ""
echo "💡 查看完整日志："
echo "   tail -f $OUTPUT_DIR/train.log"
