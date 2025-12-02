#!/bin/bash
# 完整 Baseline 测试脚本
# 测试 5 个器官 (Chest, Foot, Head, Abdomen, Pancreas) x 3 种视角 (3, 6, 9)
# 用法: ./scripts/run_all_baseline_369.sh <GPU>

set -e

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate r2_gaussian_new

cd /home/qyhu/Documents/r2_ours/r2_gaussian

GPU=${1:-0}
TIMESTAMP=$(date +%Y_%m_%d_%H_%M)
DATA_DIR="data/369"
OUTPUT_BASE="output/baseline_369_${TIMESTAMP}"

# 公共训练参数
COMMON_FLAGS="--iterations 30000 --test_iterations 10000 20000 30000 --gaussiansN 1"

# 器官和视角配置
ORGANS=("chest" "foot" "head" "abdomen" "pancreas")
VIEWS=(3 6 9)

echo "=============================================="
echo "R²-Gaussian Baseline 完整测试"
echo "时间戳: $TIMESTAMP"
echo "GPU: $GPU"
echo "输出目录: $OUTPUT_BASE"
echo "=============================================="

mkdir -p "$OUTPUT_BASE"

# 记录开始时间
echo "开始时间: $(date)" > "${OUTPUT_BASE}/experiment_log.txt"

total_experiments=$((${#ORGANS[@]} * ${#VIEWS[@]}))
current=0

for organ in "${ORGANS[@]}"; do
    for views in "${VIEWS[@]}"; do
        current=$((current + 1))

        DATA_FILE="${DATA_DIR}/${organ}_50_${views}views.pickle"
        OUTPUT_DIR="${OUTPUT_BASE}/${organ}_${views}views"

        echo ""
        echo "=============================================="
        echo "[${current}/${total_experiments}] ${organ} ${views} views"
        echo "数据: $DATA_FILE"
        echo "输出: $OUTPUT_DIR"
        echo "=============================================="

        if [ ! -f "$DATA_FILE" ]; then
            echo "警告: 数据文件不存在: $DATA_FILE"
            continue
        fi

        mkdir -p "$OUTPUT_DIR"

        # 记录实验开始
        echo "[${current}/${total_experiments}] ${organ}_${views}views 开始: $(date)" >> "${OUTPUT_BASE}/experiment_log.txt"

        CUDA_VISIBLE_DEVICES=$GPU python train.py \
            -s "$DATA_FILE" \
            -m "$OUTPUT_DIR" \
            $COMMON_FLAGS \
            2>&1 | tee "${OUTPUT_DIR}/training.log"

        # 记录实验结束
        echo "[${current}/${total_experiments}] ${organ}_${views}views 结束: $(date)" >> "${OUTPUT_BASE}/experiment_log.txt"

        echo "完成: ${organ} ${views} views"
    done
done

echo ""
echo "=============================================="
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "=============================================="

# 记录结束时间
echo "结束时间: $(date)" >> "${OUTPUT_BASE}/experiment_log.txt"

# 汇总结果
echo ""
echo "=============================================="
echo "结果汇总"
echo "=============================================="

# 生成结果汇总
python << 'PYTHON_SCRIPT'
import os
import json
import sys

output_base = os.environ.get('OUTPUT_BASE', 'output/baseline_369')
organs = ['chest', 'foot', 'head', 'abdomen', 'pancreas']
views_list = [3, 6, 9]

print(f"\n{'器官':<12} {'视角':<8} {'PSNR':<12} {'SSIM':<12}")
print("-" * 44)

results = []
for organ in organs:
    for views in views_list:
        output_dir = f"{output_base}/{organ}_{views}views"
        results_file = f"{output_dir}/results.json"

        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                psnr = data.get('psnr', 'N/A')
                ssim = data.get('ssim', 'N/A')
                if isinstance(psnr, (int, float)):
                    psnr_str = f"{psnr:.4f}"
                else:
                    psnr_str = str(psnr)
                if isinstance(ssim, (int, float)):
                    ssim_str = f"{ssim:.4f}"
                else:
                    ssim_str = str(ssim)
                print(f"{organ:<12} {views:<8} {psnr_str:<12} {ssim_str:<12}")
                results.append({
                    'organ': organ,
                    'views': views,
                    'psnr': psnr,
                    'ssim': ssim
                })
            except Exception as e:
                print(f"{organ:<12} {views:<8} {'解析失败':<12} {str(e):<12}")
        else:
            print(f"{organ:<12} {views:<8} {'未找到':<12} {'未找到':<12}")

# 保存汇总结果
summary_file = f"{output_base}/summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n汇总已保存到: {summary_file}")
PYTHON_SCRIPT

export OUTPUT_BASE
