#!/bin/bash
# ============================================================================
# Foot 3views 消融实验批量运行脚本
# ============================================================================
# 已完成: baseline, sps (GPU0), adm (GPU1)
# 待运行: gar, sps_gar, sps_adm, gar_adm, spags
# ============================================================================

set -e
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# GPU0 队列: SPS 完成后运行
echo "=== GPU 0 队列 ==="
echo "等待 SPS 完成后运行: GAR -> SPS_GAR -> SPAGS"

# GPU1 队列: ADM 完成后运行
echo "=== GPU 1 队列 ==="
echo "等待 ADM 完成后运行: SPS_ADM -> GAR_ADM"

# GPU 0 序列
run_gpu0_queue() {
    echo "[GPU0] 等待 SPS 完成..."
    while pgrep -f "foot_3views_sps" > /dev/null; do sleep 60; done

    echo "[GPU0] 启动 GAR..."
    bash cc-agent/scripts/run_spags_ablation.sh gar foot 3 0

    echo "[GPU0] 启动 SPS_GAR..."
    bash cc-agent/scripts/run_spags_ablation.sh sps_gar foot 3 0

    echo "[GPU0] 启动 SPAGS..."
    bash cc-agent/scripts/run_spags_ablation.sh spags foot 3 0

    echo "[GPU0] 队列完成!"
}

# GPU 1 序列
run_gpu1_queue() {
    echo "[GPU1] 等待 ADM 完成..."
    while pgrep -f "foot_3views_adm" > /dev/null; do sleep 60; done

    echo "[GPU1] 启动 SPS_ADM..."
    bash cc-agent/scripts/run_spags_ablation.sh sps_adm foot 3 1

    echo "[GPU1] 启动 GAR_ADM..."
    bash cc-agent/scripts/run_spags_ablation.sh gar_adm foot 3 1

    echo "[GPU1] 队列完成!"
}

# 并行运行两个队列
run_gpu0_queue &
run_gpu1_queue &

wait
echo "=== 所有消融实验完成! ==="
