#!/bin/bash
# 等待 Bino 实验完成后启动 FSGS 超参数搜索
# 用法: nohup ./scripts/queue_fsgs_after_bino.sh > output/fsgs_search/queue.log 2>&1 &

cd /home/qyhu/Documents/r2_ours/r2_gaussian

echo "=============================================="
echo "FSGS 超参数搜索队列"
echo "时间: $(date)"
echo "=============================================="
echo "等待 Bino 实验完成..."

# 获取 Bino 实验的 PID
BINO_PIDS=$(ps aux | grep "bino_search_20251128" | grep "train.py" | grep -v grep | awk '{print $2}')
echo "Bino PIDs: $BINO_PIDS"
echo ""

# 等待 Bino 实验完成
check_interval=120  # 每 2 分钟检查一次

while true; do
    # 检查是否还有 bino_search_20251128 目录相关的训练进程在运行
    RUNNING_COUNT=$(ps aux | grep "bino_search_20251128" | grep "train.py" | grep -v grep | wc -l)

    if [ "$RUNNING_COUNT" -eq 0 ]; then
        echo ""
        echo "[$(date)] Bino 实验已完成!"
        break
    fi

    # 获取进度信息
    for pid in $BINO_PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            echo -n "[$(date '+%H:%M:%S')] PID $pid 仍在运行... "
        fi
    done
    echo "等待 ${check_interval}s"
    sleep $check_interval
done

echo ""
echo "=============================================="
echo "启动 FSGS 超参数搜索"
echo "时间: $(date)"
echo "=============================================="

# 等待 5 秒确保资源释放
sleep 5

# 启动 FSGS 搜索 (10k iterations)
bash scripts/fsgs_search_parallel.sh 10000

echo ""
echo "FSGS 搜索已启动!"
echo "完成时间: $(date)"
