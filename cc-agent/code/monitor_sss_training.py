#!/usr/bin/env python3
"""
SSS (Student Splatting and Scooping) 训练监控脚本
实时监控 foot 3 views + SSS 训练进度
"""

import time
import os
from pathlib import Path

# 配置
OUTPUT_DIR = "output/2025_11_17_foot_3views_sss"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.txt")
MONITOR_INTERVAL = 30  # 秒

def parse_log_line(line):
    """解析训练日志行"""
    if "[ITER" in line:
        parts = line.split()
        iter_idx = next((i for i, p in enumerate(parts) if p.startswith("[ITER")), None)
        if iter_idx:
            iteration = parts[iter_idx].replace("[ITER", "").replace("]", "").strip()

            # 提取 Loss
            loss_idx = next((i for i, p in enumerate(parts) if "Loss:" in p), None)
            loss = parts[loss_idx + 1] if loss_idx else "N/A"

            # 提取 Gaussians 数量
            gaussians_idx = next((i for i, p in enumerate(parts) if "Gaussians:" in p or "Points:" in p), None)
            gaussians = parts[gaussians_idx + 1] if gaussians_idx else "N/A"

            return {
                "iteration": iteration,
                "loss": loss,
                "gaussians": gaussians
            }
    return None

def monitor_training():
    """监控训练进度"""
    print("=" * 80)
    print("🔍 SSS Training Monitor - foot 3 views")
    print("=" * 80)
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"📊 Log File: {LOG_FILE}")
    print(f"⏱️  Update Interval: {MONITOR_INTERVAL}s")
    print("=" * 80)

    last_position = 0
    while True:
        if not os.path.exists(LOG_FILE):
            print(f"⏳ [{time.strftime('%H:%M:%S')}] 等待训练日志生成...")
            time.sleep(MONITOR_INTERVAL)
            continue

        # 读取新增内容
        with open(LOG_FILE, "r") as f:
            f.seek(last_position)
            new_lines = f.readlines()
            last_position = f.tell()

        if new_lines:
            for line in new_lines:
                data = parse_log_line(line)
                if data:
                    print(f"📈 [{time.strftime('%H:%M:%S')}] "
                          f"Iteration {data['iteration']} | "
                          f"Loss: {data['loss']} | "
                          f"Gaussians: {data['gaussians']}")

                # 特殊事件检测
                if "SSS Status" in line:
                    print(f"🎯 {line.strip()}")
                elif "WARNING" in line or "ERROR" in line:
                    print(f"⚠️  {line.strip()}")
                elif "Training complete" in line:
                    print("✅ 训练完成!")
                    return

        time.sleep(MONITOR_INTERVAL)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\n🛑 监控已停止")
