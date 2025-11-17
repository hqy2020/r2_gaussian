#!/usr/bin/env python3
"""
SSS 训练结果分析脚本
"""

import yaml
import os
from pathlib import Path

# 配置
SSS_DIR = "output/2025_11_17_foot_3views_sss"
BASELINE_DIR = "output/foot_3_1013"
BASELINE_PSNR = 28.2796
BASELINE_SSIM = 0.8979

def read_metrics(eval_dir):
    """读取评估指标"""
    yml_path = os.path.join(eval_dir, "eval2d_render_test.yml")
    if not os.path.exists(yml_path):
        return None

    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)

    return {
        'psnr': data.get('psnr_2d', 0),
        'ssim': data.get('ssim_2d', 0)
    }

def main():
    print("=" * 60)
    print("📊 SSS 训练性能趋势分析")
    print("=" * 60)
    print()

    # 收集所有评估结果
    eval_base = Path(SSS_DIR) / "eval"
    iterations = sorted([d.name for d in eval_base.iterdir() if d.is_dir() and d.name.startswith("iter_")])

    print(f"{'Iteration':<12} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'vs Baseline':<15}")
    print("-" * 60)

    results = []
    for iter_name in iterations:
        iter_num = int(iter_name.replace("iter_", ""))
        metrics = read_metrics(eval_base / iter_name)

        if metrics:
            psnr_diff = metrics['psnr'] - BASELINE_PSNR
            ssim_diff = metrics['ssim'] - BASELINE_SSIM

            print(f"{iter_num:<12} | {metrics['psnr']:<10.2f} | {metrics['ssim']:<8.4f} | "
                  f"PSNR: {psnr_diff:+.2f} dB")

            results.append({
                'iter': iter_num,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'psnr_diff': psnr_diff
            })

    print("-" * 60)
    print(f"{'Baseline':<12} | {BASELINE_PSNR:<10.2f} | {BASELINE_SSIM:<8.4f} | (iter 10000)")
    print()

    # 性能分析
    if len(results) >= 2:
        print("📈 性能提升趋势:")
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            psnr_gain = curr['psnr'] - prev['psnr']
            iters = curr['iter'] - prev['iter']
            rate = psnr_gain / iters * 1000

            print(f"  iter {prev['iter']} → {curr['iter']}: "
                  f"+{psnr_gain:.2f} dB ({rate:+.3f} dB/k iters)")
        print()

    # 预测
    if len(results) >= 2:
        last = results[-1]
        if last['iter'] < 10000:
            remaining_iters = 10000 - last['iter']
            # 使用最近的提升率预测
            recent_gain = results[-1]['psnr'] - results[-2]['psnr']
            recent_iters = results[-1]['iter'] - results[-2]['iter']
            rate = recent_gain / recent_iters

            predicted_gain = rate * remaining_iters
            predicted_psnr = last['psnr'] + predicted_gain

            print("🔮 iter 10000 预测:")
            print(f"  基于 iter {results[-2]['iter']}→{last['iter']} 的提升率: {rate:+.5f} dB/iter")
            print(f"  预测 PSNR: {predicted_psnr:.2f} dB")
            print(f"  预测 vs Baseline: {predicted_psnr - BASELINE_PSNR:+.2f} dB")

            if predicted_psnr >= BASELINE_PSNR:
                print(f"  状态: ✅ 预计超越 baseline!")
            elif predicted_psnr >= BASELINE_PSNR - 0.5:
                print(f"  状态: ⚠️ 接近 baseline (差距 < 0.5 dB)")
            else:
                print(f"  状态: ❌ 低于 baseline")
            print()

    print("=" * 60)

if __name__ == "__main__":
    main()
