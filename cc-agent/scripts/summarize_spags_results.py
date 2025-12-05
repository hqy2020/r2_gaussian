#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPAGS 消融实验结果汇总脚本
===========================

用法:
    python cc-agent/scripts/summarize_spags_results.py --pattern "spags_3k"

输出:
    output/spags_3k_summary.md - Markdown 格式汇总表
"""

import os
import sys
import argparse
import yaml
import glob
from datetime import datetime
from pathlib import Path

# R²-Gaussian baseline 3-views PSNR
BASELINE_PSNR = {
    "chest": {"3": 26.506, "6": 30.5, "9": 32.5},
    "foot": {"3": 28.4873, "6": 32.0, "9": 34.0},
    "head": {"3": 26.6915, "6": 30.0, "9": 32.0},
    "abdomen": {"3": 29.2896, "6": 33.0, "9": 35.0},
    "pancreas": {"3": 28.7669, "6": 32.5, "9": 34.5},
}

BASELINE_SSIM = {
    "chest": {"3": 0.8413, "6": 0.92, "9": 0.95},
    "foot": {"3": 0.9005, "6": 0.94, "9": 0.96},
    "head": {"3": 0.9247, "6": 0.95, "9": 0.97},
    "abdomen": {"3": 0.9366, "6": 0.96, "9": 0.97},
    "pancreas": {"3": 0.9247, "6": 0.95, "9": 0.97},
}


def find_experiments(output_dir: str, pattern: str):
    """查找匹配的实验目录"""
    experiments = []
    for path in glob.glob(os.path.join(output_dir, f"*{pattern}*")):
        if os.path.isdir(path):
            experiments.append(path)
    return sorted(experiments)


def parse_experiment_name(exp_path: str):
    """解析实验目录名，提取器官和视角"""
    name = os.path.basename(exp_path)
    # 格式: yyyy_MM_dd_HH_mm_organ_Nviews_spags_3k

    organs = ["foot", "chest", "head", "abdomen", "pancreas"]
    views = ["3", "6", "9"]

    organ = None
    view = None

    for o in organs:
        if f"_{o}_" in name:
            organ = o
            break

    for v in views:
        if f"_{v}views" in name:
            view = v
            break

    return organ, view, name


def read_eval_results(exp_path: str, iteration: int = 30000):
    """读取评估结果"""
    eval_path = os.path.join(exp_path, "eval", f"iter_{iteration:06d}", "eval3d.yml")

    if not os.path.exists(eval_path):
        return None

    with open(eval_path, 'r') as f:
        results = yaml.safe_load(f)

    return results


def generate_summary(experiments: list, iteration: int = 30000):
    """生成汇总数据"""
    summary = []

    for exp_path in experiments:
        organ, view, name = parse_experiment_name(exp_path)
        if organ is None or view is None:
            print(f"[WARN] Could not parse: {name}")
            continue

        results = read_eval_results(exp_path, iteration)
        if results is None:
            print(f"[WARN] No eval results for: {name}")
            continue

        psnr = results.get("psnr_3d", 0)
        ssim = results.get("ssim_3d", 0)

        # 计算相对baseline的提升
        baseline_psnr = BASELINE_PSNR.get(organ, {}).get(view, 0)
        baseline_ssim = BASELINE_SSIM.get(organ, {}).get(view, 0)
        delta_psnr = psnr - baseline_psnr if baseline_psnr else 0
        delta_ssim = ssim - baseline_ssim if baseline_ssim else 0

        summary.append({
            "organ": organ,
            "view": view,
            "psnr": psnr,
            "ssim": ssim,
            "baseline_psnr": baseline_psnr,
            "baseline_ssim": baseline_ssim,
            "delta_psnr": delta_psnr,
            "delta_ssim": delta_ssim,
            "exp_name": name,
            "exp_path": exp_path,
        })

    # 按器官和视角排序
    organ_order = ["chest", "foot", "head", "abdomen", "pancreas"]
    summary.sort(key=lambda x: (organ_order.index(x["organ"]) if x["organ"] in organ_order else 99, int(x["view"])))

    return summary


def generate_markdown(summary: list, output_path: str, pattern: str):
    """生成 Markdown 汇总表"""
    lines = []

    lines.append(f"# SPAGS 3k 消融实验结果汇总")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 实验模式: {pattern}")
    lines.append(f"- 实验数量: {len(summary)}")
    lines.append("")

    # 汇总表
    lines.append("## 结果汇总")
    lines.append("")
    lines.append("| 器官 | 视角 | PSNR (dB) | SSIM | Baseline PSNR | Baseline SSIM | PSNR 提升 | SSIM 提升 |")
    lines.append("|------|------|-----------|------|---------------|---------------|-----------|-----------|")

    for row in summary:
        delta_psnr_str = f"+{row['delta_psnr']:.3f}" if row['delta_psnr'] >= 0 else f"{row['delta_psnr']:.3f}"
        delta_ssim_str = f"+{row['delta_ssim']:.4f}" if row['delta_ssim'] >= 0 else f"{row['delta_ssim']:.4f}"
        lines.append(
            f"| {row['organ'].capitalize():8s} | {row['view']:4s} | "
            f"{row['psnr']:.4f} | {row['ssim']:.4f} | "
            f"{row['baseline_psnr']:.4f} | {row['baseline_ssim']:.4f} | "
            f"{delta_psnr_str:9s} | {delta_ssim_str:9s} |"
        )

    lines.append("")

    # 按视角统计平均值
    lines.append("## 按视角统计")
    lines.append("")

    for view in ["3", "6", "9"]:
        view_data = [r for r in summary if r["view"] == view]
        if not view_data:
            continue

        avg_psnr = sum(r["psnr"] for r in view_data) / len(view_data)
        avg_ssim = sum(r["ssim"] for r in view_data) / len(view_data)
        avg_delta_psnr = sum(r["delta_psnr"] for r in view_data) / len(view_data)
        avg_delta_ssim = sum(r["delta_ssim"] for r in view_data) / len(view_data)

        lines.append(f"### {view} 视角")
        lines.append(f"- 平均 PSNR: {avg_psnr:.4f} dB")
        lines.append(f"- 平均 SSIM: {avg_ssim:.4f}")
        lines.append(f"- 平均 PSNR 提升: {'+' if avg_delta_psnr >= 0 else ''}{avg_delta_psnr:.4f} dB")
        lines.append(f"- 平均 SSIM 提升: {'+' if avg_delta_ssim >= 0 else ''}{avg_delta_ssim:.4f}")
        lines.append("")

    # 实验详情
    lines.append("## 实验详情")
    lines.append("")
    for row in summary:
        lines.append(f"- `{row['exp_name']}`")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[SUCCESS] Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SPAGS 消融实验结果汇总")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--pattern", type=str, default="spags_3k", help="实验目录匹配模式")
    parser.add_argument("--iteration", type=int, default=30000, help="评估的iteration")
    parser.add_argument("--save_path", type=str, default=None, help="汇总文件保存路径")
    args = parser.parse_args()

    output_dir = args.output_dir
    pattern = args.pattern
    iteration = args.iteration

    print(f"[INFO] Searching for experiments matching: *{pattern}*")
    print(f"[INFO] Output directory: {output_dir}")

    experiments = find_experiments(output_dir, pattern)
    print(f"[INFO] Found {len(experiments)} experiments")

    if not experiments:
        print("[WARN] No experiments found")
        return

    summary = generate_summary(experiments, iteration)
    print(f"[INFO] Generated summary for {len(summary)} experiments")

    if not summary:
        print("[WARN] No valid results found")
        return

    # 保存汇总
    save_path = args.save_path or os.path.join(output_dir, f"{pattern}_summary.md")
    generate_markdown(summary, save_path, pattern)

    # 打印简要结果
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    for row in summary:
        delta = f"+{row['delta_psnr']:.2f}" if row['delta_psnr'] >= 0 else f"{row['delta_psnr']:.2f}"
        print(f"  {row['organ']:10s} {row['view']:2s}views: PSNR={row['psnr']:.2f} ({delta} dB)")


if __name__ == "__main__":
    main()
