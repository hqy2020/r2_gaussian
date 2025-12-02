#!/usr/bin/env python3
"""
Init-PCD 超参数搜索结果分析脚本

功能:
1. 收集所有实验的 PSNR/SSIM 结果
2. 按场景分别排序
3. 正交表分析：计算各因子的主效应
4. 推荐最优配置

用法:
    python scripts/analyze_init_pcd_search.py --timestamp 2025_11_29_00_30
    python scripts/analyze_init_pcd_search.py --output-dir output/init_pcd_search
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

# L9(3³) 正交表配置
L9_ORTHOGONAL_ARRAY = {
    "exp01": {"n_points": 25000, "density_thresh": 0.03, "density_rescale": 0.10},
    "exp02": {"n_points": 25000, "density_thresh": 0.05, "density_rescale": 0.15},
    "exp03": {"n_points": 25000, "density_thresh": 0.08, "density_rescale": 0.20},
    "exp04": {"n_points": 50000, "density_thresh": 0.03, "density_rescale": 0.15},
    "exp05": {"n_points": 50000, "density_thresh": 0.05, "density_rescale": 0.20},
    "exp06": {"n_points": 50000, "density_thresh": 0.08, "density_rescale": 0.10},
    "exp07": {"n_points": 75000, "density_thresh": 0.03, "density_rescale": 0.20},
    "exp08": {"n_points": 75000, "density_thresh": 0.05, "density_rescale": 0.10},
    "exp09": {"n_points": 75000, "density_thresh": 0.08, "density_rescale": 0.15},
}

# Baseline 性能
BASELINES = {
    "foot_3": {"psnr": 28.4873, "ssim": 0.9005},
    "abdomen_9": {"psnr": 29.2896, "ssim": 0.9366},
}

FACTOR_LEVELS = {
    "n_points": [25000, 50000, 75000],
    "density_thresh": [0.03, 0.05, 0.08],
    "density_rescale": [0.10, 0.15, 0.20],
}


def find_experiments(output_dir: Path, timestamp: str = None):
    """查找所有实验目录"""
    experiments = []

    pattern = f"{timestamp}_exp*" if timestamp else "exp*"

    for exp_dir in output_dir.glob(pattern):
        if not exp_dir.is_dir():
            continue

        # 解析目录名
        name = exp_dir.name
        match = re.search(r"(exp\d+)_(foot_3|abdomen_9)", name)
        if match:
            exp_id = match.group(1)
            scene_id = match.group(2)
            experiments.append({
                "dir": exp_dir,
                "exp_id": exp_id,
                "scene_id": scene_id,
                "params": L9_ORTHOGONAL_ARRAY.get(exp_id, {}),
            })

    return experiments


def read_results(exp_dir: Path):
    """读取实验结果"""
    results = {"psnr": None, "ssim": None, "iterations": None}

    # 尝试读取 results.json
    results_file = exp_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            # 获取最后一次评估结果
            if "ours_None" in data:
                results["psnr"] = data["ours_None"].get("PSNR")
                results["ssim"] = data["ours_None"].get("SSIM")
        return results

    # 尝试从日志文件解析
    log_file = Path(str(exp_dir) + ".log")
    if log_file.exists():
        with open(log_file) as f:
            content = f.read()
            # 查找最后的评估结果
            psnr_matches = re.findall(r"PSNR[:\s]+(\d+\.?\d*)", content)
            ssim_matches = re.findall(r"SSIM[:\s]+(\d+\.?\d*)", content)
            if psnr_matches:
                results["psnr"] = float(psnr_matches[-1])
            if ssim_matches:
                results["ssim"] = float(ssim_matches[-1])

    return results


def analyze_orthogonal_effects(experiments, scene_id):
    """分析正交表的主效应"""
    scene_exps = [e for e in experiments if e["scene_id"] == scene_id and e["results"]["psnr"] is not None]

    if len(scene_exps) < 3:
        return None

    effects = {}

    for factor, levels in FACTOR_LEVELS.items():
        level_means = {}
        for level in levels:
            level_exps = [e for e in scene_exps if e["params"].get(factor) == level]
            if level_exps:
                mean_psnr = sum(e["results"]["psnr"] for e in level_exps) / len(level_exps)
                level_means[level] = mean_psnr

        if level_means:
            # 计算效应范围（最大 - 最小）
            effect_range = max(level_means.values()) - min(level_means.values())
            best_level = max(level_means, key=level_means.get)
            effects[factor] = {
                "level_means": level_means,
                "effect_range": effect_range,
                "best_level": best_level,
            }

    return effects


def print_results(experiments, timestamp):
    """打印分析结果"""
    print("\n" + "=" * 80)
    print(f"Init-PCD 超参数搜索结果分析")
    if timestamp:
        print(f"Timestamp: {timestamp}")
    print("=" * 80)

    for scene_id in ["foot_3", "abdomen_9"]:
        baseline = BASELINES[scene_id]
        scene_exps = [e for e in experiments if e["scene_id"] == scene_id]
        valid_exps = [e for e in scene_exps if e["results"]["psnr"] is not None]

        print(f"\n{'─' * 80}")
        print(f"场景: {scene_id} | Baseline PSNR: {baseline['psnr']:.4f}, SSIM: {baseline['ssim']:.4f}")
        print(f"{'─' * 80}")

        if not valid_exps:
            print("  [无有效结果]")
            continue

        # 按 PSNR 排序
        sorted_exps = sorted(valid_exps, key=lambda x: x["results"]["psnr"], reverse=True)

        print(f"\n{'Rank':<6} {'Exp':<8} {'n_points':<10} {'thresh':<10} {'rescale':<10} {'PSNR':<12} {'SSIM':<12} {'Δ PSNR':<10}")
        print("-" * 88)

        for i, exp in enumerate(sorted_exps, 1):
            p = exp["params"]
            r = exp["results"]
            delta = r["psnr"] - baseline["psnr"]
            delta_str = f"{delta:+.4f}" if delta else "N/A"
            ssim_str = f"{r['ssim']:.4f}" if r["ssim"] else "N/A"

            marker = "★" if delta > 0 else " "
            print(f"{marker}{i:<5} {exp['exp_id']:<8} {p.get('n_points', 'N/A'):<10} {p.get('density_thresh', 'N/A'):<10} {p.get('density_rescale', 'N/A'):<10} {r['psnr']:.4f}      {ssim_str:<12} {delta_str}")

        # 正交表效应分析
        effects = analyze_orthogonal_effects(experiments, scene_id)
        if effects:
            print(f"\n正交表主效应分析:")
            print("-" * 60)

            # 按效应大小排序
            sorted_factors = sorted(effects.items(), key=lambda x: x[1]["effect_range"], reverse=True)

            for factor, data in sorted_factors:
                print(f"\n  {factor} (效应范围: {data['effect_range']:.4f} dB, 最优水平: {data['best_level']})")
                for level, mean in sorted(data["level_means"].items()):
                    marker = "→" if level == data["best_level"] else " "
                    print(f"    {marker} {level}: {mean:.4f} dB")

    # 总结
    print(f"\n{'=' * 80}")
    print("推荐配置总结")
    print("=" * 80)

    for scene_id in ["foot_3", "abdomen_9"]:
        baseline = BASELINES[scene_id]
        scene_exps = [e for e in experiments if e["scene_id"] == scene_id and e["results"]["psnr"] is not None]

        if scene_exps:
            best = max(scene_exps, key=lambda x: x["results"]["psnr"])
            delta = best["results"]["psnr"] - baseline["psnr"]
            status = "✓ 超越 Baseline" if delta > 0 else "✗ 未超越 Baseline"
            print(f"\n{scene_id}:")
            print(f"  最优配置: {best['exp_id']}")
            print(f"  参数: n_points={best['params'].get('n_points')}, thresh={best['params'].get('density_thresh')}, rescale={best['params'].get('density_rescale')}")
            print(f"  PSNR: {best['results']['psnr']:.4f} (Δ{delta:+.4f})")
            print(f"  状态: {status}")


def main():
    parser = argparse.ArgumentParser(description="Init-PCD 超参数搜索结果分析")
    parser.add_argument("--timestamp", type=str, help="实验时间戳 (e.g., 2025_11_29_00_30)")
    parser.add_argument("--output-dir", type=str, default="output/init_pcd_search", help="实验输出目录")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return

    experiments = find_experiments(output_dir, args.timestamp)

    if not experiments:
        print(f"No experiments found in {output_dir}")
        if args.timestamp:
            print(f"Timestamp pattern: {args.timestamp}_exp*")
        return

    print(f"Found {len(experiments)} experiments")

    # 读取结果
    for exp in experiments:
        exp["results"] = read_results(exp["dir"])

    # 打印分析
    print_results(experiments, args.timestamp)


if __name__ == "__main__":
    main()
