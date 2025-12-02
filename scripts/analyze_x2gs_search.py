#!/usr/bin/env python3
"""
X²-Gaussian TV 正则化超参数搜索结果分析脚本

用法:
    python scripts/analyze_x2gs_search.py [--dir OUTPUT_DIR]

输出:
    - 各超参数配置的 PSNR/SSIM 对比表
    - 最佳配置推荐
    - 与 baseline 的对比分析
"""

import argparse
import glob
import os
import re
import yaml
from collections import defaultdict
from typing import Dict, List, Tuple


# Baseline 参考值 (从消融实验获得)
BASELINES = {
    "foot_3views": {"psnr": 28.7314, "ssim": 0.8985},
    "abdomen_9views": {"psnr": 36.9478, "ssim": 0.9813},
}

# SOTA 参考值
SOTA = {
    "foot_3views": {"psnr": 28.4873, "ssim": 0.9005},
    "abdomen_9views": {"psnr": 29.2896, "ssim": 0.9366},
}


def parse_experiment_name(name: str) -> Dict:
    """解析实验目录名，提取配置信息"""
    # 格式: 2025_11_29_12_30_foot_3views_x2_tv1
    pattern = r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})_(\w+)_(\d+)views_x2_tv(\w+)"
    match = re.match(pattern, name)
    if match:
        timestamp, organ, views, tv_str = match.groups()
        # 还原 TV lambda 值
        if tv_str == "0":
            tv_lambda = 0.0
        elif len(tv_str) <= 4:
            tv_lambda = float(f"0.{tv_str.zfill(4)}")
        else:
            tv_lambda = float(f"0.{tv_str}")
        return {
            "timestamp": timestamp,
            "organ": organ,
            "views": int(views),
            "tv_lambda": tv_lambda,
            "dataset": f"{organ}_{views}views",
        }
    return None


def load_results(output_dir: str) -> List[Dict]:
    """加载所有实验结果"""
    results = []

    for exp_dir in sorted(glob.glob(os.path.join(output_dir, "*_x2_tv*"))):
        if not os.path.isdir(exp_dir):
            continue

        exp_name = os.path.basename(exp_dir)
        config = parse_experiment_name(exp_name)
        if not config:
            continue

        # 查找评估结果
        eval_file = os.path.join(exp_dir, "eval/iter_030000/eval2d_render_test.yml")
        if not os.path.exists(eval_file):
            # 尝试查找其他迭代
            for iter_dir in glob.glob(os.path.join(exp_dir, "eval/iter_*")):
                alt_file = os.path.join(iter_dir, "eval2d_render_test.yml")
                if os.path.exists(alt_file):
                    eval_file = alt_file
                    break

        if os.path.exists(eval_file):
            with open(eval_file) as f:
                data = yaml.safe_load(f)
            config["psnr"] = data.get("psnr_2d", 0)
            config["ssim"] = data.get("ssim_2d", 0)
            config["exp_dir"] = exp_dir
            results.append(config)

    return results


def analyze_results(results: List[Dict]) -> None:
    """分析实验结果"""
    # 按数据集分组
    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r["dataset"]].append(r)

    print("=" * 80)
    print("X²-Gaussian TV 正则化超参数搜索结果")
    print("=" * 80)

    best_configs = {}

    for dataset in sorted(by_dataset.keys()):
        exp_list = by_dataset[dataset]
        exp_list.sort(key=lambda x: x["tv_lambda"])

        baseline = BASELINES.get(dataset, {})
        sota = SOTA.get(dataset, {})

        print(f"\n### {dataset.upper()} ###")
        print(f"Baseline PSNR: {baseline.get('psnr', 'N/A'):.4f} dB")
        print(f"SOTA PSNR: {sota.get('psnr', 'N/A'):.4f} dB")
        print("-" * 70)
        print(f"{'TV Lambda':<15} {'PSNR (dB)':<12} {'vs Baseline':<15} {'SSIM':<10} {'Status'}")
        print("-" * 70)

        best_psnr = -float("inf")
        best_config = None

        for exp in exp_list:
            psnr = exp["psnr"]
            ssim = exp["ssim"]
            tv_lambda = exp["tv_lambda"]

            # 计算与 baseline 的差异
            diff = psnr - baseline.get("psnr", 0)
            diff_str = f"{diff:+.4f} dB"

            # 判断状态
            if psnr > baseline.get("psnr", 0):
                status = "✅ > baseline"
            elif psnr > sota.get("psnr", 0):
                status = "⚠️ > SOTA only"
            else:
                status = "❌ < SOTA"

            print(f"{tv_lambda:<15.5f} {psnr:<12.4f} {diff_str:<15} {ssim:<10.4f} {status}")

            if psnr > best_psnr:
                best_psnr = psnr
                best_config = exp

        if best_config:
            best_configs[dataset] = best_config
            print("-" * 70)
            print(f"🏆 Best: TV={best_config['tv_lambda']:.5f}, PSNR={best_config['psnr']:.4f} dB")

    # 综合分析
    print("\n" + "=" * 80)
    print("综合分析")
    print("=" * 80)

    # 检查是否有配置在两个数据集上都超过 baseline
    if len(best_configs) >= 2:
        all_tv_lambdas = set()
        for dataset, exp_list in by_dataset.items():
            for exp in exp_list:
                all_tv_lambdas.add(exp["tv_lambda"])

        print("\n各 TV Lambda 在所有数据集上的表现:")
        print("-" * 70)

        for tv_lambda in sorted(all_tv_lambdas):
            results_for_lambda = {}
            all_pass = True
            for dataset, exp_list in by_dataset.items():
                for exp in exp_list:
                    if exp["tv_lambda"] == tv_lambda:
                        results_for_lambda[dataset] = exp
                        baseline = BASELINES.get(dataset, {})
                        if exp["psnr"] <= baseline.get("psnr", 0):
                            all_pass = False
                        break

            if results_for_lambda:
                status = "✅ ALL PASS" if all_pass else "❌"
                parts = [f"{dataset}: {r['psnr']:.2f}" for dataset, r in results_for_lambda.items()]
                print(f"TV={tv_lambda:.5f}: {', '.join(parts)} {status}")


def main():
    parser = argparse.ArgumentParser(description="Analyze X²-Gaussian hyperparameter search results")
    parser.add_argument(
        "--dir",
        default="output/x2gs_tv_search",
        help="Output directory containing experiment results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory not found: {args.dir}")
        print("Please run the search first: bash scripts/x2gs_hyperparam_search.sh")
        return

    results = load_results(args.dir)

    if not results:
        print(f"No results found in {args.dir}")
        print("Make sure experiments have completed (look for eval/iter_030000/eval2d_render_test.yml)")
        return

    analyze_results(results)


if __name__ == "__main__":
    main()
