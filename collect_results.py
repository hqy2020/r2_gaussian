#!/usr/bin/env python3
"""收集 NeRF 基准方法的评估结果"""
import os
import yaml
from pathlib import Path
from collections import defaultdict

def collect_results():
    output_dir = Path("output")
    results = defaultdict(dict)

    # 收集 2025_12_15 开始的实验
    for exp_dir in output_dir.glob("*_2025_12_1*"):
        name = exp_dir.name

        # 解析方法和场景
        for method in ["tensorf", "saxnerf", "naf"]:
            if name.endswith(f"_{method}"):
                # 提取器官和视角
                parts = name.split("_")
                for i, p in enumerate(parts):
                    if p in ["foot", "chest", "head", "abdomen", "pancreas"]:
                        organ = p
                        views = parts[i+1].replace("views", "")
                        break
                else:
                    continue

                # 读取评估结果
                eval_file = exp_dir / f"eval/iter_030000/eval3d_{method}.yml"
                if eval_file.exists():
                    with open(eval_file) as f:
                        data = yaml.safe_load(f)
                    psnr = data.get("psnr_3d", 0)
                    ssim = data.get("ssim_3d", 0)
                    results[method][f"{organ}_{views}views"] = {"psnr": psnr, "ssim": ssim}

    return results

def print_results(results):
    organs = ["foot", "chest", "head", "abdomen", "pancreas"]
    views_list = ["3", "6", "9"]

    for method in ["tensorf", "saxnerf", "naf"]:
        if method not in results:
            print(f"\n=== {method.upper()} ===")
            print("(无结果)")
            continue

        print(f"\n=== {method.upper()} ===")
        print(f"{'场景':<20} {'PSNR':>8} {'SSIM':>8}")
        print("-" * 40)

        total_psnr, total_ssim, count = 0, 0, 0
        for organ in organs:
            for views in views_list:
                key = f"{organ}_{views}views"
                if key in results[method]:
                    r = results[method][key]
                    print(f"{key:<20} {r['psnr']:>8.2f} {r['ssim']:>8.4f}")
                    total_psnr += r['psnr']
                    total_ssim += r['ssim']
                    count += 1

        if count > 0:
            print("-" * 40)
            print(f"{'平均':<20} {total_psnr/count:>8.2f} {total_ssim/count:>8.4f}")
            print(f"(已完成 {count}/15 场景)")

if __name__ == "__main__":
    results = collect_results()
    print_results(results)
