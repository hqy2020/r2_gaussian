#!/usr/bin/env python3
"""FSGS 超参数搜索结果分析脚本"""

import pandas as pd
import os
from pathlib import Path

# 验证标准
THRESHOLDS = {
    "foot": {"psnr": 28.53, "ssim": 0.8976},
    "abdomen": {"psnr": 29.29, "ssim": 0.9366},
}

def analyze_results():
    results_file = "output/fsgs_search/results.csv"

    if not os.path.exists(results_file):
        print("结果文件不存在，请先运行搜索实验")
        return

    df = pd.read_csv(results_file)

    if df.empty:
        print("结果文件为空")
        return

    print("=" * 80)
    print("FSGS 超参数搜索结果分析")
    print("=" * 80)

    for organ in ["foot", "abdomen"]:
        organ_df = df[df["organ"] == organ].copy()
        if organ_df.empty:
            print(f"\n{organ}: 无结果")
            continue

        thresh = THRESHOLDS[organ]
        views = organ_df["views"].iloc[0]

        print(f"\n{'=' * 40}")
        print(f"{organ.upper()}-{views}views 结果")
        print(f"验证标准: PSNR > {thresh['psnr']}, SSIM > {thresh['ssim']}")
        print(f"{'=' * 40}")

        # 添加是否超越标准的标记
        organ_df["pass_psnr"] = organ_df["psnr"] > thresh["psnr"]
        organ_df["pass_ssim"] = organ_df["ssim"] > thresh["ssim"]
        organ_df["pass_both"] = organ_df["pass_psnr"] & organ_df["pass_ssim"]

        # 计算相对提升
        baseline_row = organ_df[organ_df["config"] == "baseline"]
        if not baseline_row.empty:
            baseline_psnr = baseline_row["psnr"].values[0]
            baseline_ssim = baseline_row["ssim"].values[0]
            organ_df["delta_psnr"] = organ_df["psnr"] - baseline_psnr
            organ_df["delta_ssim"] = organ_df["ssim"] - baseline_ssim
        else:
            organ_df["delta_psnr"] = 0
            organ_df["delta_ssim"] = 0

        # 按 PSNR 排序
        organ_df = organ_df.sort_values("psnr", ascending=False)

        print(f"\n按 PSNR 排序 (Top 5):")
        print("-" * 80)
        for i, row in organ_df.head(5).iterrows():
            status = "✅" if row["pass_both"] else ("⚠️" if row["pass_psnr"] or row["pass_ssim"] else "❌")
            print(f"{status} {row['config']:12s}: PSNR={row['psnr']:.4f} ({row['delta_psnr']:+.4f}), "
                  f"SSIM={row['ssim']:.4f} ({row['delta_ssim']:+.4f})")
            print(f"   参数: dpw={row['depth_pseudo_weight']}, fdw={row['fsgs_depth_weight']}, "
                  f"pt={row['proximity_threshold']}, pk={row['proximity_k_neighbors']}, start={row['start_sample_pseudo']}")

        # 统计通过率
        pass_count = organ_df["pass_both"].sum()
        total_count = len(organ_df)
        print(f"\n通过率: {pass_count}/{total_count} ({100*pass_count/total_count:.1f}%)")

        # 最佳配置
        if pass_count > 0:
            best = organ_df[organ_df["pass_both"]].iloc[0]
            print(f"\n🏆 最佳配置: {best['config']}")
            print(f"   PSNR={best['psnr']:.4f}, SSIM={best['ssim']:.4f}")
            print(f"   参数: depth_pseudo_weight={best['depth_pseudo_weight']}, "
                  f"fsgs_depth_weight={best['fsgs_depth_weight']}")
            print(f"         proximity_threshold={best['proximity_threshold']}, "
                  f"proximity_k_neighbors={int(best['proximity_k_neighbors'])}")
            print(f"         start_sample_pseudo={int(best['start_sample_pseudo'])}")

    # 综合分析
    print("\n" + "=" * 80)
    print("综合分析")
    print("=" * 80)

    # 找出在两个场景都表现好的配置
    foot_df = df[df["organ"] == "foot"].set_index("config")
    abd_df = df[df["organ"] == "abdomen"].set_index("config")

    common_configs = set(foot_df.index) & set(abd_df.index)

    if common_configs:
        print("\n配置在两个场景的表现:")
        print("-" * 80)
        for config in sorted(common_configs):
            if config in foot_df.index and config in abd_df.index:
                foot_psnr = foot_df.loc[config, "psnr"]
                abd_psnr = abd_df.loc[config, "psnr"]
                foot_pass = foot_psnr > THRESHOLDS["foot"]["psnr"]
                abd_pass = abd_psnr > THRESHOLDS["abdomen"]["psnr"]

                status = "✅✅" if (foot_pass and abd_pass) else ("✅❌" if foot_pass else ("❌✅" if abd_pass else "❌❌"))
                print(f"{status} {config:12s}: Foot={foot_psnr:.4f}, Abdomen={abd_psnr:.4f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_results()
