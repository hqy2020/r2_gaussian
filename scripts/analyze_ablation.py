#!/usr/bin/env python3
"""
R²-Gaussian 消融实验结果分析脚本
分析 16 配置 × 3 场景 = 48 个实验的结果
"""

import os
import yaml
import glob
import pandas as pd
from datetime import datetime

# 配置定义
CONFIG_NAMES = {
    0: "Baseline",
    1: "I",
    2: "X",
    3: "F",
    4: "B",
    5: "I+X",
    6: "I+F",
    7: "I+B",
    8: "X+F",
    9: "X+B",
    10: "F+B",
    11: "I+X+F",
    12: "I+X+B",
    13: "I+F+B",
    14: "X+F+B",
    15: "Full"
}

# 场景定义
SCENE_NAMES = {
    0: "Foot-3",
    1: "Chest-6",
    2: "Abdomen-9"
}

# Baseline SOTA 值 (从 CLAUDE.md)
BASELINE_SOTA = {
    "Foot-3": {"psnr": 28.487, "ssim": 0.9005},
    "Chest-6": {"psnr": 27.5, "ssim": 0.87},  # 估计值
    "Abdomen-9": {"psnr": 30.5, "ssim": 0.95}  # 估计值
}


def find_experiment_dirs(base_dir="output/ablation"):
    """查找所有消融实验目录"""
    results = []

    for scene_id, scene_name in SCENE_NAMES.items():
        organ, views = scene_name.lower().split("-")
        views = views.replace("views", "")

        # 匹配模式: *_organ_Nviews_configname
        pattern = os.path.join(base_dir, f"*_{organ}_{views}views_*")
        dirs = glob.glob(pattern)

        for d in dirs:
            if os.path.isdir(d):
                # 提取配置名
                dirname = os.path.basename(d)
                config_name = dirname.split(f"{views}views_")[-1]
                results.append({
                    "dir": d,
                    "scene": scene_name,
                    "config": config_name
                })

    return results


def read_results(exp_dir):
    """读取实验结果"""
    eval_path = os.path.join(exp_dir, "eval", "iter_030000", "eval2d_render_test.yml")

    if not os.path.exists(eval_path):
        # 尝试其他迭代
        for iter_num in [20000, 10000]:
            alt_path = os.path.join(exp_dir, "eval", f"iter_{iter_num:06d}", "eval2d_render_test.yml")
            if os.path.exists(alt_path):
                eval_path = alt_path
                break
        else:
            return None

    try:
        with open(eval_path, 'r') as f:
            data = yaml.safe_load(f)
        return {
            "psnr": data.get("psnr_2d", 0),
            "ssim": data.get("ssim_2d", 0)
        }
    except Exception as e:
        print(f"读取失败 {eval_path}: {e}")
        return None


def analyze_results():
    """分析所有结果"""
    experiments = find_experiment_dirs()

    if not experiments:
        print("未找到实验结果，请确认实验已完成")
        return

    # 收集结果
    results = []
    for exp in experiments:
        metrics = read_results(exp["dir"])
        if metrics:
            baseline = BASELINE_SOTA.get(exp["scene"], {"psnr": 0, "ssim": 0})
            results.append({
                "场景": exp["scene"],
                "配置": exp["config"],
                "PSNR": metrics["psnr"],
                "SSIM": metrics["ssim"],
                "ΔPSNR": metrics["psnr"] - baseline["psnr"],
                "ΔSSIM": metrics["ssim"] - baseline["ssim"],
                "目录": exp["dir"]
            })

    if not results:
        print("未找到有效结果")
        return

    # 创建 DataFrame
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("R²-Gaussian 消融实验结果分析")
    print("=" * 80)

    # 按场景分组显示
    for scene in SCENE_NAMES.values():
        scene_df = df[df["场景"] == scene].sort_values("PSNR", ascending=False)
        if scene_df.empty:
            continue

        print(f"\n### {scene} ###")
        print(scene_df[["配置", "PSNR", "SSIM", "ΔPSNR", "ΔSSIM"]].to_string(index=False))

        # 找出最佳配置
        best = scene_df.iloc[0]
        print(f"\n最佳配置: {best['配置']} (PSNR={best['PSNR']:.3f}, ΔPSNR={best['ΔPSNR']:+.3f})")

    # 找出全场景最佳配置
    print("\n" + "=" * 80)
    print("全场景分析")
    print("=" * 80)

    # 按配置聚合
    config_stats = df.groupby("配置").agg({
        "ΔPSNR": ["mean", "min", "max", "std"],
        "ΔSSIM": ["mean", "min", "max"]
    }).round(4)

    # 计算全场景超越 baseline 的次数
    df["超越Baseline"] = df["ΔPSNR"] > 0
    win_counts = df.groupby("配置")["超越Baseline"].sum()

    print("\n各配置在 3 场景的平均提升:")
    summary = df.groupby("配置").agg({
        "ΔPSNR": "mean",
        "ΔSSIM": "mean"
    }).round(4)
    summary["全胜场景数"] = win_counts
    summary = summary.sort_values("ΔPSNR", ascending=False)
    print(summary.to_string())

    # 找出全场景都超越 baseline 的配置
    full_win_configs = summary[summary["全胜场景数"] == 3]
    if not full_win_configs.empty:
        print("\n🏆 全场景超越 Baseline 的配置:")
        print(full_win_configs.to_string())

        # 推荐最佳配置
        best_config = full_win_configs["ΔPSNR"].idxmax()
        best_gain = full_win_configs.loc[best_config, "ΔPSNR"]
        print(f"\n✅ 推荐配置: {best_config} (平均 ΔPSNR = {best_gain:+.4f} dB)")
    else:
        print("\n⚠️ 没有配置在全部 3 场景都超越 Baseline")
        print("最接近的配置:")
        print(summary.head(3).to_string())

    # 保存详细结果
    output_file = f"output/ablation/analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存到: {output_file}")

    return df


if __name__ == "__main__":
    analyze_results()
