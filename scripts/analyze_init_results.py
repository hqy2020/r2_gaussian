#!/usr/bin/env python3
"""
R²-Gaussian 点云初始化优化实验结果分析脚本
自动提取所有实验的 PSNR/SSIM 并生成排序报告
"""

import os
import yaml
import argparse
import pandas as pd
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="分析初始化优化实验结果")
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="实验时间戳 (格式: YYYY_MM_DD_HH_MM)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="训练迭代次数 (默认: 10000)"
    )
    parser.add_argument(
        "--baseline-psnr",
        type=float,
        default=28.480,
        help="Baseline PSNR (默认: 28.480)"
    )
    parser.add_argument(
        "--baseline-ssim",
        type=float,
        default=0.9008,
        help="Baseline SSIM (默认: 0.9008)"
    )
    return parser.parse_args()

def load_experiment_result(output_dir, exp_name, iterations):
    """加载单个实验的结果"""
    result_path = output_dir / exp_name / "eval" / f"iter_{iterations:06d}" / "eval2d_render_test.yml"

    if not result_path.exists():
        print(f"⚠️  警告: 结果文件不存在: {result_path}")
        return None

    try:
        with open(result_path, 'r') as f:
            data = yaml.safe_load(f)
        return {
            "psnr": data.get("psnr", None),
            "ssim": data.get("ssim", None)
        }
    except Exception as e:
        print(f"❌ 错误: 无法读取 {result_path}: {e}")
        return None

def main():
    args = parse_args()

    # 实验配置
    output_base = Path(f"output/init_optim_{args.timestamp}")

    if not output_base.exists():
        print(f"❌ 错误: 输出目录不存在: {output_base}")
        sys.exit(1)

    experiments = [
        {
            "id": "exp1_baseline",
            "name": "Baseline",
            "config": "50k, random, no denoise"
        },
        {
            "id": "exp2_denoise",
            "name": "De-Init降噪",
            "config": "50k, random, denoise σ=3"
        },
        {
            "id": "exp3_weighted",
            "name": "密度加权采样",
            "config": "50k, density_weighted, no denoise"
        },
        {
            "id": "exp4_more_points",
            "name": "增加点数",
            "config": "75k, random, no denoise"
        },
        {
            "id": "exp5_combined",
            "name": "组合优化",
            "config": "60k, density_weighted, denoise σ=3, rescale=0.20"
        },
        {
            "id": "exp6_high_thresh",
            "name": "更严格过滤",
            "config": "50k, random, denoise σ=3, thresh=0.08"
        }
    ]

    print("=" * 80)
    print("R²-Gaussian 点云初始化优化实验结果分析")
    print("=" * 80)
    print(f"时间戳: {args.timestamp}")
    print(f"训练迭代: {args.iterations}")
    print(f"Baseline: PSNR {args.baseline_psnr:.3f} dB, SSIM {args.baseline_ssim:.4f}")
    print("=" * 80)
    print()

    # 收集结果
    results = []
    for exp in experiments:
        result = load_experiment_result(output_base, exp["id"], args.iterations)
        if result and result["psnr"] is not None:
            results.append({
                "实验名称": exp["name"],
                "配置": exp["config"],
                "PSNR (dB)": result["psnr"],
                "SSIM": result["ssim"],
                "PSNR_Δ": result["psnr"] - args.baseline_psnr,
                "SSIM_Δ": result["ssim"] - args.baseline_ssim
            })
        else:
            print(f"⚠️  跳过 {exp['name']}: 结果未找到")

    if len(results) == 0:
        print("❌ 错误: 没有找到任何有效结果")
        sys.exit(1)

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 排序：优先 PSNR，次要 SSIM
    df_sorted = df.sort_values(["PSNR (dB)", "SSIM"], ascending=False)

    # 打印结果表格
    print("## 实验结果汇总（按 PSNR 降序）")
    print()
    print(df_sorted.to_string(index=False))
    print()

    # 识别最佳配置
    best_psnr_idx = df_sorted["PSNR (dB)"].idxmax()
    best_ssim_idx = df_sorted["SSIM"].idxmax()
    best_combined_idx = df_sorted.iloc[0].name  # 排序后的第一个

    print("=" * 80)
    print("## 关键发现")
    print("=" * 80)
    print()

    # PSNR 最佳
    best_psnr_row = df.loc[best_psnr_idx]
    print(f"🏆 **PSNR 最高**: {best_psnr_row['实验名称']}")
    print(f"   PSNR: {best_psnr_row['PSNR (dB)']:.3f} dB (+{best_psnr_row['PSNR_Δ']:.3f})")
    print(f"   SSIM: {best_psnr_row['SSIM']:.4f} ({best_psnr_row['SSIM_Δ']:+.4f})")
    print(f"   配置: {best_psnr_row['配置']}")
    print()

    # SSIM 最佳
    best_ssim_row = df.loc[best_ssim_idx]
    print(f"🏆 **SSIM 最高**: {best_ssim_row['实验名称']}")
    print(f"   PSNR: {best_ssim_row['PSNR (dB)']:.3f} dB ({best_ssim_row['PSNR_Δ']:+.3f})")
    print(f"   SSIM: {best_ssim_row['SSIM']:.4f} (+{best_ssim_row['SSIM_Δ']:.4f})")
    print(f"   配置: {best_ssim_row['配置']}")
    print()

    # 综合最佳
    best_combined_row = df.loc[best_combined_idx]
    print(f"🎯 **综合最佳** (PSNR优先): {best_combined_row['实验名称']}")
    print(f"   PSNR: {best_combined_row['PSNR (dB)']:.3f} dB ({best_combined_row['PSNR_Δ']:+.3f})")
    print(f"   SSIM: {best_combined_row['SSIM']:.4f} ({best_combined_row['SSIM_Δ']:+.4f})")
    print(f"   配置: {best_combined_row['配置']}")
    print()

    # 统计分析
    improvements = (df["PSNR_Δ"] > 0).sum()
    total = len(df)
    print(f"📊 **统计**: {improvements}/{total} 个配置超过 baseline PSNR")
    print()

    # 推荐下一步
    print("=" * 80)
    print("## 推荐下一步")
    print("=" * 80)
    print()

    if best_combined_row['PSNR_Δ'] > 0.1:
        print("✅ **建议执行阶段二**: 选出 PSNR 提升最显著的 2 个配置进行 30k 完整训练")
        print()
        # 选出前2名
        top2 = df_sorted.head(2)
        for idx, (i, row) in enumerate(top2.iterrows(), 1):
            print(f"   {idx}. {row['实验名称']}: PSNR +{row['PSNR_Δ']:.3f} dB, SSIM {row['SSIM_Δ']:+.4f}")
    else:
        print("⚠️  **警告**: 所有配置的 PSNR 提升都 ≤ 0.1 dB")
        print("   建议:")
        print("   1. 检查实验是否正确运行（查看日志）")
        print("   2. 尝试更激进的超参数（如 denoise_sigma=5, n_points=100k）")
        print("   3. 考虑引入其他初始化策略（如多尺度初始化）")
    print()

    # 保存结果到 CSV
    csv_path = output_base / f"results_summary_{args.iterations}k.csv"
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ 结果已保存至: {csv_path}")
    print()

if __name__ == "__main__":
    main()
