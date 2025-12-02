#!/usr/bin/env python3
"""
Bino 超参数搜索结果分析脚本
用法: python scripts/analyze_bino_search.py <search_dir>
"""

import sys
import pandas as pd
from pathlib import Path

def load_results(search_dir: Path):
    """加载两个场景的结果"""
    foot_file = search_dir / "foot_3views_results.csv"
    abdomen_file = search_dir / "abdomen_9views_results.csv"

    results = {}

    if foot_file.exists():
        df = pd.read_csv(foot_file)
        df = df[df['psnr'] != 'ERROR']
        df['psnr'] = df['psnr'].astype(float)
        df['delta_psnr'] = df['delta_psnr'].astype(float)
        results['foot'] = df
        print(f"Foot-3: 加载 {len(df)} 条结果")

    if abdomen_file.exists():
        df = pd.read_csv(abdomen_file)
        df = df[df['psnr'] != 'ERROR']
        df['psnr'] = df['psnr'].astype(float)
        df['delta_psnr'] = df['delta_psnr'].astype(float)
        results['abdomen'] = df
        print(f"Abdomen-9: 加载 {len(df)} 条结果")

    return results

def find_universal_best(results):
    """找到两个场景都超越 baseline 的最佳通用参数"""
    if 'foot' not in results or 'abdomen' not in results:
        print("需要两个场景的结果才能分析")
        return None

    foot_df = results['foot']
    abdomen_df = results['abdomen']

    # 合并结果
    merged = foot_df.merge(
        abdomen_df,
        on=['loss_weight', 'angle_offset', 'start_iter'],
        suffixes=('_foot', '_abdomen')
    )

    # 计算最小提升（木桶原理）
    merged['min_delta'] = merged[['delta_psnr_foot', 'delta_psnr_abdomen']].min(axis=1)
    merged['avg_delta'] = merged[['delta_psnr_foot', 'delta_psnr_abdomen']].mean(axis=1)

    # 筛选两个场景都超越 baseline 的
    both_positive = merged[(merged['delta_psnr_foot'] > 0) & (merged['delta_psnr_abdomen'] > 0)]

    print("\n" + "="*70)
    print("分析结果")
    print("="*70)

    print(f"\n总共 {len(merged)} 组参数完成测试")
    print(f"两个场景都超越 baseline: {len(both_positive)} 组")

    if len(both_positive) > 0:
        print("\n" + "-"*70)
        print("成功参数组合 (按 min_delta 排序):")
        print("-"*70)

        cols = ['loss_weight', 'angle_offset', 'start_iter',
                'delta_psnr_foot', 'delta_psnr_abdomen', 'min_delta']
        display_df = both_positive[cols].sort_values('min_delta', ascending=False)

        for _, row in display_df.iterrows():
            print(f"  lw={row['loss_weight']:.2f}, ao={row['angle_offset']:.2f}, si={int(row['start_iter'])}")
            print(f"    Foot-3:    Δ={row['delta_psnr_foot']:+.3f} dB")
            print(f"    Abdomen-9: Δ={row['delta_psnr_abdomen']:+.3f} dB")
            print(f"    Min Δ:     {row['min_delta']:+.3f} dB")
            print()

        # 最佳参数
        best = display_df.iloc[0]
        print("="*70)
        print("推荐的通用最佳参数:")
        print("="*70)
        print(f"  --binocular_loss_weight {best['loss_weight']}")
        print(f"  --binocular_max_angle_offset {best['angle_offset']}")
        print(f"  --binocular_start_iter {int(best['start_iter'])}")
        print(f"  --binocular_warmup_iters 3000")
        print(f"  --binocular_smooth_weight 0.05")
        print()
        print(f"预期效果:")
        print(f"  Foot-3:    +{best['delta_psnr_foot']:.3f} dB")
        print(f"  Abdomen-9: +{best['delta_psnr_abdomen']:.3f} dB")

        return best
    else:
        print("\n没有参数组合能同时在两个场景超越 baseline")

        # 显示各场景最佳
        print("\n各场景单独最佳:")
        foot_best = foot_df.loc[foot_df['delta_psnr'].idxmax()]
        abd_best = abdomen_df.loc[abdomen_df['delta_psnr'].idxmax()]

        print(f"\nFoot-3 最佳: lw={foot_best['loss_weight']}, ao={foot_best['angle_offset']}, si={foot_best['start_iter']}")
        print(f"  Δ={foot_best['delta_psnr']:+.3f} dB")

        print(f"\nAbdomen-9 最佳: lw={abd_best['loss_weight']}, ao={abd_best['angle_offset']}, si={abd_best['start_iter']}")
        print(f"  Δ={abd_best['delta_psnr']:+.3f} dB")

        return None

def show_heatmap(results):
    """显示参数热力图"""
    for scene, df in results.items():
        print(f"\n{scene.upper()} 参数敏感性分析:")
        print("-"*50)

        # 按 loss_weight 分组
        for lw in sorted(df['loss_weight'].unique()):
            subset = df[df['loss_weight'] == lw]
            avg_delta = subset['delta_psnr'].mean()
            print(f"  loss_weight={lw:.2f}: avg_Δ={avg_delta:+.3f} dB")

        print()

        # 按 angle_offset 分组
        for ao in sorted(df['angle_offset'].unique()):
            subset = df[df['angle_offset'] == ao]
            avg_delta = subset['delta_psnr'].mean()
            print(f"  angle_offset={ao:.2f}: avg_Δ={avg_delta:+.3f} dB")

        print()

        # 按 start_iter 分组
        for si in sorted(df['start_iter'].unique()):
            subset = df[df['start_iter'] == si]
            avg_delta = subset['delta_psnr'].mean()
            print(f"  start_iter={int(si)}: avg_Δ={avg_delta:+.3f} dB")

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/analyze_bino_search.py <search_dir>")
        print("示例: python scripts/analyze_bino_search.py output/bino_search_20251128")
        sys.exit(1)

    search_dir = Path(sys.argv[1])

    if not search_dir.exists():
        print(f"错误: 目录不存在 {search_dir}")
        sys.exit(1)

    results = load_results(search_dir)

    if not results:
        print("没有找到结果文件")
        sys.exit(1)

    # 参数敏感性分析
    show_heatmap(results)

    # 找最佳通用参数
    best = find_universal_best(results)

    return best

if __name__ == "__main__":
    main()
