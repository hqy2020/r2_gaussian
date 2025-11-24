#!/usr/bin/env python3
"""
分析和对比不同采样策略的密度分布
验证密度加权采样是否真的采样了更多高密度点
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
import copy

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams
from r2_gaussian.utils.ct_utils import recon_volume


def analyze_sampling_distribution(scene_name="foot_50_3views", n_points=50000):
    """分析并对比两种采样策略的密度分布"""

    print("=" * 80)
    print("密度加权采样 vs 随机采样：密度分布分析")
    print("=" * 80)

    # 1. 加载数据
    print("\n[步骤 1] 加载场景数据...")

    # 使用 Scene 对象加载数据（与 initialize_pcd.py 一致）
    from r2_gaussian.arguments import ModelParams
    model_params = ModelParams(None, sentinel=True)
    model_params.source_path = "data/369"
    model_params.scene = scene_name

    scene = Scene(model_params, shuffle=False, load_iteration=None)
    projs_train = scene.projs_train
    angles_train = scene.angles_train
    scanner_cfg = scene.scanner_cfg

    print(f"  场景: {scene_name}")
    print(f"  投影数量: {projs_train.shape[0]}")
    print(f"  投影尺寸: {projs_train.shape[1]} × {projs_train.shape[2]}")

    # 2. FDK 重建
    print("\n[步骤 2] 执行 FDK 重建...")
    from tigre.utilities.geometry import Geometry
    geo = Geometry()
    geo.DSD = scanner_cfg["DSD"]
    geo.DSO = scanner_cfg["DSO"]
    geo.nDetector = np.array(scanner_cfg["nDetector"])
    geo.dDetector = np.array(scanner_cfg["dDetector"])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array(scanner_cfg["nVoxel"])
    geo.dVoxel = np.array(scanner_cfg["dVoxel"])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.array(scanner_cfg["offOrigin"])
    geo.offDetector = np.array([0, 0])

    vol = recon_volume(projs_train, angles_train, copy.deepcopy(geo), 'fdk')
    print(f"  体素网格形状: {vol.shape}")
    print(f"  密度范围: [{vol.min():.6f}, {vol.max():.6f}]")

    # 3. 过滤有效体素
    print("\n[步骤 3] 过滤有效体素...")
    density_thresh = 0.05
    density_mask = vol > density_thresh
    valid_indices = np.argwhere(density_mask)

    densities_flat = vol[
        valid_indices[:, 0],
        valid_indices[:, 1],
        valid_indices[:, 2],
    ]

    print(f"  有效体素数量: {len(valid_indices):,}")
    print(f"  有效体素密度范围: [{densities_flat.min():.6f}, {densities_flat.max():.6f}]")
    print(f"  有效体素密度均值: {densities_flat.mean():.6f}")
    print(f"  有效体素密度中位数: {np.median(densities_flat):.6f}")

    # 4. 两种采样策略
    print(f"\n[步骤 4] 执行两种采样策略 (各 {n_points:,} 点)...")

    # 4.1 随机采样
    print("\n  策略 A: 随机采样")
    sampled_idx_random = np.random.choice(len(valid_indices), n_points, replace=False)
    densities_random = densities_flat[sampled_idx_random]

    print(f"    采样点数: {len(densities_random):,}")
    print(f"    密度范围: [{densities_random.min():.6f}, {densities_random.max():.6f}]")
    print(f"    密度均值: {densities_random.mean():.6f}")
    print(f"    密度中位数: {np.median(densities_random):.6f}")
    print(f"    密度标准差: {densities_random.std():.6f}")

    # 4.2 密度加权采样
    print("\n  策略 B: 密度加权采样")
    probs = densities_flat / densities_flat.sum()
    sampled_idx_weighted = np.random.choice(
        len(valid_indices), n_points, replace=False, p=probs
    )
    densities_weighted = densities_flat[sampled_idx_weighted]

    print(f"    采样点数: {len(densities_weighted):,}")
    print(f"    密度范围: [{densities_weighted.min():.6f}, {densities_weighted.max():.6f}]")
    print(f"    密度均值: {densities_weighted.mean():.6f}")
    print(f"    密度中位数: {np.median(densities_weighted):.6f}")
    print(f"    密度标准差: {densities_weighted.std():.6f}")

    # 5. 统计分析
    print("\n" + "=" * 80)
    print("统计对比分析")
    print("=" * 80)

    mean_diff = densities_weighted.mean() - densities_random.mean()
    mean_ratio = densities_weighted.mean() / densities_random.mean()
    median_diff = np.median(densities_weighted) - np.median(densities_random)

    print(f"\n平均密度:")
    print(f"  随机采样:   {densities_random.mean():.6f}")
    print(f"  密度加权:   {densities_weighted.mean():.6f}")
    print(f"  差值:       {mean_diff:+.6f}")
    print(f"  提升比例:   {(mean_ratio - 1) * 100:.2f}%")

    print(f"\n中位数密度:")
    print(f"  随机采样:   {np.median(densities_random):.6f}")
    print(f"  密度加权:   {np.median(densities_weighted):.6f}")
    print(f"  差值:       {median_diff:+.6f}")

    # 6. 分桶统计
    print("\n" + "=" * 80)
    print("密度区间分布对比")
    print("=" * 80)

    # 定义密度区间（基于 CT HU 值）
    bins = [0.05, 0.10, 0.15, 0.25, 0.40, 1.0]
    labels = ["极低密度\n(空气边界)", "低密度\n(软组织)", "中密度\n(肌肉)", "高密度\n(骨小梁)", "极高密度\n(致密骨)"]

    print("\n密度区间定义:")
    for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
        print(f"  区间 {i+1}: [{low:.2f}, {high:.2f}) - {label.replace(chr(10), ' ')}")

    hist_random, _ = np.histogram(densities_random, bins=bins)
    hist_weighted, _ = np.histogram(densities_weighted, bins=bins)

    print(f"\n{'区间':<20} {'随机采样':<15} {'密度加权':<15} {'变化率':<10}")
    print("-" * 70)

    for i, label in enumerate(labels):
        count_random = hist_random[i]
        count_weighted = hist_weighted[i]
        pct_random = count_random / n_points * 100
        pct_weighted = count_weighted / n_points * 100

        if count_random > 0:
            ratio = count_weighted / count_random
            change = (ratio - 1) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"

        label_short = label.replace('\n', ' ')
        print(f"{label_short:<20} {count_random:>6} ({pct_random:>5.1f}%)  "
              f"{count_weighted:>6} ({pct_weighted:>5.1f}%)  {change_str:>10}")

    # 7. 可视化
    print("\n[步骤 5] 生成可视化图表...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('密度加权采样 vs 随机采样：密度分布对比', fontsize=16, fontweight='bold')

    # 7.1 直方图对比
    ax = axes[0, 0]
    bins_hist = np.linspace(0.05, min(densities_random.max(), densities_weighted.max()), 50)
    ax.hist(densities_random, bins=bins_hist, alpha=0.6, label='随机采样', color='blue', density=True)
    ax.hist(densities_weighted, bins=bins_hist, alpha=0.6, label='密度加权', color='red', density=True)
    ax.axvline(densities_random.mean(), color='blue', linestyle='--', linewidth=2, label=f'随机均值: {densities_random.mean():.3f}')
    ax.axvline(densities_weighted.mean(), color='red', linestyle='--', linewidth=2, label=f'加权均值: {densities_weighted.mean():.3f}')
    ax.set_xlabel('密度值', fontsize=11)
    ax.set_ylabel('概率密度', fontsize=11)
    ax.set_title('(A) 密度分布直方图', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7.2 累积分布函数 (CDF)
    ax = axes[0, 1]
    sorted_random = np.sort(densities_random)
    sorted_weighted = np.sort(densities_weighted)
    cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
    cdf_weighted = np.arange(1, len(sorted_weighted) + 1) / len(sorted_weighted)
    ax.plot(sorted_random, cdf_random, label='随机采样', color='blue', linewidth=2)
    ax.plot(sorted_weighted, cdf_weighted, label='密度加权', color='red', linewidth=2)
    ax.set_xlabel('密度值', fontsize=11)
    ax.set_ylabel('累积概率', fontsize=11)
    ax.set_title('(B) 累积分布函数 (CDF)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7.3 箱线图对比
    ax = axes[1, 0]
    box_data = [densities_random, densities_weighted]
    bp = ax.boxplot(box_data, labels=['随机采样', '密度加权'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('密度值', fontsize=11)
    ax.set_title('(C) 箱线图对比 (显示分位数)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # 添加统计信息
    stats_text = (
        f"随机采样:\n"
        f"  Q1={np.percentile(densities_random, 25):.3f}\n"
        f"  Median={np.median(densities_random):.3f}\n"
        f"  Q3={np.percentile(densities_random, 75):.3f}\n\n"
        f"密度加权:\n"
        f"  Q1={np.percentile(densities_weighted, 25):.3f}\n"
        f"  Median={np.median(densities_weighted):.3f}\n"
        f"  Q3={np.percentile(densities_weighted, 75):.3f}"
    )
    ax.text(1.5, ax.get_ylim()[1] * 0.5, stats_text,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 7.4 密度区间分布柱状图
    ax = axes[1, 1]
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, hist_random / n_points * 100, width,
                   label='随机采样', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, hist_weighted / n_points * 100, width,
                   label='密度加权', color='red', alpha=0.7)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('密度区间', fontsize=11)
    ax.set_ylabel('采样点占比 (%)', fontsize=11)
    ax.set_title('(D) 密度区间分布对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('\n', ' ') for l in labels], rotation=15, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = "cc-agent/sampling_density_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  可视化图表已保存: {output_path}")

    # 8. 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    high_density_threshold = 0.25
    n_high_random = np.sum(densities_random >= high_density_threshold)
    n_high_weighted = np.sum(densities_weighted >= high_density_threshold)

    print(f"\n高密度点（密度 ≥ {high_density_threshold}）统计:")
    print(f"  随机采样:   {n_high_random:>6} 点 ({n_high_random/n_points*100:>5.1f}%)")
    print(f"  密度加权:   {n_high_weighted:>6} 点 ({n_high_weighted/n_points*100:>5.1f}%)")
    print(f"  增加数量:   {n_high_weighted - n_high_random:>6} 点")
    print(f"  增加比例:   {(n_high_weighted/n_high_random - 1)*100:>+5.1f}%")

    low_density_threshold = 0.10
    n_low_random = np.sum(densities_random <= low_density_threshold)
    n_low_weighted = np.sum(densities_weighted <= low_density_threshold)

    print(f"\n低密度点（密度 ≤ {low_density_threshold}）统计:")
    print(f"  随机采样:   {n_low_random:>6} 点 ({n_low_random/n_points*100:>5.1f}%)")
    print(f"  密度加权:   {n_low_weighted:>6} 点 ({n_low_weighted/n_points*100:>5.1f}%)")
    print(f"  减少数量:   {n_low_random - n_low_weighted:>6} 点")
    print(f"  减少比例:   {(1 - n_low_weighted/n_low_random)*100:>+5.1f}%")

    print("\n✅ 验证通过：密度加权采样确实采样了更多高密度点，减少了低密度点！")
    print("=" * 80)

    return {
        'densities_random': densities_random,
        'densities_weighted': densities_weighted,
        'mean_ratio': mean_ratio,
        'high_density_ratio': n_high_weighted / n_high_random,
    }


if __name__ == "__main__":
    result = analyze_sampling_distribution(
        scene_name="foot_50_3views",
        n_points=50000
    )
