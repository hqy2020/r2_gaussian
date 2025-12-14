#!/usr/bin/env python3
"""
SPS (Spatial Prior Seeding) 诊断工具

功能：
1. 分析初始化点云的空间分布
2. 对比随机采样 vs 密度加权采样
3. 分析密度分布和采样偏差
4. 输出诊断报告

使用方法：
    python diagnose_sps.py --init_file data/369/init_foot_50_3views.npy --output_dir diagnosis/sps/

    # 对比随机 vs SPS 初始化
    python diagnose_sps.py \
        --baseline_init data/369/init_foot_50_3views.npy \
        --sps_init data/369-sps/init_foot_50_3views.npy \
        --output_dir diagnosis/sps/
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
# 兼容无显示/无 X server 环境（如 SSH/容器）
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_init_file(path: str) -> Dict:
    """
    加载初始化点云文件

    返回:
        dict: 包含 positions, densities
    """
    data = np.load(path)

    # 格式: [N, 4] -> [x, y, z, density]
    if data.ndim == 2 and data.shape[1] >= 4:
        positions = data[:, :3]
        densities = data[:, 3]
    elif data.ndim == 2 and data.shape[1] == 3:
        positions = data
        densities = np.ones(len(data))  # 没有密度信息
    else:
        raise ValueError(f"无法解析初始化文件格式: shape={data.shape}")

    return {
        'positions': positions,
        'densities': densities,
        'raw_data': data
    }


def compute_spatial_statistics(positions: np.ndarray) -> dict:
    """
    计算空间分布统计
    """
    # 边界框
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_size = bbox_max - bbox_min

    # 中心
    center = positions.mean(axis=0)

    # 各轴标准差
    std = positions.std(axis=0)

    # 点间距统计（采样计算，避免 O(N²)）
    # 优先使用 SciPy KDTree；若环境缺少 SciPy，则退化为小样本 brute-force（仍可用于诊断趋势）
    try:
        from scipy.spatial import cKDTree  # type: ignore
        use_kdtree = True
    except Exception:
        use_kdtree = False

    n_samples = min(1000, len(positions)) if use_kdtree else min(200, len(positions))
    sample_idx = np.random.choice(len(positions), n_samples, replace=False)
    sample_positions = positions[sample_idx]

    if use_kdtree:
        tree = cKDTree(positions)
        distances, _ = tree.query(sample_positions, k=2)  # k=2：第 1 个为自己
        nn_distances = distances[:, 1]
    else:
        nn_distances = []
        for p in sample_positions:
            d2 = np.sum((positions - p[None, :]) ** 2, axis=1)
            d2 = d2[d2 > 0]  # 排除自身
            nn_distances.append(np.sqrt(np.min(d2)) if d2.size > 0 else 0.0)
        nn_distances = np.asarray(nn_distances, dtype=np.float64)

    return {
        'count': len(positions),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_size': bbox_size.tolist(),
        'center': center.tolist(),
        'std_xyz': std.tolist(),
        'nn_distance_mean': float(np.mean(nn_distances)),
        'nn_distance_std': float(np.std(nn_distances)),
        'nn_distance_min': float(np.min(nn_distances)),
        'nn_distance_max': float(np.max(nn_distances)),
    }


def compute_density_statistics(densities: np.ndarray) -> dict:
    """
    计算密度分布统计
    """
    return {
        'count': len(densities),
        'mean': float(np.mean(densities)),
        'std': float(np.std(densities)),
        'min': float(np.min(densities)),
        'max': float(np.max(densities)),
        'median': float(np.median(densities)),
        'percentile_25': float(np.percentile(densities, 25)),
        'percentile_75': float(np.percentile(densities, 75)),
        'percentile_90': float(np.percentile(densities, 90)),
        'percentile_99': float(np.percentile(densities, 99)),
        'zero_ratio': float(np.mean(densities <= 1e-6)),
    }


def compute_distribution_uniformity(positions: np.ndarray, n_bins: int = 10) -> dict:
    """
    计算空间分布均匀性

    使用体素化网格统计每个格子的点数，计算均匀性指标
    """
    # 归一化到 [0, 1]
    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    bbox_size = bbox_max - bbox_min + 1e-6
    normalized = (positions - bbox_min) / bbox_size

    # 体素化
    voxel_indices = (normalized * n_bins).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, n_bins - 1)

    # 统计每个体素的点数
    counts = np.zeros((n_bins, n_bins, n_bins))
    for idx in voxel_indices:
        counts[idx[0], idx[1], idx[2]] += 1

    # 非空体素
    non_empty_counts = counts[counts > 0]

    # 均匀性指标
    expected_per_voxel = len(positions) / (n_bins ** 3)

    return {
        'n_bins': n_bins,
        'total_voxels': n_bins ** 3,
        'non_empty_voxels': int(np.sum(counts > 0)),
        'occupancy_ratio': float(np.mean(counts > 0)),
        'expected_per_voxel': float(expected_per_voxel),
        'actual_mean': float(np.mean(non_empty_counts)) if len(non_empty_counts) > 0 else 0,
        'actual_std': float(np.std(non_empty_counts)) if len(non_empty_counts) > 0 else 0,
        'cv': float(np.std(non_empty_counts) / np.mean(non_empty_counts)) if len(non_empty_counts) > 0 and np.mean(non_empty_counts) > 0 else 0,  # 变异系数
    }


def plot_spatial_distribution(positions: np.ndarray, densities: np.ndarray,
                              output_dir: Path, prefix: str = ''):
    """
    生成空间分布可视化
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：三视图投影
    # XY 投影
    ax = axes[0, 0]
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=densities,
                         cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XY Projection')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Density')

    # XZ 投影
    ax = axes[0, 1]
    scatter = ax.scatter(positions[:, 0], positions[:, 2], c=densities,
                         cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('XZ Projection')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Density')

    # YZ 投影
    ax = axes[0, 2]
    scatter = ax.scatter(positions[:, 1], positions[:, 2], c=densities,
                         cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('YZ Projection')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Density')

    # 第二行：直方图
    # 密度直方图
    ax = axes[1, 0]
    ax.hist(densities, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Density')
    ax.set_ylabel('Count')
    ax.set_title('Density Distribution')
    ax.axvline(np.mean(densities), color='red', linestyle='--', label=f'Mean={np.mean(densities):.4f}')
    ax.legend()

    # X 坐标直方图
    ax = axes[1, 1]
    ax.hist(positions[:, 0], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Count')
    ax.set_title('X Distribution')

    # Z 坐标直方图
    ax = axes[1, 2]
    ax.hist(positions[:, 2], bins=50, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Z')
    ax.set_ylabel('Count')
    ax.set_title('Z Distribution')

    plt.tight_layout()
    output_path = output_dir / f'{prefix}sps_spatial_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存空间分布图: {output_path}")


def plot_comparison(baseline_data: dict, sps_data: dict,
                    baseline_stats: dict, sps_stats: dict,
                    output_dir: Path):
    """
    生成 Baseline vs SPS 对比图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    baseline_pos = baseline_data['positions']
    baseline_den = baseline_data['densities']
    sps_pos = sps_data['positions']
    sps_den = sps_data['densities']

    # 密度直方图对比
    ax = axes[0, 0]
    ax.hist(baseline_den, bins=50, alpha=0.5, label='Baseline', color='blue')
    ax.hist(sps_den, bins=50, alpha=0.5, label='SPS', color='orange')
    ax.set_xlabel('Density')
    ax.set_ylabel('Count')
    ax.set_title('Density Distribution Comparison')
    ax.legend()

    # X 分布对比
    ax = axes[0, 1]
    ax.hist(baseline_pos[:, 0], bins=50, alpha=0.5, label='Baseline', color='blue')
    ax.hist(sps_pos[:, 0], bins=50, alpha=0.5, label='SPS', color='orange')
    ax.set_xlabel('X')
    ax.set_ylabel('Count')
    ax.set_title('X Distribution Comparison')
    ax.legend()

    # Z 分布对比
    ax = axes[0, 2]
    ax.hist(baseline_pos[:, 2], bins=50, alpha=0.5, label='Baseline', color='blue')
    ax.hist(sps_pos[:, 2], bins=50, alpha=0.5, label='SPS', color='orange')
    ax.set_xlabel('Z')
    ax.set_ylabel('Count')
    ax.set_title('Z Distribution Comparison')
    ax.legend()

    # Baseline XY 投影
    ax = axes[1, 0]
    ax.scatter(baseline_pos[:, 0], baseline_pos[:, 1], c=baseline_den,
               cmap='viridis', s=1, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Baseline XY Projection')
    ax.set_aspect('equal')

    # SPS XY 投影
    ax = axes[1, 1]
    ax.scatter(sps_pos[:, 0], sps_pos[:, 1], c=sps_den,
               cmap='viridis', s=1, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('SPS XY Projection')
    ax.set_aspect('equal')

    # 统计对比
    ax = axes[1, 2]
    ax.axis('off')

    # 统计信息（尽量用已计算的 stats，避免重复计算且保持一致）
    baseline_occ = baseline_stats['uniformity']['occupancy_ratio']
    sps_occ = sps_stats['uniformity']['occupancy_ratio']
    baseline_cv = baseline_stats['uniformity']['cv']
    sps_cv = sps_stats['uniformity']['cv']
    baseline_mean = baseline_stats['density']['mean']
    sps_mean = sps_stats['density']['mean']

    baseline_center = np.asarray(baseline_stats['spatial']['center'], dtype=np.float64)
    sps_center = np.asarray(sps_stats['spatial']['center'], dtype=np.float64)
    center_shift = float(np.linalg.norm(sps_center - baseline_center))

    stats_text = "统计对比:\n\n"
    stats_text += "Baseline:\n"
    stats_text += f"  - 点数: {len(baseline_pos)}\n"
    stats_text += f"  - 密度均值: {baseline_mean:.4f}\n"
    stats_text += f"  - 密度标准差: {baseline_stats['density']['std']:.4f}\n"
    stats_text += f"  - 占用率: {baseline_occ*100:.1f}%\n"
    stats_text += f"  - CV: {baseline_cv:.2f}\n\n"

    stats_text += "SPS:\n"
    stats_text += f"  - 点数: {len(sps_pos)}\n"
    stats_text += f"  - 密度均值: {sps_mean:.4f}\n"
    stats_text += f"  - 密度标准差: {sps_stats['density']['std']:.4f}\n"
    stats_text += f"  - 占用率: {sps_occ*100:.1f}%\n"
    stats_text += f"  - CV: {sps_cv:.2f}\n\n"

    stats_text += "差异:\n"
    if baseline_mean > 0:
        stats_text += f"  - 密度均值变化: {(sps_mean / baseline_mean - 1) * 100:+.1f}%\n"
    stats_text += f"  - 占用率变化: {(sps_occ - baseline_occ) * 100:+.1f}pp\n"
    if baseline_cv > 0:
        stats_text += f"  - CV变化: {(sps_cv / baseline_cv - 1) * 100:+.1f}%\n"
    stats_text += f"  - 中心偏移(L2): {center_shift:.3f}\n"

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'sps_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存对比图: {output_path}")


def generate_diagnosis_report(data: dict, stats: dict, output_path: str, name: str = 'SPS'):
    """
    生成诊断报告
    """
    report = {
        'name': name,
        'spatial_statistics': stats['spatial'],
        'density_statistics': stats['density'],
        'uniformity': stats['uniformity'],
        'diagnosis': []
    }

    diagnosis = report['diagnosis']

    # 诊断1：点数
    if stats['spatial']['count'] < 10000:
        diagnosis.append({
            'level': 'WARNING',
            'issue': '初始化点数较少',
            'detail': f"点数 = {stats['spatial']['count']}，可能影响重建质量",
            'suggestion': '考虑增加 n_points 参数'
        })

    # 诊断2：密度分布
    if stats['density']['std'] < 0.01:
        diagnosis.append({
            'level': 'WARNING',
            'issue': '密度分布过于均匀',
            'detail': f"密度标准差 = {stats['density']['std']:.4f}，缺乏区分度",
            'suggestion': '检查是否使用了密度加权采样'
        })

    # 诊断3：空间分布均匀性（过度集中可能导致初始化覆盖不足）
    if stats['uniformity']['cv'] > 1.0:
        diagnosis.append({
            'level': 'INFO',
            'issue': '空间分布不均匀',
            'detail': f"变异系数 CV = {stats['uniformity']['cv']:.2f}",
            'suggestion': '若后续训练效果不佳，可尝试 mixed/adaptive、降低 gamma、或启用 sps_denoise / density_clip'
        })

    # 诊断4：零密度比例
    if stats['density']['zero_ratio'] > 0.1:
        diagnosis.append({
            'level': 'WARNING',
            'issue': '存在大量零密度点',
            'detail': f"{stats['density']['zero_ratio']*100:.1f}% 的点密度接近0",
            'suggestion': '检查密度阈值设置'
        })

    if not diagnosis:
        diagnosis.append({
            'level': 'OK',
            'issue': '初始化点云状态正常',
            'detail': '各项指标在合理范围内',
            'suggestion': '无需调整'
        })

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ 保存诊断报告: {output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print(f"{name} 诊断报告摘要")
    print("="*60)
    print(f"点数: {stats['spatial']['count']}")
    print(f"边界框大小: {stats['spatial']['bbox_size']}")
    print(f"密度统计: 均值={stats['density']['mean']:.4f}, 标准差={stats['density']['std']:.4f}")
    print(f"密度范围: [{stats['density']['min']:.4f}, {stats['density']['max']:.4f}]")
    print(f"空间均匀性: 占用率={stats['uniformity']['occupancy_ratio']*100:.1f}%, CV={stats['uniformity']['cv']:.2f}")
    print("\n诊断结果:")
    for d in diagnosis:
        print(f"  [{d['level']}] {d['issue']}")
        print(f"         {d['detail']}")
    print("="*60)

    return report


def generate_comparison_report(baseline_stats: dict, sps_stats: dict, output_path: str):
    """
    生成 Baseline vs SPS 的对比诊断报告（用于解释 SPS 不及预期的潜在原因）
    """
    baseline_center = np.asarray(baseline_stats['spatial']['center'], dtype=np.float64)
    sps_center = np.asarray(sps_stats['spatial']['center'], dtype=np.float64)

    baseline_mean = float(baseline_stats['density']['mean'])
    sps_mean = float(sps_stats['density']['mean'])
    baseline_cv = float(baseline_stats['uniformity']['cv'])
    sps_cv = float(sps_stats['uniformity']['cv'])
    baseline_occ = float(baseline_stats['uniformity']['occupancy_ratio'])
    sps_occ = float(sps_stats['uniformity']['occupancy_ratio'])
    center_shift = float(np.linalg.norm(sps_center - baseline_center))

    report = {
        "name": "Baseline_vs_SPS",
        "metrics": {
            "density_mean_ratio": (sps_mean / baseline_mean) if baseline_mean > 0 else None,
            "cv_ratio": (sps_cv / baseline_cv) if baseline_cv > 0 else None,
            "occupancy_delta": sps_occ - baseline_occ,
            "center_shift_l2": center_shift,
        },
        "diagnosis": []
    }

    diag = report["diagnosis"]

    # 1) 空间覆盖不足
    if sps_occ < 0.90:
        diag.append({
            "level": "WARNING",
            "issue": "SPS 覆盖率偏低",
            "detail": f"occupancy_ratio={sps_occ:.3f} (<0.90)",
            "suggestion": "提高 mixed/adaptive 的均匀占比，或降低 density_thresh"
        })

    # 2) 过度集中（相对 baseline 明显更不均匀）
    if baseline_cv > 0 and (sps_cv / baseline_cv) > 1.4 and sps_cv > 0.8:
        diag.append({
            "level": "INFO",
            "issue": "SPS 分布更集中/不均匀",
            "detail": f"CV: {baseline_cv:.2f} -> {sps_cv:.2f}",
            "suggestion": "尝试增大 sps_uniform_ratio、减小 sps_density_gamma、启用 sps_denoise 或 density_clip"
        })

    # 3) 密度均值显著抬高（可能放大 FDK 伪影/高密度尖峰）
    if baseline_mean > 0 and (sps_mean / baseline_mean) > 1.8:
        diag.append({
            "level": "INFO",
            "issue": "SPS 采样偏向更高密度区域",
            "detail": f"density_mean: {baseline_mean:.4f} -> {sps_mean:.4f}",
            "suggestion": "若出现条纹伪影或训练不稳定，尝试启用 density_clip（如 99-99.5）或 sps_denoise"
        })

    # 4) 初始化中心偏移
    if center_shift > 0.20:
        diag.append({
            "level": "INFO",
            "issue": "SPS 初始化中心偏移较大",
            "detail": f"center_shift(L2)={center_shift:.3f}",
            "suggestion": "检查 FDK 伪影/阈值设置；可尝试 sps_denoise、降低 gamma 或提高均匀占比"
        })

    if not diag:
        diag.append({
            "level": "OK",
            "issue": "SPS 与 Baseline 差异在可控范围",
            "detail": "未检测到明显的覆盖不足或过度集中",
            "suggestion": "若效果仍不佳，优先从 denoise/clip/gamma/均匀占比四个方向调参"
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ 保存对比诊断报告: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='SPS 诊断工具')
    parser.add_argument('--init_file', type=str, default=None,
                        help='单个初始化文件路径 (.npy)')
    parser.add_argument('--baseline_init', type=str, default=None,
                        help='Baseline初始化文件路径')
    parser.add_argument('--sps_init', type=str, default=None,
                        help='SPS初始化文件路径')
    parser.add_argument('--output_dir', type=str, default='diagnosis/sps',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SPS 诊断工具")
    print(f"  输出目录: {output_dir}")
    print()

    # 单文件分析
    if args.init_file:
        print(f"加载初始化文件: {args.init_file}")
        data = load_init_file(args.init_file)
        print(f"  加载 {len(data['positions'])} 个点")

        # 计算统计
        stats = {
            'spatial': compute_spatial_statistics(data['positions']),
            'density': compute_density_statistics(data['densities']),
            'uniformity': compute_distribution_uniformity(data['positions']),
        }

        # 生成可视化
        print("\n生成可视化...")
        plot_spatial_distribution(data['positions'], data['densities'], output_dir)

        # 生成报告
        print("\n生成诊断报告...")
        generate_diagnosis_report(data, stats, str(output_dir / 'sps_diagnosis_report.json'))

    # 对比分析
    if args.baseline_init and args.sps_init:
        print(f"\n加载对比文件...")
        print(f"  Baseline: {args.baseline_init}")
        print(f"  SPS: {args.sps_init}")

        baseline_data = load_init_file(args.baseline_init)
        sps_data = load_init_file(args.sps_init)

        print(f"  Baseline: {len(baseline_data['positions'])} 个点")
        print(f"  SPS: {len(sps_data['positions'])} 个点")

        # 计算统计
        baseline_stats = {
            'spatial': compute_spatial_statistics(baseline_data['positions']),
            'density': compute_density_statistics(baseline_data['densities']),
            'uniformity': compute_distribution_uniformity(baseline_data['positions']),
        }

        sps_stats = {
            'spatial': compute_spatial_statistics(sps_data['positions']),
            'density': compute_density_statistics(sps_data['densities']),
            'uniformity': compute_distribution_uniformity(sps_data['positions']),
        }

        # 生成可视化
        print("\n生成可视化...")
        plot_spatial_distribution(baseline_data['positions'], baseline_data['densities'],
                                  output_dir, prefix='baseline_')
        plot_spatial_distribution(sps_data['positions'], sps_data['densities'],
                                  output_dir, prefix='sps_')
        plot_comparison(baseline_data, sps_data, baseline_stats, sps_stats, output_dir)
        generate_comparison_report(baseline_stats, sps_stats,
                                   str(output_dir / 'sps_comparison_report.json'))

        # 生成报告
        print("\n生成诊断报告...")
        generate_diagnosis_report(baseline_data, baseline_stats,
                                  str(output_dir / 'baseline_diagnosis_report.json'),
                                  name='Baseline')
        generate_diagnosis_report(sps_data, sps_stats,
                                  str(output_dir / 'sps_diagnosis_report.json'),
                                  name='SPS')

    print("\n诊断完成！")


if __name__ == '__main__':
    main()
