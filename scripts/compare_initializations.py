#!/usr/bin/env python3
"""
对比不同初始化方法的工具
功能：
1. 生成多个初始化变体（baseline, de-init, smart-sampling, combined）
2. 定量评估（3D PSNR, 点云统计）
3. 定性可视化（点云分布、密度对比）
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.append("./")
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.arguments import ModelParams, PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.image_utils import metric_vol
from r2_gaussian.utils.general_utils import t2a
import torch


def evaluate_pointcloud_quality(xyz, density, vol_gt, scanner_cfg):
    """
    定量评估点云质量

    Returns:
        metrics (dict): 各种评估指标
    """
    metrics = {}

    # 1. 点云统计
    metrics['n_points'] = len(xyz)
    metrics['density_mean'] = float(np.mean(density))
    metrics['density_std'] = float(np.std(density))
    metrics['density_min'] = float(np.min(density))
    metrics['density_max'] = float(np.max(density))

    # 2. 空间分布统计
    for i, axis in enumerate(['x', 'y', 'z']):
        metrics[f'{axis}_mean'] = float(np.mean(xyz[:, i]))
        metrics[f'{axis}_std'] = float(np.std(xyz[:, i]))
        metrics[f'{axis}_range'] = float(np.ptp(xyz[:, i]))

    # 3. 密度分布统计
    density_percentiles = [10, 25, 50, 75, 90]
    for p in density_percentiles:
        metrics[f'density_p{p}'] = float(np.percentile(density, p))

    # 4. 高/低密度点比例
    high_density_thresh = 0.3
    metrics['high_density_ratio'] = float(np.mean(density > high_density_thresh))

    # 5. 点云重建 3D PSNR（需要 GaussianModel）
    try:
        # 创建临时 Gaussian model
        model_params = ModelParams(argparse.ArgumentParser())
        args = argparse.Namespace()
        for key, val in vars(model_params).items():
            setattr(args, key, val)

        gaussians = GaussianModel(scale_bound=None, args=args)
        gaussians.create_from_pcd(xyz, density, 1.0)

        pipe_params = PipelineParams(argparse.ArgumentParser())
        pipe_args = argparse.Namespace()
        for key, val in vars(pipe_params).items():
            setattr(pipe_args, key, val)

        # 体素化查询
        with torch.no_grad():
            vol_pred = query(
                gaussians,
                scanner_cfg["offOrigin"],
                scanner_cfg["nVoxel"],
                scanner_cfg["sVoxel"],
                pipe_args,
            )["vol"]

            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            metrics['psnr_3d'] = float(psnr_3d)
            print(f"    3D PSNR: {psnr_3d:.4f} dB")
    except Exception as e:
        print(f"    Warning: Failed to compute 3D PSNR: {e}")
        metrics['psnr_3d'] = None

    return metrics


def visualize_comparison(pointclouds_dict, output_path="init_comparison.png"):
    """
    可视化对比多个点云

    Args:
        pointclouds_dict: {name: (xyz, density)}
        output_path: 输出图片路径
    """
    n_methods = len(pointclouds_dict)
    fig = plt.figure(figsize=(6 * n_methods, 12))

    for idx, (name, (xyz, density)) in enumerate(pointclouds_dict.items()):
        # 随机采样 5k 点用于可视化
        n_vis = min(5000, len(xyz))
        indices = np.random.choice(len(xyz), n_vis, replace=False)
        xyz_vis = xyz[indices]
        density_vis = density[indices]

        # 3D 散点图
        ax1 = fig.add_subplot(3, n_methods, idx + 1, projection='3d')
        scatter = ax1.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2],
                             c=density_vis, cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{name}\n{len(xyz):,} points')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)

        # XY 平面投影
        ax2 = fig.add_subplot(3, n_methods, n_methods + idx + 1)
        ax2.scatter(xyz_vis[:, 0], xyz_vis[:, 1], c=density_vis, cmap='viridis', s=1, alpha=0.5)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('XY Projection')
        ax2.set_aspect('equal')

        # 密度分布直方图
        ax3 = fig.add_subplot(3, n_methods, 2 * n_methods + idx + 1)
        ax3.hist(density, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(density.mean(), color='red', linestyle='--',
                   label=f'Mean: {density.mean():.4f}')
        ax3.set_xlabel('Density')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Density Distribution')
        ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比可视化已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="对比不同初始化方法")
    parser.add_argument("--data", type=str, required=True, help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="init_comparison",
                       help="输出目录")
    parser.add_argument("--methods", type=str, nargs='+',
                       default=['baseline', 'denoise', 'smart', 'combined'],
                       help="要测试的方法")
    parser.add_argument("--n_points", type=int, default=50000, help="采样点数")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("初始化方法对比实验")
    print("=" * 80)
    print(f"数据集: {args.data}")
    print(f"输出目录: {output_dir}")
    print(f"测试方法: {args.methods}")
    print()

    # 加载数据集
    print("加载数据集...")
    model_params = ModelParams(argparse.ArgumentParser())
    args_temp = argparse.Namespace()
    for key, val in vars(model_params).items():
        setattr(args_temp, key, val)
    model_params = model_params.extract(args_temp)
    model_params.source_path = args.data

    scene = Scene(model_params, shuffle=False)
    vol_gt = scene.vol_gt.cuda()
    scanner_cfg = scene.scanner_cfg

    # 生成并评估各种初始化方法
    pointclouds = {}
    all_metrics = {}

    for method in args.methods:
        print(f"\n{'=' * 80}")
        print(f"生成初始化: {method}")
        print(f"{'=' * 80}")

        # 构建命令行参数
        cmd_parts = [
            "python", "initialize_pcd.py",
            "--data", args.data,
            "--n_points", str(args.n_points),
        ]

        if method == 'baseline':
            output_name = f"init_{Path(args.data).stem}_baseline.npy"
        elif method == 'denoise':
            output_name = f"init_{Path(args.data).stem}_denoise.npy"
            cmd_parts += ["--enable_denoise"]
        elif method == 'smart':
            output_name = f"init_{Path(args.data).stem}_smart.npy"
            cmd_parts += ["--enable_smart_sampling"]
        elif method == 'combined':
            output_name = f"init_{Path(args.data).stem}_combined.npy"
            cmd_parts += ["--enable_denoise", "--enable_smart_sampling"]
        else:
            print(f"  ⚠️  未知方法: {method}")
            continue

        output_path = output_dir / output_name
        cmd_parts += ["--output", str(output_path)]

        # 执行初始化
        cmd = " ".join(cmd_parts)
        print(f"  命令: {cmd}")
        ret = os.system(cmd)

        if ret != 0:
            print(f"  ❌ 生成失败")
            continue

        # 加载点云
        pointcloud = np.load(output_path)
        xyz = pointcloud[:, :3]
        density = pointcloud[:, 3]
        pointclouds[method] = (xyz, density)

        # 评估
        print(f"\n  评估 {method}:")
        metrics = evaluate_pointcloud_quality(xyz, density, vol_gt, scanner_cfg)
        all_metrics[method] = metrics

        # 打印关键指标
        print(f"    点数: {metrics['n_points']:,}")
        print(f"    密度: {metrics['density_mean']:.4f} ± {metrics['density_std']:.4f}")
        print(f"    密度范围: [{metrics['density_min']:.4f}, {metrics['density_max']:.4f}]")
        print(f"    高密度点比例: {metrics['high_density_ratio']:.2%}")

    # 保存所有评估结果
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ 评估指标已保存: {metrics_path}")

    # 生成对比可视化
    if len(pointclouds) > 0:
        print(f"\n生成对比可视化...")
        visualize_comparison(pointclouds, output_dir / "comparison.png")

    # 生成对比表格
    print(f"\n" + "=" * 80)
    print("📊 对比总结")
    print("=" * 80)
    print(f"{'方法':<15} {'3D PSNR (dB)':<15} {'密度均值':<15} {'高密度比例':<15}")
    print("-" * 80)

    baseline_psnr = all_metrics.get('baseline', {}).get('psnr_3d', None)
    for method, metrics in all_metrics.items():
        psnr = metrics.get('psnr_3d', None)
        psnr_str = f"{psnr:.4f}" if psnr is not None else "N/A"

        if psnr is not None and baseline_psnr is not None and method != 'baseline':
            diff = psnr - baseline_psnr
            psnr_str += f" ({diff:+.4f})"

        print(f"{method:<15} {psnr_str:<15} "
              f"{metrics['density_mean']:.4f}        "
              f"{metrics['high_density_ratio']:.2%}")

    print("=" * 80)


if __name__ == "__main__":
    main()
