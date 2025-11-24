#!/usr/bin/env python3
"""
可视化 FDK 生成的初始点云
用法: python scripts/visualize_init_pointcloud.py --npy_path data/369/init_foot_50_3views.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os


def visualize_pointcloud(npy_path, output_path=None, n_samples=5000):
    """
    可视化点云并保存图片

    Args:
        npy_path: .npy 文件路径
        output_path: 输出图片路径（如果为 None，自动生成）
        n_samples: 用于可视化的采样点数（太多会很慢）
    """
    # 加载点云
    point_cloud = np.load(npy_path)
    print(f"加载点云: {npy_path}")
    print(f"点云形状: {point_cloud.shape}")
    print(f"点数: {point_cloud.shape[0]:,}")

    # 提取坐标和密度
    xyz = point_cloud[:, :3]
    density = point_cloud[:, 3]

    # 统计信息
    print(f"\n坐标范围:")
    print(f"  X: [{xyz[:, 0].min():.4f}, {xyz[:, 0].max():.4f}]")
    print(f"  Y: [{xyz[:, 1].min():.4f}, {xyz[:, 1].max():.4f}]")
    print(f"  Z: [{xyz[:, 2].min():.4f}, {xyz[:, 2].max():.4f}]")
    print(f"\n密度范围: [{density.min():.4f}, {density.max():.4f}]")
    print(f"密度均值: {density.mean():.4f}")

    # 随机采样（避免可视化太慢）
    if xyz.shape[0] > n_samples:
        indices = np.random.choice(xyz.shape[0], n_samples, replace=False)
        xyz_vis = xyz[indices]
        density_vis = density[indices]
        print(f"\n随机采样 {n_samples:,} 个点进行可视化")
    else:
        xyz_vis = xyz
        density_vis = density

    # 创建图形（2x2 布局）
    fig = plt.figure(figsize=(16, 12))

    # 1. 3D 散点图（按密度着色）
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(xyz_vis[:, 0], xyz_vis[:, 1], xyz_vis[:, 2],
                         c=density_vis, cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Point Cloud (colored by density)\n{point_cloud.shape[0]:,} points total')
    plt.colorbar(scatter, ax=ax1, label='Density', shrink=0.5)

    # 2. XY 平面投影
    ax2 = fig.add_subplot(222)
    ax2.scatter(xyz_vis[:, 0], xyz_vis[:, 1], c=density_vis, cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Plane Projection')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # 3. XZ 平面投影
    ax3 = fig.add_subplot(223)
    ax3.scatter(xyz_vis[:, 0], xyz_vis[:, 2], c=density_vis, cmap='viridis', s=1, alpha=0.5)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Plane Projection')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 4. 密度分布直方图
    ax4 = fig.add_subplot(224)
    ax4.hist(density, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Density')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Density Distribution')
    ax4.axvline(density.mean(), color='red', linestyle='--', label=f'Mean: {density.mean():.4f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]
        output_path = f"{base_name}_visualization.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化图片已保存: {output_path}")

    # 显示图片
    # plt.show()  # 注释掉以避免弹窗

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 FDK 初始点云")
    parser.add_argument("--npy_path", type=str, default="data/369/init_foot_50_3views.npy",
                       help="点云 .npy 文件路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出图片路径（默认自动生成）")
    parser.add_argument("--n_samples", type=int, default=5000,
                       help="可视化采样点数（避免太慢）")

    args = parser.parse_args()

    visualize_pointcloud(args.npy_path, args.output, args.n_samples)
