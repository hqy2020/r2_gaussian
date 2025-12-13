#!/usr/bin/env python3
"""
诊断脚本：检查 NAF 训练和评估的完整数据流
"""

import numpy as np
import pickle
from pathlib import Path

def check_scanner_config():
    """检查扫描仪配置"""
    base_path = "/home/qyhu/Documents/r2_ours/r2_gaussian/data/369"
    pickle_path = f"{base_path}/foot_50_3views.pickle"

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\n{'='*60}")
    print("扫描仪配置 (scanner_cfg)")
    print(f"{'='*60}")

    # 关键配置
    print(f"\nnVoxel: {data.get('nVoxel', 'N/A')}")  # 体素网格大小
    print(f"dVoxel: {data.get('dVoxel', 'N/A')}")   # 体素尺寸
    print(f"offOrigin: {data.get('offOrigin', 'N/A')}")  # 原点偏移

    # 计算 sVoxel = nVoxel * dVoxel
    nVoxel = np.array(data.get('nVoxel', [0, 0, 0]))
    dVoxel = np.array(data.get('dVoxel', [0, 0, 0]))
    sVoxel = nVoxel * dVoxel
    print(f"\nsVoxel (计算值 = nVoxel * dVoxel): {sVoxel}")
    print(f"场景边界 (max(sVoxel)/2): {max(sVoxel) / 2}")

    # 体素坐标范围
    print(f"\n体素坐标范围:")
    for i, axis in enumerate(['x', 'y', 'z']):
        print(f"  {axis}: [{-sVoxel[i]/2:.4f}, {sVoxel[i]/2:.4f}]")

    # 检查 GT 体积
    if 'image' in data:
        image = data['image']
        print(f"\nGT 体积 ('image'):")
        print(f"  形状: {image.shape}")
        print(f"  范围: [{image.min():.6f}, {image.max():.6f}]")
        print(f"  均值: {image.mean():.6f}")
        print(f"  非零比例: {(image > 0).sum() / image.size:.4f}")

    return data


def check_training_loop_data():
    """检查训练循环中的数据使用"""
    base_path = "/home/qyhu/Documents/r2_ours/r2_gaussian/data/369"

    view_configs = [
        ("foot_50_3views.pickle", 3),
        ("foot_50_6views.pickle", 6),
        ("foot_50_9views.pickle", 9),
    ]

    print(f"\n{'='*60}")
    print("训练数据覆盖分析")
    print(f"{'='*60}")

    for pickle_name, num_views in view_configs:
        pickle_path = f"{base_path}/{pickle_name}"
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        train = data['train']
        projs = train['projections']  # [N, H, W]
        angles = train['angles']

        print(f"\n{pickle_name}:")
        print(f"  训练视角数: {num_views}")
        print(f"  训练角度 (弧度): {angles}")
        print(f"  训练角度 (度数): {np.degrees(angles)}")

        # 分析每个投影的信息量
        for i in range(num_views):
            proj = projs[i]
            info_content = (proj > 0.001).sum() / proj.size  # 有效像素比例
            print(f"  视角 {i} (角度={np.degrees(angles[i]):.1f}°): 有效像素比例={info_content:.4f}")

        # 检查视角之间的差异
        if num_views > 1:
            print(f"  视角间差异:")
            for i in range(num_views):
                for j in range(i+1, num_views):
                    diff = np.abs(projs[i] - projs[j]).mean()
                    print(f"    视角 {i} vs {j}: 平均差异 = {diff:.6f}")


def analyze_training_iterations():
    """分析训练迭代对不同视角数量的影响"""
    print(f"\n{'='*60}")
    print("训练迭代分析")
    print(f"{'='*60}")

    iterations = 30000
    n_rays = 1024  # 每次采样的射线数

    for num_views in [3, 6, 9]:
        total_rays = iterations * n_rays
        rays_per_view = total_rays / num_views
        epochs = iterations / num_views  # 完整遍历所有视角的次数

        print(f"\n{num_views} 视角:")
        print(f"  总迭代次数: {iterations}")
        print(f"  每次射线数: {n_rays}")
        print(f"  总采样射线: {total_rays:,}")
        print(f"  每视角平均射线: {rays_per_view:,.0f}")
        print(f"  完整 epoch 数: {epochs:,.0f}")

        # 计算每个像素被采样的平均次数
        image_size = 512 * 512
        samples_per_pixel = rays_per_view / image_size
        print(f"  每像素平均采样次数: {samples_per_pixel:.2f}")


if __name__ == "__main__":
    print("NAF 训练和评估诊断")
    print("="*60)

    # 1. 检查扫描仪配置
    data = check_scanner_config()

    # 2. 检查训练数据覆盖
    check_training_loop_data()

    # 3. 分析训练迭代
    analyze_training_iterations()
