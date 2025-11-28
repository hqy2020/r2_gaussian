#!/usr/bin/env python3
"""
从 data/1 的完整 50 视角数据重新生成 data/369 的 3/6/9 视角数据
同时生成标准的 random 采样初始化点云
"""
import os
import pickle
import numpy as np
from pathlib import Path

# 配置
INPUT_DIR = Path("/home/qyhu/Documents/r2_ours/r2_gaussian/data/1")
OUTPUT_DIR = Path("/home/qyhu/Documents/r2_ours/r2_gaussian/data/369_new")
ORGANS = ["chest", "foot", "abdomen", "head", "pancreas"]
TRAIN_NUMS = [3, 6, 9]
N_POINTS = 50000
DENSITY_THRESH = 0.05  # 体素密度阈值


def generate_sparse_views(input_path: Path, output_dir: Path, train_nums: list):
    """从完整 50 视角数据生成稀疏视角版本"""
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    total_views = data['numTrain']
    angles = np.array(data['train']['angles'])
    projections = np.array(data['train']['projections'])
    val_data = data['val']
    val_num = data['numVal']

    print(f"\n处理: {input_path.name}")
    print(f"  原始训练视角: {total_views}, 测试视角: {val_num}")

    for train_num in train_nums:
        # 等间隔选择训练视角
        if train_num == 1:
            train_indices = [0]
        else:
            train_indices = [round(i * (total_views - 1) / (train_num - 1)) for i in range(train_num)]

        new_data = dict(data)
        new_data['numTrain'] = len(train_indices)
        new_data['numVal'] = val_num
        new_data['train'] = {
            'angles': angles[train_indices],
            'projections': projections[train_indices]
        }
        new_data['val'] = val_data

        basename = input_path.stem  # e.g., "foot_50"
        output_name = f"{basename}_{train_num}views.pickle"
        output_path = output_dir / output_name

        with open(output_path, 'wb') as f:
            pickle.dump(new_data, f)

        print(f"  ✅ {train_num} views -> {output_path.name}")

        # 返回数据用于生成初始化点云
        yield train_num, new_data, output_path


def generate_init_pointcloud_random(data: dict, n_points: int, density_thresh: float) -> np.ndarray:
    """
    生成标准的 random 采样初始化点云

    使用 FDK 重建的体积，随机采样有效体素位置
    """
    # 从投影重建体积 (使用简化的 FDK)
    # 这里我们使用 ground truth volume 作为参考
    volume = data['image']  # (D, H, W)

    # 归一化到 [0, 1]
    vol_min, vol_max = volume.min(), volume.max()
    volume_norm = (volume - vol_min) / (vol_max - vol_min + 1e-8)

    # 找到有效体素 (密度 > 阈值)
    valid_mask = volume_norm > density_thresh
    valid_indices = np.argwhere(valid_mask)

    print(f"    有效体素数: {len(valid_indices)}")

    if len(valid_indices) < n_points:
        print(f"    警告: 有效体素不足，使用全部 {len(valid_indices)} 个")
        sample_indices = valid_indices
    else:
        # 随机采样
        rng = np.random.default_rng(seed=42)  # 固定种子保证可复现
        sample_idx = rng.choice(len(valid_indices), size=n_points, replace=False)
        sample_indices = valid_indices[sample_idx]

    # 转换为归一化坐标 [-1, 1]
    D, H, W = volume.shape
    coords = sample_indices.astype(np.float32)
    coords[:, 0] = (coords[:, 0] / (D - 1)) * 2 - 1  # z
    coords[:, 1] = (coords[:, 1] / (H - 1)) * 2 - 1  # y
    coords[:, 2] = (coords[:, 2] / (W - 1)) * 2 - 1  # x

    # 获取密度值
    densities = volume_norm[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]]

    # 组合成 (N, 4) 数组: [x, y, z, density]
    points = np.zeros((len(sample_indices), 4), dtype=np.float32)
    points[:, 0] = coords[:, 2]  # x
    points[:, 1] = coords[:, 1]  # y
    points[:, 2] = coords[:, 0]  # z
    points[:, 3] = densities

    return points


def main():
    print("=" * 60)
    print("重新生成 data/369 数据集")
    print("=" * 60)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for organ in ORGANS:
        input_path = INPUT_DIR / f"{organ}_50.pickle"

        if not input_path.exists():
            print(f"\n⚠️ 跳过 {organ}: 文件不存在")
            continue

        for train_num, data, pickle_path in generate_sparse_views(input_path, OUTPUT_DIR, TRAIN_NUMS):
            # 生成初始化点云
            init_name = f"init_{organ}_50_{train_num}views.npy"
            init_path = OUTPUT_DIR / init_name

            print(f"    生成初始化点云: {init_name}")
            points = generate_init_pointcloud_random(data, N_POINTS, DENSITY_THRESH)
            np.save(init_path, points)

            print(f"    ✅ 点云形状: {points.shape}, 密度均值: {points[:, 3].mean():.4f}")

    print("\n" + "=" * 60)
    print("完成! 新数据保存在:", OUTPUT_DIR)
    print("=" * 60)

    # 与旧数据对比
    print("\n=== 新旧数据对比 ===")
    old_dir = Path("/home/qyhu/Documents/r2_ours/r2_gaussian/data/369")

    for organ in ["foot", "chest", "abdomen"]:
        for views in [3, 6, 9]:
            old_init = old_dir / f"init_{organ}_50_{views}views.npy"
            new_init = OUTPUT_DIR / f"init_{organ}_50_{views}views.npy"

            if old_init.exists() and new_init.exists():
                old_pts = np.load(old_init)
                new_pts = np.load(new_init)
                print(f"{organ}_{views}views: 旧密度={old_pts[:, 3].mean():.4f}, 新密度={new_pts[:, 3].mean():.4f}")


if __name__ == "__main__":
    main()
