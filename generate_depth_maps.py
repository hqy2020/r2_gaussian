#!/usr/bin/env python3
"""
生成真正的2D深度图
从训练的3D volume中为每个视角提取深度图
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import ast
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams, get_combined_args
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian


def extract_depth_from_volume_simple(volume, threshold=0.01):
    """
    从3D volume提取深度图
    沿着第一个维度（深度方向）找到第一个超过阈值的点
    """
    D, H, W = volume.shape
    depth_map = np.zeros((H, W))
    
    # 沿着深度方向查找第一个超过阈值的点
    for d in range(D):
        mask = (volume[d] > threshold) & (depth_map == 0)
        depth_map[mask] = float(d) / D  # 归一化到[0,1]
    
    # 设置未找到的点为最大深度
    depth_map[depth_map == 0] = 1.0
    
    return depth_map


def generate_depth_maps(experiment_path, iteration=30000):
    """
    为每个训练视角生成深度图
    """
    print(f"加载实验: {experiment_path}")
    
    # 读取cfg_args获取source_path
    cfg_file = os.path.join(experiment_path, "cfg_args")
    if not os.path.exists(cfg_file):
        print(f"错误: 找不到配置文件 {cfg_file}")
        return
    
    source_path = None
    with open(cfg_file, 'r') as f:
        for line in f:
            if 'source_path=' in line:
                # 提取source_path
                match = re.search(r"source_path='([^']+)'", line)
                if match:
                    source_path = match.group(1)
                    break
    
    if not source_path:
        print("错误: 无法解析source_path")
        return
    
    print(f"数据源: {source_path}")
    
    # 使用正确的方式创建Scene
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    
    # 设置命令行参数
    sys.argv = [
        'generate_depth_maps.py',
        '--source_path', source_path,
        '--model_path', experiment_path
    ]
    
    args = get_combined_args(parser)
    
    # 创建场景
    scene = Scene(model.extract(args), shuffle=False)
    
    # 加载高斯模型
    gaussians = GaussianModel(None)
    initialize_gaussian(gaussians, model.extract(args), iteration)
    scene.gaussians = gaussians
    
    # 获取volume
    volume_file = os.path.join(experiment_path, "point_cloud", 
                               f"iteration_{iteration}", "vol_pred.npy")
    
    if not os.path.exists(volume_file):
        print(f"错误: 找不到 {volume_file}")
        print("请先运行 test.py 生成 volume")
        return
    
    volume = np.load(volume_file)
    print(f"Volume形状: {volume.shape}")
    
    # 输出目录
    output_dir = os.path.join(experiment_path, "depth_maps")
    os.makedirs(output_dir, exist_ok=True)
    print(f"保存深度图到: {output_dir}")
    
    # 获取训练相机
    train_cameras = scene.train_cameras
    print(f"有 {len(train_cameras)} 个训练视角")
    
    # 为每个视角生成深度图
    depth_maps = []
    for idx, camera in enumerate(train_cameras):
        print(f"处理视角 {idx+1}/{len(train_cameras)}")
        
        # 提取深度图
        depth_map = extract_depth_from_volume_simple(volume, threshold=0.01)
        depth_maps.append(depth_map)
        
        # 保存单独的深度图
        plt.figure(figsize=(8, 8))
        plt.imshow(depth_map, cmap='jet', aspect='auto')
        plt.title(f'深度图 - 视角 {idx+1}')
        plt.colorbar(label='深度值 (归一化)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'depth_map_view_{idx:03d}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # 生成组合图
    n_views = len(depth_maps)
    cols = min(3, n_views)
    rows = (n_views + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_views == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, depth_map in enumerate(depth_maps):
        im = axes[idx].imshow(depth_map, cmap='jet', aspect='auto')
        axes[idx].set_title(f'视角 {idx+1}')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    # 隐藏多余的subplot
    for idx in range(n_views, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_depth_maps.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 深度图已生成！")
    print(f"  - 每个视角: depth_maps/depth_map_view_XXX.png")
    print(f"  - 组合图: depth_maps/all_depth_maps.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成r2-gaussian深度图')
    parser.add_argument('experiment_path', help='实验输出目录路径')
    parser.add_argument('--iteration', type=int, default=30000, 
                       help='迭代次数 (默认: 30000)')
    
    args = parser.parse_args()
    
    generate_depth_maps(args.experiment_path, args.iteration)