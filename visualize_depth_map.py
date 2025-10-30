#!/usr/bin/env python3
"""
可视化 r2-gaussian 深度图
直接读取 vol_pred.npy 并可视化
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def visualize_volume_slices(experiment_path, iteration=30000):
    """
    直接读取并可视化 vol_pred.npy 的切片
    """
    print(f"加载实验: {experiment_path}")
    
    # 加载vol_pred.npy
    vol_file = os.path.join(experiment_path, "point_cloud", f"iteration_{iteration}", "vol_pred.npy")
    vol_gt_file = os.path.join(experiment_path, "point_cloud", f"iteration_{iteration}", "vol_gt.npy")
    
    if not os.path.exists(vol_file):
        print(f"错误: 找不到 {vol_file}")
        return
    
    # 读取volume
    print(f"读取 volume 数据...")
    vol_pred = np.load(vol_file)
    print(f"Volume形状: {vol_pred.shape}")
    
    # 输出目录
    output_dir = os.path.join(experiment_path, "depth_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"保存可视化结果到: {output_dir}")
    
    # 可视化多个切片
    D, H, W = vol_pred.shape
    slices_to_save = [D//4, D//2, 3*D//4]  # 选几个代表性的切片
    
    print(f"\n生成 {len(slices_to_save)} 个切片的可视化...")
    
    # 创建子图
    fig, axes = plt.subplots(1, len(slices_to_save), figsize=(5*len(slices_to_save), 5))
    if len(slices_to_save) == 1:
        axes = [axes]
    
    for i, slice_idx in enumerate(slices_to_save):
        im = axes[i].imshow(vol_pred[slice_idx], cmap='viridis', aspect='auto')
        axes[i].set_title(f'Slice {slice_idx}/{D}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_slices.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存中间切片
    mid_slice = D // 2
    plt.figure(figsize=(10, 10))
    plt.imshow(vol_pred[mid_slice], cmap='viridis')
    plt.title(f'中间切片 (slice {mid_slice}/{D})')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'slice_{mid_slice}.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 如果有GT，也可视化GT
    if os.path.exists(vol_gt_file):
        print("发现 vol_gt.npy，生成对比图...")
        vol_gt = np.load(vol_gt_file)
        
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(vol_gt[mid_slice], cmap='viridis')
        plt.title('Ground Truth')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(vol_pred[mid_slice], cmap='viridis')
        plt.title('Predicted')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        diff = np.abs(vol_gt[mid_slice] - vol_pred[mid_slice])
        plt.imshow(diff, cmap='hot')
        plt.title('Difference')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_slice.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\n✓ 可视化完成！")
    print(f"  - 结果保存在: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化r2-gaussian深度图')
    parser.add_argument('experiment_path', help='实验输出目录路径')
    parser.add_argument('--iteration', type=int, default=30000, 
                       help='迭代次数 (默认: 30000)')
    
    args = parser.parse_args()
    
    visualize_volume_slices(args.experiment_path, args.iteration)