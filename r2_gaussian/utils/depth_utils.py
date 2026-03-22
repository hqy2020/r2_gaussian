#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import numpy as np


def extract_depth_from_volume(density_volume, camera_params, threshold=0.01):
    """
    从 3D density volume 中提取深度图
    
    Args:
        density_volume: 3D tensor of shape (D, H, W) - density volume
        camera_params: 相机参数，包含视角信息
        threshold: 密度阈值，用于确定深度
    
    Returns:
        depth_map: 2D tensor of shape (H, W) - 深度图
    """
    device = density_volume.device
    D, H, W = density_volume.shape
    
    # 创建深度图
    depth_map = torch.zeros(H, W, device=device)
    
    # 沿着深度方向查找第一个超过阈值的点
    for h in range(H):
        for w in range(W):
            # 沿着深度方向搜索
            for d in range(D):
                if density_volume[d, h, w] > threshold:
                    depth_map[h, w] = float(d) / D  # 归一化深度
                    break
            else:
                # 如果没有找到超过阈值的点，设置为最大深度
                depth_map[h, w] = 1.0
    
    return depth_map


def extract_depth_from_volume_ray_casting(density_volume, camera_params, threshold=0.01):
    """
    使用 ray casting 方法从 3D density volume 中提取深度图
    
    Args:
        density_volume: 3D tensor of shape (D, H, W) - density volume
        camera_params: 相机参数，包含视角信息
        threshold: 密度阈值，用于确定深度
    
    Returns:
        depth_map: 2D tensor of shape (H, W) - 深度图
    """
    device = density_volume.device
    D, H, W = density_volume.shape
    
    # 创建深度图
    depth_map = torch.zeros(H, W, device=device)
    
    # 沿着深度方向查找第一个超过阈值的点
    # 使用向量化操作提高效率
    for d in range(D):
        mask = (density_volume[d] > threshold) & (depth_map == 0)
        depth_map[mask] = float(d) / D
    
    # 对于没有找到超过阈值的点，设置为最大深度
    depth_map[depth_map == 0] = 1.0
    
    return depth_map


def extract_depth_from_gaussians(gaussians, camera_params, resolution=(256, 256)):
    """
    直接从 Gaussians 计算深度图（不经过 voxelization）
    
    Args:
        gaussians: GaussianModel 对象
        camera_params: 相机参数
        resolution: 输出分辨率 (H, W)
    
    Returns:
        depth_map: 2D tensor of shape (H, W) - 深度图
    """
    device = gaussians.get_xyz.device
    H, W = resolution
    
    # 获取高斯参数
    means3D = gaussians.get_xyz  # (N, 3)
    scales = gaussians.get_scaling  # (N, 3)
    rotations = gaussians.get_rotation  # (N, 4)
    
    # 创建像素坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    
    # 初始化深度图
    depth_map = torch.zeros(H, W, device=device)
    
    # 简化的深度计算：使用高斯中心点的深度
    # 这里可以根据需要实现更复杂的深度计算
    for i in range(means3D.shape[0]):
        # 获取高斯中心点的深度
        gaussian_depth = means3D[i, 2]  # z坐标作为深度
        
        # 这里需要根据相机参数将3D点投影到2D
        # 简化实现：假设相机在原点看向z轴正方向
        if gaussian_depth > 0:  # 只考虑正深度
            # 简化的投影（实际应该使用相机内参和外参）
            proj_x = int(means3D[i, 0] * W / 2 + W / 2)
            proj_y = int(means3D[i, 1] * H / 2 + H / 2)
            
            if 0 <= proj_x < W and 0 <= proj_y < H:
                depth_map[proj_y, proj_x] = gaussian_depth
    
    return depth_map


def compute_depth_loss(rendered_depth, gt_depth, loss_type='l1'):
    """
    计算深度损失
    
    Args:
        rendered_depth: 渲染的深度图
        gt_depth: 真实深度图
        loss_type: 损失类型 ('l1', 'l2', 'pearson')
    
    Returns:
        loss: 深度损失值
    """
    if loss_type == 'l1':
        return F.l1_loss(rendered_depth, gt_depth)
    elif loss_type == 'l2':
        return F.mse_loss(rendered_depth, gt_depth)
    elif loss_type == 'pearson':
        # 使用皮尔逊相关系数作为损失
        from .loss_utils import pearson_corrcoef
        correlation = pearson_corrcoef(rendered_depth.flatten(), gt_depth.flatten())
        return 1.0 - correlation  # 转换为损失（越小越好）
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def depth_consistency_loss(depth_maps, camera_params):
    """
    计算多视角深度一致性损失
    
    Args:
        depth_maps: 多个视角的深度图列表
        camera_params: 相机参数列表
    
    Returns:
        consistency_loss: 一致性损失
    """
    if len(depth_maps) < 2:
        return torch.tensor(0.0, device=depth_maps[0].device)
    
    # 简化的实现：计算相邻视角深度图的差异
    total_loss = 0.0
    for i in range(len(depth_maps) - 1):
        diff = torch.abs(depth_maps[i] - depth_maps[i + 1])
        total_loss += diff.mean()
    
    return total_loss / (len(depth_maps) - 1)






