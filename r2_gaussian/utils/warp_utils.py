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

def inverse_warp(source_image, source_depth, target_depth, 
                 source_extrinsic, target_extrinsic, 
                 source_intrinsic):
    """
    使用逆变形将源视图变形到目标视图 - 参考IPSM的warp.py实现
    
    Args:
        source_image: (C, H, W) 源图像
        source_depth: (H, W) 或 (1, H, W) 源深度图
        target_depth: (H, W) 或 (1, H, W) 目标深度图
        source_extrinsic: (4, 4) 源相机外参矩阵 (world_view_transform)
        target_extrinsic: (4, 4) 目标相机外参矩阵 (world_view_transform)
        source_intrinsic: (3, 3) 源相机内参矩阵
    
    Returns:
        dict containing:
            warped_img: 变形后的图像 (C, H, W)
            mask_warp: warp掩码 (H, W)
            mask_depth_strict: 深度一致性掩码 (H, W)
    """
    device = source_image.device
    C, H_source, W_source = source_image.shape
    
    # 确保深度图形状一致
    if source_depth.dim() == 3:
        source_depth = source_depth.squeeze(0)  # (H, W)
    if target_depth.dim() == 3:
        target_depth = target_depth.squeeze(0)  # (H, W)
    
    # 获取深度图尺寸
    H_source_depth, W_source_depth = source_depth.shape
    H_target_depth, W_target_depth = target_depth.shape
    
    # 确保source_depth与source_image尺寸匹配
    if H_source_depth != H_source or W_source_depth != W_source:
        # 需要resize source_depth到source_image尺寸（使用文件顶部已导入的F）
        source_depth = F.interpolate(
            source_depth.unsqueeze(0).unsqueeze(0),
            size=(H_source, W_source),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        H_source_depth, W_source_depth = H_source, W_source
    
    # 使用target_depth的尺寸创建像素坐标网格（因为我们要变形到target视角）
    H, W = H_target_depth, W_target_depth
    
    # 创建像素坐标网格 (u, v)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 将像素坐标转换为齐次坐标 (H*W, 3)
    pixel_coords = torch.stack([
        x_coords.flatten(),
        y_coords.flatten(),
        torch.ones(H * W, device=device)
    ], dim=0)  # (3, H*W)
    
    # 使用目标深度反投影到3D点
    depth_flat = target_depth.flatten()  # (H*W,)
    
    # 反投影到目标相机坐标系（在CPU上计算inverse和矩阵乘法避免CUBLAS错误）
    inv_K = torch.inverse(source_intrinsic.cpu())
    pixel_coords_cpu = pixel_coords.cpu()
    cam_coords = (inv_K @ pixel_coords_cpu).to(device)  # (3, H*W)
    cam_coords = cam_coords * depth_flat.unsqueeze(0)  # (3, H*W)
    
    # 转换到世界坐标
    cam_coords_homo = torch.cat([
        cam_coords,
        torch.ones(1, H*W, device=device)
    ], dim=0)  # (4, H*W)
    
    # 世界坐标 = target_extrinsic^-1 @ cam_coords（在CPU上计算inverse和矩阵乘法避免CUBLAS错误）
    target_extrinsic_inv = torch.inverse(target_extrinsic.cpu())
    cam_coords_homo_cpu = cam_coords_homo.cpu()
    world_coords = (target_extrinsic_inv @ cam_coords_homo_cpu).to(device)  # (4, H*W)
    
    # 源相机坐标 = source_extrinsic @ world_coords（在CPU上计算矩阵乘法避免CUBLAS错误）
    source_extrinsic_cpu = source_extrinsic.cpu()
    world_coords_cpu = world_coords.cpu()
    source_cam_coords_homo = (source_extrinsic_cpu @ world_coords_cpu).to(device)  # (4, H*W)
    source_cam_coords = source_cam_coords_homo[:3, :]  # (3, H*W)
    
    # 使用源深度验证
    source_depth_expected = source_cam_coords[2, :]  # (H*W,)
    
    # 投影到源图像平面（在CPU上计算矩阵乘法避免CUBLAS错误）
    source_intrinsic_cpu = source_intrinsic.cpu()
    source_cam_coords_cpu = source_cam_coords.cpu()
    source_proj = (source_intrinsic_cpu @ source_cam_coords_cpu).to(device)  # (3, H*W)
    source_proj = source_proj[:2, :] / (source_proj[2:3, :] + 1e-7)  # (2, H*W)
    
    # 归一化到[-1, 1]用于grid_sample（注意：这里W和H是target的尺寸）
    source_x = 2.0 * source_proj[0, :] / W_source - 1.0  # (H*W,)
    source_y = 2.0 * source_proj[1, :] / H_source - 1.0  # (H*W,)
    
    grid = torch.stack([source_x, source_y], dim=-1).reshape(H, W, 2)  # (H, W, 2)
    
    # 采样源图像（注意：source_image是(H_source, W_source)，grid是(H, W)）
    source_image_expanded = source_image.unsqueeze(0)  # (1, C, H_source, W_source)
    warped_img = F.grid_sample(
        source_image_expanded,
        grid.unsqueeze(0),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    ).squeeze(0)  # (C, H, W) - 输出尺寸是target的尺寸
    
    # 创建warp掩码（检查是否在图像边界内）
    valid_x = (source_proj[0, :] >= 0) & (source_proj[0, :] < W_source)
    valid_y = (source_proj[1, :] >= 0) & (source_proj[1, :] < H_source)
    mask_warp = (valid_x & valid_y).reshape(H, W).float()  # (H, W)
    
    # 深度一致性掩码（深度差异小于阈值）
    # 注意：需要将source_depth resize到target的尺寸进行比较
    if H_source_depth != H or W_source_depth != W:
        source_depth_resized = F.interpolate(
            source_depth.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
    else:
        source_depth_resized = source_depth
    
    source_depth_expected_2d = source_depth_expected.reshape(H, W)
    depth_diff = torch.abs(source_depth_expected_2d - source_depth_resized)
    depth_threshold = 0.1 * torch.clamp(source_depth_resized, min=1e-3)  # 10%深度差异
    mask_depth_strict = (depth_diff < depth_threshold).float()  # (H, W)
    
    return {
        'warped_img': warped_img,
        'mask_warp': mask_warp,
        'mask_depth_strict': mask_depth_strict
    }

