"""
Binocular Stereo Consistency Utilities for R²-Gaussian

论文来源: Binocular-Guided 3D Gaussian Splatting with View Consistency
          for Sparse View Synthesis (NeurIPS 2024)

适配说明:
- R²-Gaussian 使用 X-ray CT 投影（平行束/锥束），与原论文的透视相机不同
- CT 数据是 360° 旋转扫描，而非前向场景
- 本实现针对医学 CT 数据进行了特定优化

核心组件:
1. inverse_warp_images: 使用视差进行图像 warp
2. create_shifted_camera: 生成平移后的虚拟相机
3. compute_disparity: 从深度图计算视差
4. BinocularConsistencyLoss: 整合上述组件的完整损失

注意: 已移除 SmoothLoss（边缘感知平滑损失），统一使用纯图像一致性约束

作者: Claude Code Agent
日期: 2025-11-25
更新: 2025-12-03 - 移除 SmoothLoss，简化为纯图像一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math


def inverse_warp_images(
    source_image: torch.Tensor,
    disparity: torch.Tensor,
    direction: str = "left_to_right"
) -> torch.Tensor:
    """
    使用视差进行图像 warp

    将源图像根据视差 warp 到目标视角。使用双线性插值确保可微分。

    Args:
        source_image: 源图像 [B, C, H, W]
        disparity: 视差图 [B, 1, H, W]，表示水平像素偏移
        direction: warp 方向
            - "left_to_right": 从左图 warp 到右图视角
            - "right_to_left": 从右图 warp 到左图视角

    Returns:
        warped_image: warp 后的图像 [B, C, H, W]
    """
    B, C, H, W = source_image.shape
    device = source_image.device

    # 创建采样网格
    # meshgrid 生成归一化坐标 [-1, 1]
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # 扩展到 batch 维度
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

    # 将视差转换为归一化坐标偏移
    # disparity 单位是像素，需要转换为 [-1, 1] 范围
    disparity_normalized = disparity.squeeze(1) * 2.0 / W

    # 根据方向应用视差
    if direction == "right_to_left":
        # 从右图 warp 到左图: x' = x + d
        x_warped = x_grid + disparity_normalized
    else:  # left_to_right
        # 从左图 warp 到右图: x' = x - d
        x_warped = x_grid - disparity_normalized

    # 组合采样网格 [B, H, W, 2]
    grid = torch.stack([x_warped, y_grid], dim=-1)

    # 使用 grid_sample 进行双线性插值
    # padding_mode='zeros' 会将越界区域填充为0
    warped_image = F.grid_sample(
        source_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped_image


def create_shifted_camera_ct(
    viewpoint_camera,
    trans_dist: float,
    scanner_cfg: dict
) -> dict:
    """
    为 CT 数据创建平移后的虚拟相机

    与原论文的透视相机不同，CT 使用平行束或锥束投影。
    对于 CT，我们通过小角度旋转来模拟"平移"效果。

    Args:
        viewpoint_camera: 原始相机对象
        trans_dist: 平移距离（对于CT转换为角度偏移）
        scanner_cfg: CT 扫描器配置

    Returns:
        shifted_camera_info: 包含平移相机信息的字典
    """
    # 对于 CT 数据，trans_dist 被解释为角度偏移（弧度）
    # 典型值: 0.02-0.1 rad ≈ 1-6°
    angle_offset = trans_dist

    # 获取原始相机角度
    original_angle = viewpoint_camera.angle

    # 计算新角度
    new_angle = original_angle + angle_offset

    # 创建新的旋转矩阵 (绕 Y 轴旋转，假设 CT 是绕垂直轴旋转)
    cos_a = math.cos(new_angle)
    sin_a = math.sin(new_angle)

    # 旋转矩阵 (绕 Y 轴)
    R_new = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float32)

    # 计算新的平移向量 (相机位置)
    # 假设相机在以原点为中心的圆上
    camera_distance = np.linalg.norm(viewpoint_camera.T)
    T_new = np.array([
        camera_distance * sin_a,
        viewpoint_camera.T[1],  # 保持 Y 坐标不变
        -camera_distance * cos_a
    ], dtype=np.float32)

    return {
        'R': R_new,
        'T': T_new,
        'angle': new_angle,
        'angle_offset': angle_offset
    }


def compute_disparity_from_depth(
    depth: torch.Tensor,
    focal_length: float,
    baseline: float
) -> torch.Tensor:
    """
    从深度图计算视差

    基于双目立体几何: d = f * B / Z
    其中 d 是视差，f 是焦距，B 是基线，Z 是深度

    Args:
        depth: 深度图 [B, 1, H, W] 或 [H, W]
        focal_length: 焦距（像素单位）
        baseline: 基线距离（与深度单位相同）

    Returns:
        disparity: 视差图，与输入相同形状
    """
    # 避免除零
    depth_safe = depth.clamp(min=1e-5)
    disparity = focal_length * baseline / depth_safe
    return disparity


def estimate_depth_from_gaussians(
    gaussians,
    viewpoint_camera,
    pipe,
    render_func
) -> torch.Tensor:
    """
    从高斯点云估计深度图

    由于 R²-Gaussian 的 rasterizer 不直接输出深度，
    我们使用加权平均的方式从渲染结果估计深度。

    方法: 对于每个像素，计算贡献该像素的高斯点的深度加权平均

    Args:
        gaussians: GaussianModel 实例
        viewpoint_camera: 相机视角
        pipe: 渲染管线参数
        render_func: 渲染函数

    Returns:
        depth: 估计的深度图 [1, H, W]
    """
    # 获取高斯点的3D位置
    points_3d = gaussians.get_xyz  # [N, 3]

    # 获取相机参数
    world_view_transform = viewpoint_camera.world_view_transform  # [4, 4]

    # 将点变换到相机坐标系
    ones = torch.ones(points_3d.shape[0], 1, device=points_3d.device)
    points_homo = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    points_cam = (world_view_transform @ points_homo.T).T  # [N, 4]

    # 提取深度（Z 坐标）
    depths = points_cam[:, 2]  # [N]

    # 使用密度作为权重
    densities = gaussians.get_density.squeeze()  # [N]

    # 简单估计：返回平均深度
    # 更精确的方法需要修改 rasterizer，这里使用简化版本
    mean_depth = (depths * densities).sum() / (densities.sum() + 1e-5)

    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width

    # 创建均匀深度图（简化版本）
    depth_map = torch.full((1, H, W), mean_depth.item(), device=points_3d.device)

    return depth_map


class BinocularConsistencyLoss(nn.Module):
    """
    双目立体一致性损失（简化版）

    核心思想:
    1. 对训练视角进行小角度旋转，生成虚拟双目视角对
    2. 从深度估计视差
    3. 使用视差 warp 图像
    4. 计算 warp 后图像与原图的一致性损失（仅 L1，无边缘感知平滑）

    针对医学 CT 的优化:
    - 使用角度偏移代替线性平移（适应 360° 旋转扫描）
    - 支持平行束和锥束投影

    更新历史:
    - 2025-12-03: 移除 SmoothLoss，简化为纯图像一致性约束
    """

    def __init__(
        self,
        max_angle_offset: float = 0.1,  # 最大角度偏移（弧度），约 5.7°
        start_iteration: int = 10000,   # 开始应用损失的迭代数（CT 可以更早）
        warmup_iterations: int = 2000,  # warmup 迭代数
    ):
        """
        Args:
            max_angle_offset: 最大角度偏移，对于 CT 建议 0.05-0.15 rad
            start_iteration: 开始应用损失的迭代数
            warmup_iterations: 损失权重 warmup 的迭代数
        """
        super(BinocularConsistencyLoss, self).__init__()

        self.max_angle_offset = max_angle_offset
        self.start_iteration = start_iteration
        self.warmup_iterations = warmup_iterations

    def get_loss_weight(self, iteration: int) -> float:
        """
        获取当前迭代的损失权重（支持 warmup）
        """
        if iteration < self.start_iteration:
            return 0.0

        elapsed = iteration - self.start_iteration
        if elapsed < self.warmup_iterations:
            # 线性 warmup
            return elapsed / self.warmup_iterations

        return 1.0

    def forward(
        self,
        rendered_image: torch.Tensor,
        gt_image: torch.Tensor,
        shifted_rendered_image: torch.Tensor,
        depth_map: torch.Tensor,
        focal_length: float,
        baseline: float,
        iteration: int
    ) -> Dict[str, torch.Tensor]:
        """
        计算双目一致性损失

        Args:
            rendered_image: 原视角渲染图像 [C, H, W]
            gt_image: 原视角 GT 图像 [C, H, W]
            shifted_rendered_image: 平移视角渲染图像 [C, H, W]
            depth_map: 深度图 [1, H, W]
            focal_length: 焦距（像素单位）
            baseline: 基线距离（角度偏移转换后的等效基线）
            iteration: 当前迭代数

        Returns:
            损失字典，包含 'consistency', 'total'
        """
        weight = self.get_loss_weight(iteration)

        if weight == 0.0:
            zero = torch.tensor(0.0, device=rendered_image.device)
            return {
                'consistency': zero,
                'total': zero
            }

        # 确保维度正确
        if rendered_image.dim() == 3:
            rendered_image = rendered_image.unsqueeze(0)
        if gt_image.dim() == 3:
            gt_image = gt_image.unsqueeze(0)
        if shifted_rendered_image.dim() == 3:
            shifted_rendered_image = shifted_rendered_image.unsqueeze(0)
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0).unsqueeze(0)
        elif depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(0)

        # 计算视差
        disparity = compute_disparity_from_depth(depth_map, focal_length, baseline)

        # Warp shifted image 到原视角
        warped_image = inverse_warp_images(
            shifted_rendered_image,
            disparity,
            direction="right_to_left"
        )

        # 创建有效区域掩码（排除 warp 后的边界无效区域）
        ones_mask = torch.ones_like(shifted_rendered_image[:, :1, :, :])
        warped_mask = inverse_warp_images(
            ones_mask,
            disparity,
            direction="right_to_left"
        )
        valid_mask = (warped_mask > 0.5).float()

        # 一致性损失 (L1)
        consistency_loss = F.l1_loss(
            warped_image * valid_mask,
            gt_image * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-5)

        # 应用权重
        total_loss = consistency_loss * weight

        return {
            'consistency': total_loss,
            'total': total_loss
        }


def get_random_angle_offset(max_offset: float) -> float:
    """
    获取随机角度偏移

    Args:
        max_offset: 最大偏移（弧度）

    Returns:
        随机偏移值，在 [-max_offset, max_offset] 范围内
    """
    import random
    return (random.random() * 2 - 1) * max_offset


# CT 特定的深度估计函数
def estimate_depth_for_ct(
    gaussians,
    viewpoint_camera,
    method: str = "weighted_average"
) -> torch.Tensor:
    """
    为 CT 数据估计深度图

    由于 CT 使用 X-ray 投影，传统的深度概念需要重新定义。
    对于 CT，我们估计的是沿射线方向的"有效深度"。

    Args:
        gaussians: GaussianModel 实例
        viewpoint_camera: 相机视角
        method: 深度估计方法
            - "weighted_average": 密度加权平均深度
            - "max_density": 最大密度位置的深度
            - "first_surface": 第一个表面的深度（阈值法）

    Returns:
        depth: 深度图 [1, H, W]
    """
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    device = gaussians.get_xyz.device

    # 获取高斯点位置和密度
    points = gaussians.get_xyz  # [N, 3]
    densities = gaussians.get_density.squeeze()  # [N]

    # 变换到相机坐标系
    world_view_transform = viewpoint_camera.world_view_transform
    ones = torch.ones(points.shape[0], 1, device=device)
    points_homo = torch.cat([points, ones], dim=1)
    points_cam = (world_view_transform @ points_homo.T).T[:, :3]

    # 提取深度（负 Z 方向通常指向相机前方）
    depths = points_cam[:, 2]

    if method == "weighted_average":
        # 全局加权平均（简化版本）
        valid_mask = densities > 1e-3
        if valid_mask.sum() > 0:
            mean_depth = (depths[valid_mask] * densities[valid_mask]).sum() / densities[valid_mask].sum()
        else:
            mean_depth = depths.mean()
        depth_map = torch.full((1, H, W), mean_depth.item(), device=device)

    elif method == "max_density":
        # 最大密度位置
        max_idx = densities.argmax()
        depth_at_max = depths[max_idx]
        depth_map = torch.full((1, H, W), depth_at_max.item(), device=device)

    else:  # first_surface
        # 第一个表面（深度最小的高密度点）
        threshold = densities.mean() * 0.5
        surface_mask = densities > threshold
        if surface_mask.sum() > 0:
            surface_depth = depths[surface_mask].min()
        else:
            surface_depth = depths.min()
        depth_map = torch.full((1, H, W), surface_depth.item(), device=device)

    return depth_map
