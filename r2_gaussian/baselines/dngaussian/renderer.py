#
# DNGaussian X-ray 渲染器
#
# 基于 X-Gaussian 渲染器，添加深度渲染支持
#

import torch
import math
from typing import Dict

from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

from .model import DNGaussianModel


def render_dngaussian(
    viewpoint_camera,
    pc: DNGaussianModel,
    pipe,
    bg_color: torch.Tensor = None,
    scaling_modifier: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    DNGaussian X-ray 渲染函数

    Args:
        viewpoint_camera: 相机视角
        pc: DNGaussianModel 实例
        pipe: PipelineParams
        bg_color: 背景颜色 (可选)
        scaling_modifier: 尺度修正因子

    Returns:
        dict:
            - render: 渲染的 X-ray 投影 [1, H, W]
            - viewspace_points: 屏幕空间点 (用于梯度)
            - visibility_filter: 可见性掩码
            - radii: 2D 半径
    """
    # 创建屏幕空间点张量
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置相机参数
    mode = viewpoint_camera.mode
    if mode == 0:
        # 平行投影
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        # 透视投影
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError(f"Unsupported camera mode: {mode}")

    # 光栅化设置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug if hasattr(pipe, 'debug') else False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取高斯参数
    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density  # 使用调制后的密度

    # 处理协方差
    scales = None
    rotations = None
    cov3D_precomp = None
    if hasattr(pipe, 'compute_cov3D_python') and pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 渲染
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def render_dngaussian_depth(
    viewpoint_camera,
    pc: DNGaussianModel,
    pipe,
    scaling_modifier: float = 1.0,
    use_base_density: bool = False,
) -> torch.Tensor:
    """
    渲染深度图（用于深度正则化）

    通过计算 Gaussian 中心到相机的距离加权平均来估计深度

    Args:
        viewpoint_camera: 相机视角
        pc: DNGaussianModel 实例
        pipe: PipelineParams
        scaling_modifier: 尺度修正因子
        use_base_density: 是否使用基础密度（不经过 neural renderer）

    Returns:
        [1, H, W] 深度图
    """
    # 获取相机位置
    camera_pos = viewpoint_camera.camera_center

    # 计算每个 Gaussian 到相机的深度
    positions = pc.get_xyz
    depths = torch.norm(positions - camera_pos, dim=1, keepdim=True)  # [N, 1]

    # 获取密度作为权重
    if use_base_density:
        weights = pc.get_base_density
    else:
        weights = pc.get_density

    # 使用渲染器渲染深度加权图
    # 方法：将深度值作为"颜色"进行渲染
    # 这里我们使用一个技巧：渲染深度*density 和 density，然后相除

    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda"
    )

    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError(f"Unsupported camera mode: {mode}")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # 渲染 density * depth
    depth_weighted_density = weights * depths
    rendered_depth_weighted, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=depth_weighted_density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # 渲染 density
    rendered_density, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=weights,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # 计算深度：depth = (density * depth) / density
    eps = 1e-6
    depth_map = rendered_depth_weighted / (rendered_density + eps)

    # 归一化到 [0, 1]
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > eps:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

    return depth_map


def query_dngaussian(
    pc: DNGaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe,
    scaling_modifier=1.0,
) -> Dict[str, torch.Tensor]:
    """
    DNGaussian 体积查询函数

    Args:
        pc: DNGaussianModel 实例
        center: 体积中心
        nVoxel: 体素数量 [nx, ny, nz]
        sVoxel: 体素大小 [sx, sy, sz]
        pipe: PipelineParams
        scaling_modifier: 尺度修正因子

    Returns:
        dict:
            - vol: [nx, ny, nz] 体积
            - radii: 半径
    """
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug if hasattr(pipe, 'debug') else False,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density  # 使用调制后的密度

    scales = None
    rotations = None
    cov3D_precomp = None
    if hasattr(pipe, 'compute_cov3D_python') and pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }
