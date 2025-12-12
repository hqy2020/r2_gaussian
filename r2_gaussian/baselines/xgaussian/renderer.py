#
# X-Gaussian renderer adapted for r2_gaussian framework
#
# 复用 R²-Gaussian 的 xray-gaussian-rasterization 进行 X-ray 投影
#

import torch
import math
from typing import Dict

from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from .model import XGaussianModel
from ..registry import BaseRenderer


def render_xgaussian(
    viewpoint_camera,
    pc: XGaussianModel,
    pipe,
    bg_color: torch.Tensor = None,
    scaling_modifier: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    X-Gaussian 渲染函数

    使用 R²-Gaussian 的 xray-gaussian-rasterization 进行 X-ray 投影。
    X-Gaussian 的 opacity 在这里作为密度进行投影。

    Args:
        viewpoint_camera: 相机视角
        pc: XGaussianModel 实例
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

    # 设置相机参数（与 R²-Gaussian 保持一致）
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

    # 光栅化设置（与 R²-Gaussian render() 一致）
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

    # 使用 opacity 作为密度（get_density 返回 get_opacity）
    density = pc.get_density

    # 处理协方差
    scales = None
    rotations = None
    cov3D_precomp = None
    if hasattr(pipe, 'compute_cov3D_python') and pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 渲染（与 R²-Gaussian 保持一致，参数名为 opacities）
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


class XGaussianRenderer(BaseRenderer):
    """X-Gaussian 渲染器类"""

    def __init__(self, pipe=None):
        self.pipe = pipe

    def render(
        self,
        viewpoint,
        model: XGaussianModel,
        pipe=None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """渲染接口实现"""
        if pipe is None:
            pipe = self.pipe
        return render_xgaussian(viewpoint, model, pipe, **kwargs)


def query_xgaussian(
    pc: XGaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe,
    scaling_modifier=1.0,
) -> Dict[str, torch.Tensor]:
    """
    X-Gaussian 体积查询函数

    直接复用 R²-Gaussian 的 GaussianVoxelizer
    """
    from xray_gaussian_rasterization_voxelization import (
        GaussianVoxelizationSettings,
        GaussianVoxelizer,
    )

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
    density = pc.get_density  # X-Gaussian 的 get_density 返回 opacity

    # 处理协方差
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
