#
# FSGS renderer adapted for CT reconstruction
#
# 复用 R2-Gaussian 的 xray-gaussian-rasterization 进行 X-ray 投影
# 支持 confidence 加权渲染
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

from .model import FSGSModel
from ..registry import BaseRenderer


def render_fsgs(
    viewpoint_camera,
    pc: FSGSModel,
    pipe,
    bg_color: torch.Tensor = None,
    scaling_modifier: float = 1.0,
    use_confidence: bool = None,
) -> Dict[str, torch.Tensor]:
    """
    FSGS 渲染函数

    使用 R2-Gaussian 的 xray-gaussian-rasterization 进行 X-ray 投影。
    支持 confidence 加权渲染。

    Args:
        viewpoint_camera: 相机视角
        pc: FSGSModel 实例
        pipe: PipelineParams
        bg_color: 背景颜色 (可选)
        scaling_modifier: 尺度修正因子
        use_confidence: 是否使用 confidence 加权 (默认使用配置)

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

    # 设置相机参数（与 R2-Gaussian 保持一致）
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

    # 使用 opacity 作为密度
    density = pc.get_density

    # FSGS: confidence 加权 (可选)
    if use_confidence is None:
        use_confidence = pc.config.use_confidence
    if use_confidence and pc.confidence.shape[0] > 0:
        density = density * pc.confidence

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


class FSGSRenderer(BaseRenderer):
    """FSGS 渲染器类"""

    def __init__(self, pipe=None):
        self.pipe = pipe

    def render(
        self,
        viewpoint,
        model: FSGSModel,
        pipe=None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """渲染接口实现"""
        if pipe is None:
            pipe = self.pipe
        return render_fsgs(viewpoint, model, pipe, **kwargs)


def query_fsgs(
    pc: FSGSModel,
    center,
    nVoxel,
    sVoxel,
    pipe,
    scaling_modifier=1.0,
    use_confidence: bool = None,
) -> Dict[str, torch.Tensor]:
    """
    FSGS 体积查询函数

    直接复用 R2-Gaussian 的 GaussianVoxelizer

    Args:
        pc: FSGSModel 实例
        center: 体积中心
        nVoxel: 体素数量 [nx, ny, nz]
        sVoxel: 体素尺寸 [sx, sy, sz]
        pipe: PipelineParams
        scaling_modifier: 尺度修正因子
        use_confidence: 是否使用 confidence 加权

    Returns:
        dict:
            - vol: 3D 体积 [nx, ny, nz]
            - radii: 2D 半径
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
    density = pc.get_density

    # FSGS: confidence 加权
    if use_confidence is None:
        use_confidence = pc.config.use_confidence
    if use_confidence and pc.confidence.shape[0] > 0:
        density = density * pc.confidence

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
