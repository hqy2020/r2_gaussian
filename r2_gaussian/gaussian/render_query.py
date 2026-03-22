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
import sys
import torch
import math
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.arguments import PipelineParams
from r2_gaussian.utils.sss_utils import student_t_2d, scooping_blend

# drop日志限频：记录上一次打印的训练迭代
_last_drop_log_iter = -1


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Query a volume with voxelization.
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
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
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


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
    enable_drop=False,
    drop_rate: float = 0.10,
    iteration: int = None,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

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
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density
    
    # SSS: ENHANCED Student's t distribution with progressive scooping
    if pc.use_student_t:
        opacity = pc.get_opacity
        nu = pc.get_nu  # Degrees of freedom
        
        # ENHANCED nu range for more expressiveness
        nu = torch.clamp(nu, min=1.5, max=10.0)
        
        # PROGRESSIVE SCOOPING: Allow negative opacity gradually
        if iteration is not None:
            # Phase 1 (0-10k): Only positive opacity
            if iteration < 10000:
                opacity_for_rendering = torch.clamp(opacity, min=0.001, max=1.0)
            # Phase 2 (10k-20k): Allow small negative values
            elif iteration < 20000:
                progress = (iteration - 10000) / 10000.0
                min_opacity = -0.1 * progress  # Gradually allow up to -0.1
                opacity_for_rendering = torch.clamp(opacity, min=min_opacity, max=1.0)
            # Phase 3 (20k+): Full scooping range
            else:
                opacity_for_rendering = torch.clamp(opacity, min=-0.3, max=1.0)
        else:
            # Default: conservative positive range
            opacity_for_rendering = torch.clamp(opacity, min=0.001, max=1.0)
        
        # Enhanced logging for performance monitoring
        if iteration is not None and iteration % 2500 == 0:
            neg_ratio = (opacity < 0).float().mean().item() * 100
            print(f"🎓 [SSS Enhanced] Iter {iteration}: nu[{nu.min():.1f},{nu.max():.1f}], opacity[{opacity.min():.3f},{opacity.max():.3f}], neg%={neg_ratio:.1f}%")
    else:
        opacity = density  # Use density as opacity for backward compatibility
        opacity_for_rendering = density
        nu = None

    # 添加可选的 drop 方法（对所有高斯点生效）
    if enable_drop:
        # 保存原始密度值用于对比
        original_density = density.clone()

        # 直接对所有点应用随机丢弃（不依赖 unique_gidx）
        mask = (torch.rand_like(density) > float(drop_rate)).float()
        density = density * mask

        # 统计并按 500 轮训练限频打印一次
        if iteration is not None and iteration % 500 == 0:
            non_zero_before = (original_density > 0).sum().item()
            non_zero_after = (density > 0).sum().item()
            dropped_points = non_zero_before - non_zero_after
            global _last_drop_log_iter
            if iteration != _last_drop_log_iter:
                _last_drop_log_iter = iteration
                print(
                    f"[iter {iteration}] Drop生效: 原始非零点数={non_zero_before}, 应用drop后非零点数={non_zero_after}, 丢弃点数={dropped_points}, 丢弃比例={dropped_points/max(non_zero_before,1):.2%}"
                )

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # SSS: Use the appropriate opacity (can be negative for scooping)
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity_for_rendering,  # Use stable density for rendering
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    
    # SSS: No post-processing needed for simplified Student's t distribution
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
