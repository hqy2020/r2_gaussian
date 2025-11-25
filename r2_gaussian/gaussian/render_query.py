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
    is_train=False,
    iteration=0,
    model_params=None,
):
    """
    Render an X-ray projection with rasterization.

    Args:
        is_train: 是否为训练模式（DropGaussian 仅在训练时启用）
        iteration: 当前迭代数（用于 DropGaussian 渐进式调整）
        model_params: 模型参数（用于获取 DropGaussian 配置）
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

    # 🎯 DropGaussian: 稀疏视角正则化 (CVPR 2025)
    # 改进版：视角感知 + 分阶段策略
    # 仅在训练时应用，测试时使用全部 Gaussian
    if is_train and model_params is not None and model_params.use_drop_gaussian:
        # === 分阶段策略 ===
        # warmup 阶段不 drop，让网络充分利用稀疏信息建立结构
        if model_params.drop_view_aware and iteration < model_params.drop_warmup_iter:
            drop_rate = 0.0
        else:
            # 基础 drop rate（官方公式）
            if model_params.drop_view_aware:
                # 从 warmup 结束后开始计算进度
                effective_iter = iteration - model_params.drop_warmup_iter
                progress = min(effective_iter / model_params.drop_full_iter, 1.0)
            else:
                progress = min(iteration / model_params.drop_full_iter, 1.0)
            base_drop_rate = model_params.drop_gamma * progress

            # === 视角感知调整 ===
            if model_params.drop_view_aware:
                # 计算当前视角到最近训练视角的距离
                current_idx = viewpoint_camera.uid
                n_total = 50  # 总视角数（CT 扫描固定为 50）
                n_train = model_params.num_train_views  # 训练视角数量

                # 训练视角均匀分布：[0, n_total/n_train, 2*n_total/n_train, ...]
                train_indices = [int(i * n_total / n_train) for i in range(n_train)]

                # 计算到最近训练视角的距离（考虑环形拓扑）
                min_dist = min(
                    min(abs(current_idx - t), n_total - abs(current_idx - t))
                    for t in train_indices
                )

                # 最大可能距离（两个训练视角之间的一半）
                max_dist = n_total / (2 * n_train)

                # 距离衰减因子：距离越远，drop rate 越低
                # dist_factor ∈ [drop_min_factor, 1.0]
                dist_ratio = min(min_dist / max_dist, 1.0)
                dist_factor = 1.0 - model_params.drop_dist_scale * dist_ratio
                dist_factor = max(dist_factor, model_params.drop_min_factor)

                drop_rate = base_drop_rate * dist_factor
            else:
                drop_rate = base_drop_rate

        # 应用 dropout（仅当 drop_rate > 0）
        if drop_rate > 0:
            compensation = torch.ones(density.shape[0], dtype=torch.float32, device="cuda")
            d = torch.nn.Dropout(p=drop_rate)
            compensation = d(compensation)
            density = density * compensation[:, None]

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
