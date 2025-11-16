"""
FSGS 可微深度渲染器
实现论文中的 alpha-blending 深度渲染 (Eq. 7)

参考论文: FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting
Section 3.3: Differentiable Depth Rasterization
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


class FSGSDepthRenderer:
    """
    FSGS 风格的可微深度渲染器
    使用 alpha-blending 渲染深度图
    """

    def __init__(self):
        """初始化深度渲染器"""
        pass

    def render_depth_alpha_blending(self,
                                    viewpoint_camera,
                                    pc,
                                    pipe,
                                    bg_color: torch.Tensor,
                                    scaling_modifier: float = 1.0,
                                    override_color: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        使用 alpha-blending 渲染深度图 (FSGS 论文 Eq. 7)

        d = Σ_{i=1}^n d_i * α_i * Π_{j=1}^{i-1}(1 - α_j)

        Args:
            viewpoint_camera: 视角相机
            pc: GaussianModel 对象
            pipe: Pipeline 参数
            bg_color: 背景颜色
            scaling_modifier: 缩放修饰符
            override_color: 可选的颜色覆盖

        Returns:
            包含渲染深度图的字典 {'depth': depth_map, 'rendered_image': image, ...}
        """
        # 创建深度零张量
        screenspace_points = torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        ) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # 设置光栅化参数
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # 获取尺度和旋转
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # 获取颜色
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2
                )
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                    pc.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # 渲染图像（需要用于获取alpha值）
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # 计算深度值（从相机中心到高斯中心的距离）
        # 这是z-buffer值，对应论文中的d_i
        camera_center = viewpoint_camera.camera_center  # (3,)
        depths = torch.norm(means3D - camera_center.unsqueeze(0), dim=1)  # (N,)

        # 使用深度值代替颜色进行渲染
        # 创建深度颜色：将深度值扩展为3通道
        depth_colors = depths.unsqueeze(1).expand(-1, 3)  # (N, 3)

        # 使用相同的rasterizer设置渲染深度
        depth_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=depth_colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # 提取深度通道（三个通道应该相同）
        depth_map = depth_image[0]  # 取第一个通道

        return {
            "render": rendered_image,
            "depth": depth_map,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def normalize_depth(self, depth: torch.Tensor,
                       min_percentile: float = 1.0,
                       max_percentile: float = 99.0) -> torch.Tensor:
        """
        归一化深度图到 [0, 1] 范围

        Args:
            depth: 输入深度图
            min_percentile: 最小百分位数（用于robust归一化）
            max_percentile: 最大百分位数

        Returns:
            归一化后的深度图
        """
        if depth.numel() == 0:
            return depth

        # 过滤无效值
        valid_depths = depth[torch.isfinite(depth)]

        if valid_depths.numel() == 0:
            return torch.zeros_like(depth)

        # 使用百分位数进行robust归一化
        d_min = torch.quantile(valid_depths, min_percentile / 100.0)
        d_max = torch.quantile(valid_depths, max_percentile / 100.0)

        if d_max > d_min:
            normalized = (depth - d_min) / (d_max - d_min)
            return torch.clamp(normalized, 0.0, 1.0)
        else:
            return torch.zeros_like(depth)


def create_fsgs_depth_renderer():
    """创建FSGS深度渲染器实例"""
    return FSGSDepthRenderer()


# 导入必要的辅助函数
import math

def eval_sh(deg, sh, dirs):
    """
    评估球谐函数
    从gaussian_renderer/__init__.py移植
    """
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    result = C0 * sh[..., 0]

    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                        C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                        C3[1] * xy * z * sh[..., 10] +
                        C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                        C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                        C3[5] * z * (xx - yy) * sh[..., 14] +
                        C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result
