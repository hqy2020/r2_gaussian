"""
IPSM (Inline Prior Guided Score Matching) 工具函数

简化版：针对 X-ray CT 360° 旋转扫描的特性
- 小角度旋转 ≈ 图像水平平移
- 不依赖 SD 或 MiDaS
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from r2_gaussian.dataset.cameras import Camera


# ============================================================
# 简化版 IPSM 工具函数（推荐使用）
# ============================================================

def simple_angle_warp(
    source_image: torch.Tensor,  # (C, H, W) 或 (1, C, H, W)
    angle_diff: float,  # 角度差（弧度）
    direction: str = "horizontal"
) -> torch.Tensor:
    """
    简化的角度 warp：利用 X-ray CT 的特性

    对于 360° 旋转扫描，小角度旋转近似等于图像水平平移：
    Δx = Δθ × W / (2π)

    Args:
        source_image: 源图像 (C, H, W) 或 (1, C, H, W)
        angle_diff: 角度差（弧度），正值向右平移
        direction: "horizontal" (默认) 或 "vertical"

    Returns:
        warped_image: 平移后的图像，与输入形状相同
    """
    # 确保 4D
    squeeze_output = False
    if source_image.dim() == 3:
        source_image = source_image.unsqueeze(0)
        squeeze_output = True

    B, C, H, W = source_image.shape
    device = source_image.device

    # 计算归一化平移量 [-1, 1]
    # angle_diff 弧度 -> 像素平移 -> 归一化
    if direction == "horizontal":
        # 水平平移：Δx = Δθ × W / (2π)，归一化到 [-1, 1] 需要 × 2 / W
        shift_normalized = angle_diff / np.pi  # 简化：Δθ / π
        shift_x = shift_normalized
        shift_y = 0.0
    else:
        shift_x = 0.0
        shift_y = angle_diff / np.pi

    # 创建采样网格
    theta = torch.tensor([
        [1.0, 0.0, shift_x],
        [0.0, 1.0, shift_y]
    ], dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)

    grid = F.affine_grid(theta, source_image.shape, align_corners=True)

    # 采样
    warped = F.grid_sample(
        source_image,
        grid,
        mode='bilinear',
        padding_mode='zeros',  # 边界填零
        align_corners=True
    )

    if squeeze_output:
        warped = warped.squeeze(0)

    return warped


def compute_warp_mask(
    image_shape: Tuple[int, int],  # (H, W)
    angle_diff: float,  # 弧度
    device: str = "cuda"
) -> torch.Tensor:
    """
    计算 warp 后的有效区域 mask

    由于平移会导致边界区域无效，需要生成 mask

    Args:
        image_shape: (H, W)
        angle_diff: 角度差（弧度）
        device: 设备

    Returns:
        mask: (H, W) 有效区域为 1，无效区域为 0
    """
    H, W = image_shape

    # 计算无效边界宽度
    shift_pixels = abs(angle_diff) * W / (2 * np.pi)
    invalid_width = int(np.ceil(shift_pixels))

    # 创建 mask
    mask = torch.ones(H, W, device=device)

    if invalid_width > 0 and invalid_width < W // 2:
        if angle_diff > 0:
            # 向右平移，左边界无效
            mask[:, :invalid_width] = 0
        else:
            # 向左平移，右边界无效
            mask[:, -invalid_width:] = 0

    return mask


def depth_tv_loss(depth: torch.Tensor) -> torch.Tensor:
    """
    深度图 Total Variation 损失

    鼓励深度图平滑，减少噪声

    Args:
        depth: (H, W) 或 (1, H, W) 或 (B, 1, H, W) 深度图

    Returns:
        tv_loss: 标量损失
    """
    # 确保至少 3D
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0)

    # 计算梯度
    diff_h = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])
    diff_w = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])

    # L1 TV loss
    tv_loss = diff_h.mean() + diff_w.mean()

    return tv_loss


def edge_aware_smooth_loss(
    depth: torch.Tensor,  # (H, W) 或 (1, H, W)
    image: torch.Tensor   # (C, H, W) 参考图像
) -> torch.Tensor:
    """
    边缘感知的深度平滑损失

    在图像边缘处允许更大的深度变化，在平坦区域强制平滑
    公式: L = mean(|∇d| * exp(-|∇I|))

    Args:
        depth: 深度图
        image: 参考图像（用于检测边缘）

    Returns:
        loss: 标量损失
    """
    # 确保维度正确
    if depth.dim() == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.dim() == 3:
        depth = depth.unsqueeze(0)

    if image.dim() == 3:
        image = image.unsqueeze(0)

    # 计算深度梯度
    depth_dx = torch.abs(depth[:, :, :, 1:] - depth[:, :, :, :-1])
    depth_dy = torch.abs(depth[:, :, 1:, :] - depth[:, :, :-1, :])

    # 计算图像梯度（取平均通道）
    image_gray = image.mean(dim=1, keepdim=True)
    image_dx = torch.abs(image_gray[:, :, :, 1:] - image_gray[:, :, :, :-1])
    image_dy = torch.abs(image_gray[:, :, 1:, :] - image_gray[:, :, :-1, :])

    # 边缘权重: exp(-|∇I|)
    weight_x = torch.exp(-image_dx)
    weight_y = torch.exp(-image_dy)

    # 边缘感知平滑损失
    loss = (weight_x * depth_dx).mean() + (weight_y * depth_dy).mean()

    return loss


# ============================================================
# 原始 XRayIPSMWarping 类（保留但不推荐使用）
# ============================================================


class XRayIPSMWarping:
    """
    IPSM的X-ray几何适配版本
    通过体素反投影实现inverse warping
    """

    def __init__(self, scanner_cfg: dict, pipe):
        """
        Args:
            scanner_cfg: Scanner配置（包含投影参数）
            pipe: Pipeline参数
        """
        self.scanner_cfg = scanner_cfg
        self.pipe = pipe

        # 提取X-ray投影参数
        self.n_voxel = torch.tensor(scanner_cfg["nVoxel"], dtype=torch.int32)
        self.s_voxel = torch.tensor(scanner_cfg["sVoxel"], dtype=torch.float32)
        self.off_origin = torch.tensor(scanner_cfg["offOrigin"], dtype=torch.float32)

    def warp_via_voxel_projection(
        self,
        source_image: torch.Tensor,  # (C, H, W)
        source_cam: Camera,
        target_cam: Camera,
        target_depth: torch.Tensor,  # (H, W)
        tau: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过体素反投影实现inverse warping

        流程:
        1. 从target_depth和target_cam重建3D点云
        2. 将3D点投影到source_cam获取采样坐标
        3. 从source_image采样得到warped_image
        4. 通过深度一致性生成consistency_mask

        Args:
            source_image: 源视角图像 (C, H, W) [0, 1]
            source_cam: 源相机
            target_cam: 目标相机
            target_depth: 目标视角深度图 (H, W)
            tau: Consistency mask阈值

        Returns:
            warped_image: (C, H, W) 从source warp到target的图像
            mask: (H, W) Consistency mask [0, 1]
        """
        C, H_t, W_t = source_image.shape
        device = source_image.device

        # === 步骤1: 从target深度重建3D点云 ===
        # 生成像素网格
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H_t, device=device, dtype=torch.float32),
            torch.arange(W_t, device=device, dtype=torch.float32),
            indexing='ij'
        )  # (H, W)

        # 归一化像素坐标到[-1, 1]
        u_norm = 2.0 * u_grid / (W_t - 1) - 1.0
        v_norm = 2.0 * v_grid / (H_t - 1) - 1.0

        # 使用投影矩阵的逆反投影到3D
        # 简化方案: 使用相机中心 + 射线方向 * depth
        points_3d = self._unproject_pixels(
            u_norm, v_norm, target_depth, target_cam
        )  # (H, W, 3)

        # === 步骤2: 投影到source_cam ===
        # 将3D点投影到source相机
        u_src, v_src, depth_src = self._project_points(
            points_3d, source_cam
        )  # (H, W), (H, W), (H, W)

        # === 步骤3: 从source_image采样 ===
        # 转换到grid_sample需要的格式 [-1, 1]
        grid = torch.stack([u_src, v_src], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # 采样
        warped_image = F.grid_sample(
            source_image.unsqueeze(0),  # (1, C, H, W)
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).squeeze(0)  # (C, H, W)

        # === 步骤4: 生成consistency mask ===
        # 从source渲染深度并warp回target，与target_depth比较
        # 简化实现: 使用投影深度一致性
        mask = self._compute_consistency_mask(
            depth_src, target_depth, u_src, v_src, tau
        )  # (H, W)

        return warped_image, mask

    def _unproject_pixels(
        self,
        u_norm: torch.Tensor,  # (H, W) [-1, 1]
        v_norm: torch.Tensor,  # (H, W) [-1, 1]
        depth: torch.Tensor,  # (H, W)
        camera: Camera
    ) -> torch.Tensor:
        """
        反投影像素到3D空间

        Args:
            u_norm, v_norm: 归一化像素坐标 [-1, 1]
            depth: 深度值
            camera: 相机对象

        Returns:
            points_3d: (H, W, 3) 3D点坐标
        """
        H, W = u_norm.shape

        # 从相机获取变换矩阵
        # 注意: R²-Gaussian使用X-ray投影，需要适配
        # 这里使用简化模型: 平行光源 + 相机中心

        # 获取相机中心（光源位置）
        cam_center = camera.camera_center  # (3,)

        # 相机朝向（Z轴）
        R = camera.R  # (3, 3) numpy
        R_torch = torch.from_numpy(R).float().to(u_norm.device)
        cam_direction = R_torch[:, 2]  # Z轴方向 (3,)

        # 射线方向（简化：平行光）
        # 对于X-ray，射线方向近似平行于相机Z轴
        ray_dir = cam_direction.view(1, 1, 3).expand(H, W, 3)

        # 3D点 = 相机中心 + 射线方向 * depth
        points_3d = cam_center.view(1, 1, 3) + ray_dir * depth.unsqueeze(-1)

        return points_3d

    def _project_points(
        self,
        points_3d: torch.Tensor,  # (H, W, 3)
        camera: Camera
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将3D点投影到相机平面

        Args:
            points_3d: 3D点 (H, W, 3)
            camera: 目标相机

        Returns:
            u_norm: (H, W) 归一化u坐标 [-1, 1]
            v_norm: (H, W) 归一化v坐标 [-1, 1]
            depth: (H, W) 深度值
        """
        H, W, _ = points_3d.shape
        device = points_3d.device

        # 转换为齐次坐标
        points_3d_flat = points_3d.reshape(-1, 3)  # (H*W, 3)
        points_homog = torch.cat([
            points_3d_flat,
            torch.ones(H * W, 1, device=device)
        ], dim=1)  # (H*W, 4)

        # 应用投影矩阵
        proj_mat = camera.full_proj_transform  # (4, 4)
        points_proj = points_homog @ proj_mat.T  # (H*W, 4)

        # 透视除法
        points_ndc = points_proj[:, :3] / (points_proj[:, 3:4] + 1e-8)  # (H*W, 3)

        # 提取u, v, depth
        u_norm = points_ndc[:, 0].reshape(H, W)  # [-1, 1]
        v_norm = points_ndc[:, 1].reshape(H, W)
        depth = points_ndc[:, 2].reshape(H, W)

        return u_norm, v_norm, depth

    def _compute_consistency_mask(
        self,
        depth_projected: torch.Tensor,  # (H, W)
        depth_target: torch.Tensor,  # (H, W)
        u_coords: torch.Tensor,  # (H, W)
        v_coords: torch.Tensor,  # (H, W)
        tau: float
    ) -> torch.Tensor:
        """
        计算深度一致性mask

        Args:
            depth_projected: 投影后的深度
            depth_target: 目标深度
            u_coords, v_coords: 投影坐标
            tau: 阈值

        Returns:
            mask: (H, W) 二值mask [0, 1]
        """
        # 1. 深度差异
        depth_diff = torch.abs(depth_projected - depth_target)
        depth_consistent = depth_diff < tau

        # 2. 坐标在有效范围内
        coords_valid = (
            (u_coords >= -1.0) & (u_coords <= 1.0) &
            (v_coords >= -1.0) & (v_coords <= 1.0)
        )

        # 3. 深度为正
        depth_positive = (depth_projected > 0) & (depth_target > 0)

        # 组合条件
        mask = (depth_consistent & coords_valid & depth_positive).float()

        return mask


def sample_nearby_viewpoint(
    base_camera: Camera,
    angle_range: float = 15.0,
    angle_deg: Optional[float] = None,
    device: str = "cuda"
) -> Tuple[Camera, float]:
    """
    在base_camera附近采样一个伪视角

    Args:
        base_camera: 基准相机
        angle_range: 角度扰动范围（度），当 angle_deg 未指定时使用
        angle_deg: 指定的角度（度），如果提供则使用此值，否则随机采样
        device: 设备

    Returns:
        pseudo_camera: 伪视角相机
        angle_rad: 实际使用的角度（弧度）
    """
    import copy

    # 深拷贝相机
    pseudo_cam = copy.deepcopy(base_camera)

    # 确定旋转角度
    if angle_deg is not None:
        angle_perturbation = angle_deg
    else:
        # 随机旋转角度（围绕Y轴）
        angle_perturbation = (np.random.rand() - 0.5) * 2 * angle_range

    angle_rad = np.deg2rad(angle_perturbation)

    # 旋转矩阵（绕Y轴）
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R_perturb = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    # 应用扰动
    R_new = base_camera.R @ R_perturb
    pseudo_cam.R = R_new

    # 更新变换矩阵
    from r2_gaussian.utils.graphics_utils import getWorld2View2
    pseudo_cam.world_view_transform = (
        torch.tensor(getWorld2View2(
            R_new,
            base_camera.T,
            base_camera.trans,
            base_camera.scale
        )).transpose(0, 1).to(device)
    )

    pseudo_cam.full_proj_transform = (
        pseudo_cam.world_view_transform.unsqueeze(0).bmm(
            base_camera.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)

    pseudo_cam.camera_center = pseudo_cam.world_view_transform.inverse()[3, :3]

    return pseudo_cam, angle_rad
