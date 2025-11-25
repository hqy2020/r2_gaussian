"""
FSGS 伪视角采样器
生成用于深度监督的扰动相机视角

参考 FSGS 论文: https://github.com/VITA-Group/FSGS
"""

import torch
import numpy as np
from typing import Tuple, Optional
import math


def rotation_matrix_x(angle: float) -> torch.Tensor:
    """绕 X 轴旋转矩阵"""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=torch.float32)


def rotation_matrix_y(angle: float) -> torch.Tensor:
    """绕 Y 轴旋转矩阵"""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=torch.float32)


def rotation_matrix_z(angle: float) -> torch.Tensor:
    """绕 Z 轴旋转矩阵"""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=torch.float32)


class PseudoViewSampler:
    """
    伪视角采样器

    FSGS 核心创新：生成与训练视角略有偏移的伪视角，
    利用 MiDaS 深度先验提供监督信号，增强几何一致性

    Args:
        angle_perturbation: 角度扰动范围 (弧度)，默认 15 度
        translation_perturbation: 平移扰动范围，默认 0.1
        device: 计算设备

    Example:
        >>> sampler = PseudoViewSampler(angle_perturbation=0.26, device="cuda")
        >>> pseudo_camera = sampler.sample_pseudo_view(original_camera)
    """

    def __init__(
        self,
        angle_perturbation: float = 0.26,  # ~15 度
        translation_perturbation: float = 0.1,
        device: str = "cuda"
    ):
        self.angle_perturbation = angle_perturbation
        self.translation_perturbation = translation_perturbation
        self.device = device

    def _perturb_rotation(self, R: torch.Tensor) -> torch.Tensor:
        """
        对旋转矩阵添加随机扰动

        Args:
            R: (3, 3) 原始旋转矩阵

        Returns:
            R_perturbed: (3, 3) 扰动后的旋转矩阵
        """
        # 随机生成三个轴的扰动角度
        delta_x = (torch.rand(1).item() * 2 - 1) * self.angle_perturbation
        delta_y = (torch.rand(1).item() * 2 - 1) * self.angle_perturbation
        delta_z = (torch.rand(1).item() * 2 - 1) * self.angle_perturbation

        # 构建扰动旋转矩阵
        Rx = rotation_matrix_x(delta_x).to(R.device)
        Ry = rotation_matrix_y(delta_y).to(R.device)
        Rz = rotation_matrix_z(delta_z).to(R.device)

        # 组合扰动
        delta_R = Rz @ Ry @ Rx

        # 应用扰动
        R_perturbed = delta_R @ R

        return R_perturbed

    def _perturb_translation(self, T: torch.Tensor) -> torch.Tensor:
        """
        对平移向量添加随机扰动

        Args:
            T: (3,) 原始平移向量

        Returns:
            T_perturbed: (3,) 扰动后的平移向量
        """
        delta_T = (torch.rand(3, device=T.device) * 2 - 1) * self.translation_perturbation
        return T + delta_T

    def sample_pseudo_view(
        self,
        viewpoint_camera,
        perturb_rotation: bool = True,
        perturb_translation: bool = True
    ):
        """
        从原始视角采样伪视角

        Args:
            viewpoint_camera: 原始相机对象 (需要有 R, T 属性)
            perturb_rotation: 是否扰动旋转
            perturb_translation: 是否扰动平移

        Returns:
            pseudo_camera: 伪视角相机对象 (与原始相机类型相同)
        """
        import copy

        # 深拷贝相机对象
        pseudo_camera = copy.deepcopy(viewpoint_camera)

        # 获取原始相机参数
        R = viewpoint_camera.R.clone() if torch.is_tensor(viewpoint_camera.R) else torch.tensor(viewpoint_camera.R, dtype=torch.float32)
        T = viewpoint_camera.T.clone() if torch.is_tensor(viewpoint_camera.T) else torch.tensor(viewpoint_camera.T, dtype=torch.float32)

        # 应用扰动
        if perturb_rotation:
            R = self._perturb_rotation(R)
        if perturb_translation:
            T = self._perturb_translation(T)

        # 更新伪相机参数
        if torch.is_tensor(viewpoint_camera.R):
            pseudo_camera.R = R
            pseudo_camera.T = T
        else:
            pseudo_camera.R = R.cpu().numpy()
            pseudo_camera.T = T.cpu().numpy()

        # 重新计算 world_view_transform 和 full_proj_transform
        if hasattr(pseudo_camera, 'update_transforms'):
            pseudo_camera.update_transforms()
        else:
            # 手动更新变换矩阵
            pseudo_camera.world_view_transform = self._compute_world_view_transform(R, T)
            if hasattr(pseudo_camera, 'projection_matrix'):
                pseudo_camera.full_proj_transform = (
                    pseudo_camera.world_view_transform.unsqueeze(0).bmm(
                        pseudo_camera.projection_matrix.unsqueeze(0)
                    )
                ).squeeze(0)

        return pseudo_camera

    def _compute_world_view_transform(
        self,
        R: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 world-to-view 变换矩阵

        Args:
            R: (3, 3) 旋转矩阵
            T: (3,) 平移向量

        Returns:
            transform: (4, 4) 变换矩阵
        """
        transform = torch.eye(4, dtype=R.dtype, device=R.device)
        transform[:3, :3] = R.T  # 转置
        transform[:3, 3] = T
        return transform.T  # 列主序


def sample_pseudo_views_batch(
    viewpoint_cameras: list,
    sampler: PseudoViewSampler,
    num_pseudo_per_view: int = 1
) -> list:
    """
    批量采样伪视角

    Args:
        viewpoint_cameras: 原始相机列表
        sampler: 伪视角采样器
        num_pseudo_per_view: 每个视角生成的伪视角数量

    Returns:
        pseudo_cameras: 伪视角相机列表
    """
    pseudo_cameras = []
    for camera in viewpoint_cameras:
        for _ in range(num_pseudo_per_view):
            pseudo_cameras.append(sampler.sample_pseudo_view(camera))
    return pseudo_cameras


# CT 专用伪视角采样器
class CTCircularPseudoViewSampler(PseudoViewSampler):
    """
    CT 圆形轨迹专用伪视角采样器

    考虑 CT 成像的特殊性：
    1. 相机绕 Z 轴做圆周运动
    2. 扰动主要在圆周切向方向
    3. 保持到 isocenter 的距离不变

    Args:
        angle_perturbation: 圆周角度扰动 (弧度)，默认 10 度
        elevation_perturbation: 俯仰角扰动 (弧度)，默认 5 度
        device: 计算设备
    """

    def __init__(
        self,
        angle_perturbation: float = 0.175,  # ~10 度
        elevation_perturbation: float = 0.087,  # ~5 度
        device: str = "cuda"
    ):
        super().__init__(
            angle_perturbation=angle_perturbation,
            translation_perturbation=0.0,  # CT 不扰动平移
            device=device
        )
        self.elevation_perturbation = elevation_perturbation

    def _perturb_rotation(self, R: torch.Tensor) -> torch.Tensor:
        """
        CT 专用旋转扰动

        主要扰动绕 Z 轴的角度（圆周切向）和俯仰角
        """
        # 圆周角度扰动 (绕 Z 轴)
        delta_z = (torch.rand(1).item() * 2 - 1) * self.angle_perturbation
        # 俯仰角扰动 (绕 X 轴)
        delta_x = (torch.rand(1).item() * 2 - 1) * self.elevation_perturbation

        Rz = rotation_matrix_z(delta_z).to(R.device)
        Rx = rotation_matrix_x(delta_x).to(R.device)

        # 先应用俯仰扰动，再应用圆周扰动
        delta_R = Rz @ Rx

        return delta_R @ R

    def sample_pseudo_view(
        self,
        viewpoint_camera,
        perturb_rotation: bool = True,
        perturb_translation: bool = False  # CT 默认不扰动平移
    ):
        """CT 专用伪视角采样"""
        return super().sample_pseudo_view(
            viewpoint_camera,
            perturb_rotation=perturb_rotation,
            perturb_translation=perturb_translation
        )


# 全局采样器实例
_global_sampler: Optional[PseudoViewSampler] = None


def get_pseudo_view_sampler(
    sampler_type: str = "ct_circular",
    **kwargs
) -> PseudoViewSampler:
    """
    获取伪视角采样器（单例模式）

    Args:
        sampler_type: 采样器类型
            - "default": 通用采样器
            - "ct_circular": CT 圆形轨迹采样器
        **kwargs: 传递给采样器的参数

    Returns:
        sampler: 伪视角采样器
    """
    global _global_sampler

    if _global_sampler is None:
        if sampler_type == "ct_circular":
            _global_sampler = CTCircularPseudoViewSampler(**kwargs)
        else:
            _global_sampler = PseudoViewSampler(**kwargs)

    return _global_sampler
