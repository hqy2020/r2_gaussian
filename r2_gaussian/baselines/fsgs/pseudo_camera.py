#
# CT Pseudo Camera Generator for FSGS
#
# 在训练视角之间进行圆弧插值生成伪视角
# 用于 FSGS 的伪视角深度一致性训练
#

import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Optional

from r2_gaussian.utils.graphics_utils import getWorld2View2, getProjectionMatrix


class CTPseudoCamera(nn.Module):
    """
    CT 伪相机

    用于 FSGS 伪视角训练。伪相机没有 GT 图像，
    仅用于渲染和深度一致性检查。
    """

    def __init__(
        self,
        R: np.ndarray,
        T: np.ndarray,
        angle: float,
        mode: int,
        FoVx: float,
        FoVy: float,
        width: int,
        height: int,
        scanner_cfg: Dict,
        trans: np.ndarray = None,
        scale: float = 1.0,
    ):
        super().__init__()

        self.R = R
        self.T = T
        self.angle = angle
        self.mode = mode
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        if trans is None:
            trans = np.array([0.0, 0.0, 0.0])
        self.trans = trans
        self.scale = scale

        # 计算变换矩阵
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale))
            .transpose(0, 1)
            .cuda()
            .float()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                fovX=self.FoVx,
                fovY=self.FoVy,
                mode=mode,
                scanner_cfg=scanner_cfg,
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class CTPseudoCameraGenerator:
    """
    CT 圆弧轨迹伪视角生成器

    与 FSGS 的 LLFF/360 随机姿态不同，CT 使用圆形轨迹，
    因此伪视角应该在训练视角之间进行圆弧插值。

    示例:
        - 3 views at 0, 120, 240 degrees
        - 伪视角: 40, 80, 160, 200, 280, 320 degrees
    """

    def __init__(
        self,
        train_cameras: List,
        scanner_cfg: Dict,
        n_pseudo: int = 10000,
    ):
        """
        初始化伪视角生成器

        Args:
            train_cameras: 训练相机列表
            scanner_cfg: 扫描器配置
            n_pseudo: 预生成的伪相机数量
        """
        self.train_cameras = train_cameras
        self.scanner_cfg = scanner_cfg
        self.n_pseudo = n_pseudo
        self.pseudo_cameras = []

        self._generate_pseudo_cameras()

    def _generate_pseudo_cameras(self):
        """生成伪相机"""
        if len(self.train_cameras) == 0:
            return

        # 获取训练视角的角度
        angles = [cam.angle for cam in self.train_cameras]
        angles = sorted(angles)

        n_train = len(angles)

        # 参考相机 (用于获取相机参数)
        ref_cam = self.train_cameras[0]

        # 生成策略: 在相邻训练视角之间均匀插值
        n_per_gap = max(1, self.n_pseudo // n_train)

        for i in range(n_train):
            start_angle = angles[i]
            end_angle = angles[(i + 1) % n_train]

            # 处理角度回绕 (0 -> 2*pi)
            if end_angle <= start_angle:
                end_angle += 2 * np.pi

            # 在间隙中均匀插值 (不包含端点，避免与训练视角重叠)
            interp_angles = np.linspace(
                start_angle, end_angle, n_per_gap + 2
            )[1:-1]  # 去掉端点

            for angle in interp_angles:
                # 归一化角度到 [0, 2*pi)
                angle = angle % (2 * np.pi)
                pseudo_cam = self._create_pseudo_camera(angle, ref_cam)
                self.pseudo_cameras.append(pseudo_cam)

        print(f"FSGS: Generated {len(self.pseudo_cameras)} pseudo cameras")

    def _create_pseudo_camera(self, angle: float, ref_cam) -> CTPseudoCamera:
        """
        根据角度创建伪相机

        Args:
            angle: 相机角度 (弧度)
            ref_cam: 参考相机 (用于获取相机参数)

        Returns:
            CTPseudoCamera 实例
        """
        # 计算旋转矩阵 (绕 Y 轴旋转)
        R = self._angle_to_rotation(angle)

        # 计算相机位置 (在圆弧轨迹上)
        # CT 扫描器使用 DSO (Source-to-Origin Distance)
        DSO = self.scanner_cfg.get("DSO", 1.0)
        T = np.array([
            DSO * np.sin(angle),
            0,
            DSO * np.cos(angle)
        ])

        return CTPseudoCamera(
            R=R,
            T=T,
            angle=angle,
            mode=ref_cam.mode,
            FoVx=ref_cam.FoVx,
            FoVy=ref_cam.FoVy,
            width=ref_cam.image_width,
            height=ref_cam.image_height,
            scanner_cfg=self.scanner_cfg,
            trans=ref_cam.trans if hasattr(ref_cam, 'trans') else None,
            scale=ref_cam.scale if hasattr(ref_cam, 'scale') else 1.0,
        )

    def _angle_to_rotation(self, angle: float) -> np.ndarray:
        """
        将角度转换为旋转矩阵

        Args:
            angle: 绕 Y 轴的旋转角度 (弧度)

        Returns:
            3x3 旋转矩阵
        """
        c = np.cos(angle)
        s = np.sin(angle)

        # 绕 Y 轴旋转
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

        return R

    def get_pseudo_camera(self) -> Optional[CTPseudoCamera]:
        """随机获取一个伪相机"""
        if len(self.pseudo_cameras) == 0:
            return None
        return random.choice(self.pseudo_cameras)

    def get_all_pseudo_cameras(self) -> List[CTPseudoCamera]:
        """获取所有伪相机"""
        return self.pseudo_cameras

    def __len__(self) -> int:
        return len(self.pseudo_cameras)


def tv_loss(depth: torch.Tensor) -> torch.Tensor:
    """
    Total Variation 损失 (用于深度平滑正则化)

    Args:
        depth: [H, W] 或 [1, H, W] 深度图

    Returns:
        TV 损失值
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)

    # 计算梯度
    h_diff = torch.abs(depth[1:, :] - depth[:-1, :])
    w_diff = torch.abs(depth[:, 1:] - depth[:, :-1])

    # 平均 TV 损失
    tv = h_diff.mean() + w_diff.mean()

    return tv
