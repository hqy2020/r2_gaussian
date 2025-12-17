#
# CoR-GS: Pseudo-view generator for CT projection
#
# 为 CT 投影场景生成伪视图相机，用于 Co-Regularization
#

import numpy as np
import torch
from torch import nn
from typing import List

from r2_gaussian.utils.graphics_utils import getWorld2View2, getProjectionMatrix


def angle2pose(DSO, angle):
    """
    将角度转换为相机位姿（c2w）

    复用自 dataset_readers.py
    """
    phi1 = -np.pi / 2
    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi1), -np.sin(phi1)],
        [0.0, np.sin(phi1), np.cos(phi1)],
    ])
    phi2 = np.pi / 2
    R2 = np.array([
        [np.cos(phi2), -np.sin(phi2), 0.0],
        [np.sin(phi2), np.cos(phi2), 0.0],
        [0.0, 0.0, 1.0],
    ])
    R3 = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform


class PseudoCamera(nn.Module):
    """
    伪视图相机类

    与 Camera 类似，但没有 GT 图像。用于 Co-Regularization。
    """

    def __init__(
        self,
        scanner_cfg,
        R,
        T,
        angle,
        mode,
        FoVx,
        FoVy,
        image_width,
        image_height,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ):
        super(PseudoCamera, self).__init__()

        self.uid = uid
        self.R = R
        self.T = T
        self.angle = angle
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.mode = mode
        self.image_width = image_width
        self.image_height = image_height
        self.trans = trans
        self.scale = scale

        # 计算变换矩阵
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
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


def generate_pseudo_cameras_ct(
    train_cameras,
    scanner_cfg,
    n_views: int = 1000,
    seed: int = 42,
) -> List[PseudoCamera]:
    """
    为 CT 投影场景生成伪视图相机

    策略：在训练视角的角度范围内随机采样新角度

    Args:
        train_cameras: 训练相机列表
        scanner_cfg: 扫描仪配置（包含 DSO, DSD 等）
        n_views: 伪视图数量
        seed: 随机种子

    Returns:
        pseudo_cameras: PseudoCamera 列表
    """
    np.random.seed(seed)

    # 1. 提取训练视角的角度范围
    train_angles = [cam.angle for cam in train_cameras]
    angle_min, angle_max = min(train_angles), max(train_angles)

    # 扩展角度范围（允许在训练角度间内插）
    # 对于稀疏视角，扩展范围可能不合适，保持原有范围

    # 2. 在范围内随机采样新角度
    pseudo_angles = np.random.uniform(angle_min, angle_max, n_views)

    # 3. 获取模板相机参数
    template_cam = train_cameras[0]
    DSO = scanner_cfg["DSO"]

    # 4. 构建伪视图相机
    pseudo_cameras = []
    for idx, angle in enumerate(pseudo_angles):
        # 从角度计算位姿
        c2w = angle2pose(DSO, angle)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # 转置用于 CUDA 代码
        T = w2c[:3, 3]

        pseudo_cam = PseudoCamera(
            scanner_cfg=scanner_cfg,
            R=R,
            T=T,
            angle=angle,
            mode=template_cam.mode,
            FoVx=template_cam.FoVx,
            FoVy=template_cam.FoVy,
            image_width=template_cam.image_width,
            image_height=template_cam.image_height,
            uid=10000 + idx,  # 使用大 uid 避免冲突
        )
        pseudo_cameras.append(pseudo_cam)

    print(f"[CoR-GS] Generated {len(pseudo_cameras)} pseudo cameras "
          f"(angle range: [{np.degrees(angle_min):.1f}, {np.degrees(angle_max):.1f}] deg)")

    return pseudo_cameras


def generate_pseudo_cameras_interpolate(
    train_cameras,
    scanner_cfg,
    n_views_per_interval: int = 100,
) -> List[PseudoCamera]:
    """
    在训练视角之间均匀插值生成伪视图

    这种方法确保伪视图均匀分布在已知视角之间。

    Args:
        train_cameras: 训练相机列表
        scanner_cfg: 扫描仪配置
        n_views_per_interval: 每个间隔内的伪视图数

    Returns:
        pseudo_cameras: PseudoCamera 列表
    """
    # 按角度排序训练相机
    sorted_cams = sorted(train_cameras, key=lambda x: x.angle)
    n_train = len(sorted_cams)

    template_cam = train_cameras[0]
    DSO = scanner_cfg["DSO"]

    pseudo_cameras = []
    uid_counter = 10000

    # 在每对相邻训练视角之间插值
    for i in range(n_train - 1):
        angle_start = sorted_cams[i].angle
        angle_end = sorted_cams[i + 1].angle

        # 在间隔内均匀采样（不包含端点）
        pseudo_angles = np.linspace(
            angle_start, angle_end, n_views_per_interval + 2
        )[1:-1]

        for angle in pseudo_angles:
            c2w = angle2pose(DSO, angle)
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            pseudo_cam = PseudoCamera(
                scanner_cfg=scanner_cfg,
                R=R,
                T=T,
                angle=angle,
                mode=template_cam.mode,
                FoVx=template_cam.FoVx,
                FoVy=template_cam.FoVy,
                image_width=template_cam.image_width,
                image_height=template_cam.image_height,
                uid=uid_counter,
            )
            pseudo_cameras.append(pseudo_cam)
            uid_counter += 1

    print(f"[CoR-GS] Generated {len(pseudo_cameras)} interpolated pseudo cameras")

    return pseudo_cameras
