#
# FSGS configuration for CT reconstruction
#
# Based on: FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting
# Paper: https://arxiv.org/abs/2312.00451
#

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FSGSConfig:
    """
    FSGS 配置参数 (CT 适配版)

    核心功能:
    - proximity 密化: FSGS 核心创新，在稀疏区域自动补充高斯
    - 伪视角训练: CT 圆弧插值生成中间视角
    - confidence 加权: 置信度加权渲染

    CT 适配:
    - 禁用 MiDAS 深度监督 (CT X-ray 投影无法使用)
    - 使用 xray-gaussian-rasterization 替代 RGB 光栅化器
    """

    # ============ 球谐参数 ============
    sh_degree: int = 3

    # ============ Proximity 密化参数 (FSGS 核心功能) ============
    enable_proximity: bool = True
    proximity_from_iter: int = 500
    proximity_until_iter: int = 2000  # FSGS 原始: 仅 iter < 2000 调用
    proximity_interval: int = 100
    proximity_dist_multiplier: float = 5.0  # dist > 5 * extent 触发密化
    proximity_scale_multiplier: float = 1.0  # scale > extent 触发密化
    proximity_k_neighbors: int = 3  # 每个源点生成新点数 (FSGS 默认 N=3)

    # ============ 伪视角参数 (CT 适配) ============
    enable_pseudo_view: bool = True
    pseudo_view_start_iter: int = 2000
    pseudo_view_end_iter: int = 9500
    pseudo_view_interval: int = 10
    n_pseudo_cameras: int = 10000  # 预生成的伪相机数量
    pseudo_view_depth_tv_weight: float = 0.01  # 深度 TV 正则化权重

    # ============ 深度监督参数 (CT 默认禁用 MiDAS) ============
    enable_depth_supervision: bool = False  # CT 无法使用 MiDAS
    enable_fdk_depth: bool = False  # 可选: FDK 深度替代
    depth_weight: float = 0.05
    depth_pseudo_weight: float = 0.5

    # ============ Confidence 参数 ============
    use_confidence: bool = True

    # ============ 标准密化参数 ============
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    percent_dense: float = 0.01

    # ============ 剪枝参数 ============
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005
    prune_from_iter: int = 500

    # ============ 距离剪枝参数 ============
    enable_dist_prune: bool = False  # 基于初始点云距离剪枝
    dist_prune_threshold: float = 3.0

    # ============ 学习率 ============
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # ============ 损失权重 ============
    lambda_dssim: float = 0.2

    # ============ SH 升级 ============
    sh_upgrade_interval: int = 500  # 每 500 迭代升级 SH


def get_fsgs_config() -> FSGSConfig:
    """获取默认 FSGS 配置"""
    return FSGSConfig()
