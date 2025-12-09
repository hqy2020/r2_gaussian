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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn


def tv_3d_loss(vol, reduction="sum"):

    dx = torch.abs(torch.diff(vol, dim=0))
    dy = torch.abs(torch.diff(vol, dim=1))
    dz = torch.abs(torch.diff(vol, dim=2))

    tv = torch.sum(dx) + torch.sum(dy) + torch.sum(dz)

    if reduction == "mean":
        total_elements = (
            (vol.shape[0] - 1) * vol.shape[1] * vol.shape[2]
            + vol.shape[0] * (vol.shape[1] - 1) * vol.shape[2]
            + vol.shape[0] * vol.shape[1] * (vol.shape[2] - 1)
        )
        tv = tv / total_elements
    return tv


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ============================================================================
# Sparsity Loss (from SeCuRe paper, adapted for CT reconstruction)
# ============================================================================


def sparsity_loss_ct(
    densities: torch.Tensor,
    positions: torch.Tensor,
    bbox: torch.Tensor,
    air_threshold: float = 0.01,
    delta: float = 0.5,
    boundary_weight: float = 2.0,
) -> torch.Tensor:
    """
    CT 适配的稀疏损失 - 同时惩罚低密度区域和边界外区域

    来源: SeCuRe (Sparse-view 3D Curve Reconstruction) 论文
    原公式: L_sp = Σ_i log(1 + α(p_i) / δ)

    CT 适配:
    - 惩罚低密度区域（空气）的高斯
    - 惩罚边界外区域的高斯（更高权重）

    Args:
        densities: 高斯密度 [N, 1]
        positions: 高斯位置 [N, 3]
        bbox: 边界框 [2, 3]，bbox[0] = min, bbox[1] = max
        air_threshold: 空气密度阈值（低于此值视为空气）
        delta: Cauchy loss 缩放因子
        boundary_weight: 边界外惩罚的额外权重

    Returns:
        sparsity_loss: 标量损失值
    """
    device = densities.device
    total_loss = torch.tensor(0.0, device=device)

    # === Part 1: 低密度区域惩罚 ===
    # 低密度区域不应该有高斯存在
    low_density_mask = densities.squeeze() < air_threshold
    if low_density_mask.sum() > 0:
        # Cauchy loss: log(1 + x²/δ²) - 对离群值鲁棒
        low_density_penalty = torch.log(
            1 + (densities[low_density_mask] / delta) ** 2
        )
        total_loss = total_loss + low_density_penalty.mean()

    # === Part 2: 边界外区域惩罚 ===
    # 超出 bbox 的高斯应该被强烈惩罚
    bbox_min, bbox_max = bbox[0], bbox[1]
    outside_min = (positions < bbox_min).any(dim=1)  # [N]
    outside_max = (positions > bbox_max).any(dim=1)  # [N]
    outside_bbox_mask = outside_min | outside_max

    if outside_bbox_mask.sum() > 0:
        # 边界外高斯：使用更高权重的 Cauchy loss
        boundary_penalty = torch.log(
            1 + (densities[outside_bbox_mask] / delta) ** 2
        )
        total_loss = total_loss + boundary_weight * boundary_penalty.mean()

    return total_loss


def sparsity_loss_density_weighted(
    densities: torch.Tensor,
    target_sparsity: float = 0.8,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    密度加权稀疏损失 - 鼓励紧凑的高斯分布（备选方案）

    惩罚过于分散的低密度高斯，保留高密度的核心区域

    Args:
        densities: 高斯密度 [N, 1]
        target_sparsity: 目标稀疏度（0-1，越高越稀疏）
        temperature: 软化温度

    Returns:
        sparsity_loss: 标量损失值
    """
    # 归一化密度
    normalized = densities / (densities.max() + 1e-7)

    # 软阈值函数：低密度 -> 高惩罚
    weights = torch.sigmoid((target_sparsity - normalized) / temperature)

    # 加权 L1 损失
    loss = (weights * densities).mean()

    return loss
