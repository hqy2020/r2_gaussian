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


# ==================== IPSM Loss Functions ====================

def pearson_correlation_loss(
    depth_rendered: torch.Tensor,
    depth_mono: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Pearson相关系数损失（用于深度正则化）

    Args:
        depth_rendered: 渲染深度 (H, W) 或 (1, H, W)
        depth_mono: 单目估计深度 (H, W) 或 (1, H, W)
        mask: 可选mask (H, W)，只在有效区域计算

    Returns:
        loss: 1 - Pearson相关系数 [0, 2]，越小越好
    """
    # 确保是2D tensor
    if depth_rendered.dim() == 3:
        depth_rendered = depth_rendered.squeeze(0)
    if depth_mono.dim() == 3:
        depth_mono = depth_mono.squeeze(0)

    # 应用mask
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        mask_bool = mask > 0.5
        if mask_bool.sum() < 10:  # 有效像素太少
            return torch.tensor(0.0, device=depth_rendered.device)

        d_r = depth_rendered[mask_bool]
        d_m = depth_mono[mask_bool]
    else:
        d_r = depth_rendered.flatten()
        d_m = depth_mono.flatten()

    # 计算均值
    mean_r = d_r.mean()
    mean_m = d_m.mean()

    # 计算协方差和方差
    cov = ((d_r - mean_r) * (d_m - mean_m)).mean()
    var_r = ((d_r - mean_r) ** 2).mean()
    var_m = ((d_m - mean_m) ** 2).mean()

    # Pearson相关系数
    corr = cov / (torch.sqrt(var_r * var_m) + 1e-8)

    # 返回1 - |correlation|作为loss（希望相关性接近1或-1）
    loss = 1.0 - torch.abs(corr)

    return loss


def geometry_consistency_loss(
    rendered_img: torch.Tensor,
    warped_img: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    几何一致性损失（masked L1 loss）

    Args:
        rendered_img: 渲染的伪视角图像 (C, H, W)
        warped_img: Warped图像 (C, H, W)
        mask: 一致性mask (H, W)，1表示可信区域

    Returns:
        loss: Masked L1 loss标量
    """
    # 确保mask是正确的形状
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # (1, H, W)

    # 计算差异
    diff = torch.abs(rendered_img - warped_img)

    # 应用mask
    masked_diff = diff * mask

    # 归一化（除以有效像素数）
    num_valid = mask.sum() + 1e-8
    loss = masked_diff.sum() / num_valid

    return loss


def ipsm_depth_regularization(
    depth_rendered_seen: torch.Tensor,
    depth_mono_seen: torch.Tensor,
    depth_rendered_unseen: torch.Tensor,
    depth_mono_unseen: torch.Tensor,
    eta_d: float = 0.1,
    mask_seen: torch.Tensor = None,
    mask_unseen: torch.Tensor = None
) -> torch.Tensor:
    """
    IPSM深度正则化（组合seen和unseen视角）

    L_depth = η_d * Corr(D_r^i, D_m^i) + Corr(D_r^j, D_m^j)

    Args:
        depth_rendered_seen: 已知视角渲染深度
        depth_mono_seen: 已知视角单目深度
        depth_rendered_unseen: 伪视角渲染深度
        depth_mono_unseen: 伪视角单目深度
        eta_d: seen视角权重
        mask_seen: 已知视角mask
        mask_unseen: 伪视角mask

    Returns:
        loss_depth: 组合深度损失
    """
    loss_seen = pearson_correlation_loss(
        depth_rendered_seen,
        depth_mono_seen,
        mask_seen
    )

    loss_unseen = pearson_correlation_loss(
        depth_rendered_unseen,
        depth_mono_unseen,
        mask_unseen
    )

    loss_depth = eta_d * loss_seen + loss_unseen

    return loss_depth
