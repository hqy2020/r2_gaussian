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


def compute_graph_laplacian_loss(gaussians, graph, lambda_lap=8e-4):
    """
    计算 Graph Laplacian 正则化损失 (GR-Gaussian)

    功能：
        鼓励邻近高斯点的密度值平滑，减少针状伪影

    公式：
        L_lap = λ_lap * Σ_{(i,j)∈E} w_ij * (ρ_i - ρ_j)²

    Args:
        gaussians: GaussianModel 实例
        graph: GaussianGraph 实例（包含 edge_index 和 edge_weights）
        lambda_lap: 正则化权重（论文推荐 8e-4）

    Returns:
        loss: 标量 Tensor，Graph Laplacian 损失

    参考：
        GR-Gaussian论文 Eq.19: L_norm = λ_lap * L_lap(G) + λ_tv * L_tv
    """
    # 边界情况检查
    if graph is None:
        return torch.tensor(0.0, device=gaussians.get_xyz.device, requires_grad=True)

    if graph.num_edges == 0:
        return torch.tensor(0.0, device=gaussians.get_xyz.device, requires_grad=True)

    # 获取高斯点密度（通过激活函数后的值）
    density = gaussians.get_density.squeeze()  # (N,) - 已经过 softplus 激活，确保是1D

    # 获取图结构
    edge_index = graph.get_edge_index()  # (2, E)
    edge_weights = graph.get_edge_weights()  # (E,)

    if edge_index is None or edge_weights is None:
        return torch.tensor(0.0, device=density.device, requires_grad=True)

    # 提取边的源节点和目标节点索引（确保是1D且long类型）
    src = edge_index[0].long().contiguous()  # (E,)
    dst = edge_index[1].long().contiguous()  # (E,)

    # 确保索引在有效范围内
    N = density.shape[0]
    assert src.max() < N and dst.max() < N, f"Index out of bounds: max src={src.max()}, max dst={dst.max()}, N={N}"
    assert density.dim() == 1, f"density must be 1D, got shape {density.shape}"  # 🔍 验证形状

    # 计算邻近点密度差异（使用advanced indexing）
    density_src = density[src]  # (E,)
    density_dst = density[dst]  # (E,)
    density_diff = density_src - density_dst  # (E,)

    # 确保edge_weights是1D且contiguous
    edge_weights_flat = edge_weights.contiguous().view(-1)  # (E,)

    # 加权平方差（Graph Laplacian）- 逐元素操作，避免广播
    diff_squared = density_diff * density_diff  # (E,)
    weighted_diff = edge_weights_flat * diff_squared  # (E,)

    # 求和并应用权重
    lap_loss = lambda_lap * weighted_diff.sum()

    return lap_loss
