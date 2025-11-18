"""
正则化损失函数（Regulation Losses）

包含 Total Variation (TV) 等正则化损失函数，用于约束 K-Planes 等特征网格的平滑性。
改编自 X²-Gaussian 论文的正则化策略。

作者：Claude Code Agent
日期：2025-01-18
"""

import torch
import torch.nn as nn
from typing import List


def compute_plane_tv(
    plane: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    计算单个平面的 Total Variation (TV) 损失

    Total Variation 鼓励相邻像素之间的特征平滑，防止过拟合和伪影。

    公式：
        TV(P) = Σ |P[i+1,j] - P[i,j]| + |P[i,j+1] - P[i,j]|
        （L1 版本）

    参数：
        plane (torch.Tensor): 特征平面，形状 [1, C, H, W] 或 [C, H, W]
        loss_type (str): 损失类型，"l1" 或 "l2"（默认 "l1"）

    返回：
        torch.Tensor: 标量损失值
    """
    # 确保 plane 是 4D 张量
    if plane.dim() == 3:
        plane = plane.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

    assert plane.dim() == 4, f"期望 plane 是 4D 张量，但得到 {plane.dim()}D"

    # 计算水平梯度（相邻列之间的差异）
    # plane[:, :, :, 1:] - plane[:, :, :, :-1]
    # 形状：[1, C, H, W-1]
    grad_horizontal = plane[:, :, :, 1:] - plane[:, :, :, :-1]

    # 计算垂直梯度（相邻行之间的差异）
    # plane[:, :, 1:, :] - plane[:, :, :-1, :]
    # 形状：[1, C, H-1, W]
    grad_vertical = plane[:, :, 1:, :] - plane[:, :, :-1, :]

    # 根据损失类型计算范数
    if loss_type == "l1":
        tv_loss = grad_horizontal.abs().mean() + grad_vertical.abs().mean()
    elif loss_type == "l2":
        tv_loss = grad_horizontal.pow(2).mean() + grad_vertical.pow(2).mean()
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type}，请选择 'l1' 或 'l2'")

    return tv_loss


def compute_plane_tv_loss(
    planes: List[torch.Tensor],
    weights: List[float],
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    计算所有平面的加权 Total Variation 损失

    参数：
        planes (List[torch.Tensor]): 平面列表，每个形状 [1, C, H, W]
            例如：[plane_xy, plane_xz, plane_yz]
        weights (List[float]): 每个平面的权重列表，长度与 planes 相同
            例如：[0.0001, 0.0001, 0.0001]
        loss_type (str): 损失类型，"l1" 或 "l2"（默认 "l1"）

    返回：
        torch.Tensor: 加权 TV 损失，标量

    公式：
        Total_TV_Loss = Σ weights[i] * TV(planes[i])
    """
    assert len(planes) == len(weights), \
        f"planes 和 weights 的长度必须相同，但得到 {len(planes)} vs {len(weights)}"

    total_loss = torch.tensor(0.0, device=planes[0].device, dtype=planes[0].dtype)

    for plane, weight in zip(planes, weights):
        if weight > 0:  # 仅在权重 > 0 时计算（节省计算）
            tv_loss = compute_plane_tv(plane, loss_type)
            total_loss += weight * tv_loss

    return total_loss


def compute_l1_time_planes(
    planes_time: List[torch.Tensor],
    weights: List[float],
) -> torch.Tensor:
    """
    计算时间平面的 L1 稀疏正则化（仅用于动态场景，静态场景不需要）

    参数：
        planes_time (List[torch.Tensor]): 时间平面列表
        weights (List[float]): 权重列表

    返回：
        torch.Tensor: L1 损失
    """
    # 此函数仅为兼容性保留，R²-GS 静态场景不使用
    total_loss = torch.tensor(0.0)
    for plane, weight in zip(planes_time, weights):
        if weight > 0:
            total_loss += weight * plane.abs().mean()
    return total_loss


def compute_time_smoothness(
    planes_time: List[torch.Tensor],
    weights: List[float],
) -> torch.Tensor:
    """
    计算时间平面的时间平滑性损失（仅用于动态场景，静态场景不需要）

    鼓励时间维度的平滑变化，避免剧烈跳变。

    参数：
        planes_time (List[torch.Tensor]): 时间平面列表，形状 [1, C, T, H/W]
        weights (List[float]): 权重列表

    返回：
        torch.Tensor: 时间平滑性损失
    """
    # 此函数仅为兼容性保留，R²-GS 静态场景不使用
    total_loss = torch.tensor(0.0)
    for plane, weight in zip(planes_time, weights):
        if weight > 0 and plane.shape[2] > 1:  # 时间维度 > 1
            # 计算相邻时间步的差异
            time_diff = plane[:, :, 1:, :] - plane[:, :, :-1, :]
            total_loss += weight * time_diff.abs().mean()
    return total_loss


# 测试代码（可选）
if __name__ == "__main__":
    print("Testing regulation losses...")

    # 创建测试平面
    plane_xy = torch.randn(1, 32, 64, 64, requires_grad=True).cuda()
    plane_xz = torch.randn(1, 32, 64, 64, requires_grad=True).cuda()
    plane_yz = torch.randn(1, 32, 64, 64, requires_grad=True).cuda()

    # 测试单个平面的 TV 损失
    tv_loss_single = compute_plane_tv(plane_xy, loss_type="l1")
    print(f"Single plane TV loss (L1): {tv_loss_single.item():.6f}")

    tv_loss_single_l2 = compute_plane_tv(plane_xy, loss_type="l2")
    print(f"Single plane TV loss (L2): {tv_loss_single_l2.item():.6f}")

    # 测试多个平面的加权 TV 损失
    planes = [plane_xy, plane_xz, plane_yz]
    weights = [0.0001, 0.0001, 0.0001]

    tv_loss_total = compute_plane_tv_loss(planes, weights, loss_type="l1")
    print(f"Total weighted TV loss: {tv_loss_total.item():.8f}")

    # 测试梯度反向传播
    tv_loss_total.backward()
    print(f"Gradient computed: plane_xy.grad is not None = {plane_xy.grad is not None}")

    # 测试边界情况
    # 1. 权重为 0
    weights_zero = [0.0, 0.0, 0.0]
    tv_loss_zero = compute_plane_tv_loss(planes, weights_zero)
    print(f"TV loss with zero weights: {tv_loss_zero.item():.8f}")
    assert tv_loss_zero.item() == 0.0, "零权重应返回零损失"

    # 2. 单个平面
    tv_loss_one = compute_plane_tv_loss([plane_xy], [1.0])
    print(f"TV loss for single plane: {tv_loss_one.item():.6f}")

    print("All regulation tests passed!")
