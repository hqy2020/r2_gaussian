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

# 添加torchmetrics依赖用于深度相关性计算
try:
    from torchmetrics.functional.regression import pearson_corrcoef
except ImportError:
    # 如果torchmetrics不可用，使用PyTorch内置函数
    def pearson_corrcoef(pred, target):
        """简单的皮尔逊相关系数计算"""
        pred_mean = pred.mean()
        target_mean = target.mean()
        numerator = ((pred - pred_mean) * (target - target_mean)).sum()
        denominator = torch.sqrt(((pred - pred_mean) ** 2).sum() * ((target - target_mean) ** 2).sum())
        return numerator / (denominator + 1e-8)


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


def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    ssim_value = _ssim(img1, img2, window, window_size, channel, size_average)
    
    if mask is not None:
        # Apply mask to SSIM calculation if provided
        ssim_value = ssim_value * mask.mean()
    
    return ssim_value


def loss_photometric(image, gt_image, opt, valid=None):
    """Photometric loss with mask support - 参考X-Gaussian实现"""
    Ll1 = l1_loss_mask(image, gt_image, mask=valid)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=valid)))
    return Ll1, loss


def l1_loss_mask(network_output, gt, mask=None):
    """L1 loss with mask support - 参考X-Gaussian实现"""
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()


def depth_loss(predicted_depth, gt_depth, depth_bounds=None):
    """Depth consistency loss - 新增深度损失函数"""
    # 深度一致性损失
    depth_consistency = torch.abs(predicted_depth - gt_depth)
    
    if depth_bounds is not None:
        min_depth, max_depth = depth_bounds
        # 深度范围惩罚
        depth_range_penalty = torch.where(
            (predicted_depth < min_depth) | (predicted_depth > max_depth),
            torch.abs(predicted_depth - torch.clamp(predicted_depth, min_depth, max_depth)),
            torch.zeros_like(predicted_depth)
        )
        return depth_consistency.mean() + 0.1 * depth_range_penalty.mean()
    
    return depth_consistency.mean()


def pseudo_label_loss(predicted_image, pseudo_label, confidence_mask=None):
    """Pseudo label loss - 伪标签损失函数"""
    if confidence_mask is not None:
        # 使用置信度mask加权
        loss = torch.abs(predicted_image - pseudo_label) * confidence_mask
        return loss.sum() / confidence_mask.sum()
    else:
        return l1_loss(predicted_image, pseudo_label)


def calculate_depth_loss(rendered_depth, gt_depth):
    """计算深度相关性损失 - 参考X-Gaussian-depth实现"""
    rendered_depth = rendered_depth.reshape(-1, 1)
    gt_depth = gt_depth.reshape(-1, 1)
    
    # 使用X-Gaussian-depth中的深度损失计算方式
    depth_loss = min(
        (1 - pearson_corrcoef(-gt_depth.squeeze(), rendered_depth.squeeze())),
        (1 - pearson_corrcoef((1 / (gt_depth + 200.)).squeeze(), rendered_depth.squeeze()))
    )
    return depth_loss


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


def depth_loss_fn(rendered_depth, gt_depth, loss_type='pearson'):
    """
    计算深度损失函数
    
    Args:
        rendered_depth: 渲染的深度图 (H, W)
        gt_depth: 真实深度图 (H, W)
        loss_type: 损失类型 ('l1', 'l2', 'pearson')
    
    Returns:
        loss: 深度损失值
    """
    if rendered_depth is None or gt_depth is None:
        return torch.tensor(0.0, device=rendered_depth.device if rendered_depth is not None else gt_depth.device)
    
    # 确保两个张量在同一设备上
    if rendered_depth.device != gt_depth.device:
        gt_depth = gt_depth.to(rendered_depth.device)
    
    # 确保形状一致
    if rendered_depth.shape != gt_depth.shape:
        # 如果形状不同，尝试调整大小
        if len(rendered_depth.shape) == 2 and len(gt_depth.shape) == 2:
            gt_depth = F.interpolate(gt_depth.unsqueeze(0).unsqueeze(0), 
                                    size=rendered_depth.shape, 
                                    mode='bilinear', 
                                    align_corners=False).squeeze(0).squeeze(0)
        else:
            raise ValueError(f"Shape mismatch: {rendered_depth.shape} vs {gt_depth.shape}")
    
    if loss_type == 'l1':
        return F.l1_loss(rendered_depth, gt_depth)
    elif loss_type == 'l2':
        return F.mse_loss(rendered_depth, gt_depth)
    elif loss_type == 'pearson':
        # 使用皮尔逊相关系数作为损失
        correlation = pearson_corrcoef(rendered_depth.flatten(), gt_depth.flatten())
        return 1.0 - correlation  # 转换为损失（越小越好）
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def depth_consistency_loss(depth_maps):
    """
    计算多视角深度一致性损失
    
    Args:
        depth_maps: 多个视角的深度图列表
    
    Returns:
        consistency_loss: 一致性损失
    """
    if len(depth_maps) < 2:
        return torch.tensor(0.0, device=depth_maps[0].device)
    
    # 简化的实现：计算相邻视角深度图的差异
    total_loss = 0.0
    for i in range(len(depth_maps) - 1):
        diff = torch.abs(depth_maps[i] - depth_maps[i + 1])
        total_loss += diff.mean()
    
    return total_loss / (len(depth_maps) - 1)
