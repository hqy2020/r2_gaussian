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


def compute_graph_laplacian_loss(gaussians, k=6, Lambda_lap=8e-4):
    """
    图拉普拉斯正则化损失 - 参考CoR-GS/GR-Gaussian论文
    鼓励相邻高斯点的密度平滑，与depth约束互补
    
    GPU加速版本（带自动回退到CPU）：
    - 优先使用GPU加速计算（torch.cdist + topk）
    - 如果GPU内存不足或出错，自动回退到CPU版本（sklearn）
    
    Args:
        gaussians: GaussianModel实例
        k: KNN邻居数量（默认6，根据CoR-GS论文）
        Lambda_lap: 正则化权重（默认8e-4，根据CoR-GS论文）
    Returns:
        loss: 标量损失值
    """
    import torch
    
    # 获取高斯点位置和密度
    xyz = gaussians.get_xyz  # (N, 3)
    density = gaussians.get_density  # (N,)
    
    N = xyz.shape[0]
    if N < k + 1:
        return torch.tensor(0.0, device=xyz.device, requires_grad=True)
    
    # 尝试GPU加速版本（优先）
    try:
        # 检查点数量，避免GPU内存溢出
        # 如果点数过多（>10万），使用分批处理或回退到CPU
        max_gpu_points = 100000
        if N > max_gpu_points:
            raise RuntimeError(f"Too many points ({N}) for GPU computation, using CPU fallback")
        
        # GPU加速KNN搜索：使用torch.cdist + topk（完全在GPU上）
        # 计算所有点对之间的欧氏距离
        dists = torch.cdist(xyz, xyz)  # (N, N) - GPU并行计算
        
        # 获取每个点的k+1个最近邻（包括自己）
        _, indices = torch.topk(dists, k+1, dim=1, largest=False)  # (N, k+1) - GPU计算
        
        # 跳过自己（第一个邻居是自己，距离为0）
        neighbor_indices = indices[:, 1:]  # (N, k) - 获取k个真实邻居
        
        # 批量获取邻居距离（向量化操作）
        batch_indices = torch.arange(N, device=xyz.device).unsqueeze(1).expand(-1, k)  # (N, k)
        neighbor_dists = dists[batch_indices, neighbor_indices]  # (N, k) - 邻居距离
        
        # 计算权重（高斯核：距离越近权重越大）
        neighbor_dists_mean = neighbor_dists.mean(dim=1, keepdim=True)  # (N, 1)
        weights = torch.exp(-neighbor_dists / (neighbor_dists_mean + 1e-7))  # (N, k)
        
        # 批量计算密度差异（向量化）
        density_expanded = density.unsqueeze(1)  # (N, 1)
        density_neighbors = density[neighbor_indices]  # (N, k) - 批量获取邻居密度
        density_diff = density_expanded - density_neighbors  # (N, k)
        
        # 加权平方差（向量化计算）
        weighted_loss = weights * (density_diff ** 2)  # (N, k)
        
        # 平均损失并乘以权重
        loss = weighted_loss.mean() * Lambda_lap
        
        return loss
        
    except RuntimeError as e:
        # GPU计算失败（内存不足或其他CUDA错误），回退到CPU版本
        # 注意：CUDA的OOM错误也会抛出RuntimeError，所以只捕获RuntimeError即可
        # 只在第一次失败时打印警告，避免日志过多
        import warnings
        if not hasattr(compute_graph_laplacian_loss, '_cpu_fallback_warned'):
            warnings.warn(f"GPU computation failed ({str(e)}), falling back to CPU version. "
                        f"This may be slower but will work correctly.")
            compute_graph_laplacian_loss._cpu_fallback_warned = True
        
        # CPU回退版本（原始实现）
        from sklearn.neighbors import NearestNeighbors
        
        device = xyz.device
        
        # 构建KNN图（CPU计算）
        xyz_np = xyz.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(xyz_np)
        distances, indices = nbrs.kneighbors(xyz_np)
        
        # 计算拉普拉斯损失：L = sum_{i,j} w_{ij} * (d_i - d_j)^2
        total_loss = 0.0
        count = 0
        
        for i in range(len(xyz_np)):
            # 获取k个邻居（跳过自己）
            neighbors = indices[i][1:]
            neighbor_dists = distances[i][1:]
            
            if len(neighbors) == 0:
                continue
                
            # 计算权重（高斯核：距离越近权重越大）
            weights = torch.exp(-torch.tensor(neighbor_dists, device=device) / (neighbor_dists.mean() + 1e-7))
            
            # 计算密度差异
            density_i = density[i]
            density_neighbors = density[neighbors]
            density_diff = density_i - density_neighbors
            
            # 加权平方差
            weighted_loss = weights * (density_diff ** 2)
            total_loss += weighted_loss.sum()
            count += len(neighbors)
        
        if count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 平均损失并乘以权重
        loss = (total_loss / count) * Lambda_lap
        return loss
