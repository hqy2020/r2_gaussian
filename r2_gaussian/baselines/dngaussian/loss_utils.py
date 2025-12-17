#
# DNGaussian 深度正则化损失函数
#
# 从 DNGaussian 移植并适配 CT 重建场景
#

import torch
import torch.nn.functional as F


def normalize(input, mean=None, std=None):
    """
    对 patch 进行归一化

    Args:
        input: [B, N] 输入张量，B 是 batch，N 是 patch 内像素数
        mean: 可选，指定均值
        std: 可选，指定标准差

    Returns:
        归一化后的张量
    """
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2 * torch.std(input.reshape(-1)))


def patchify(input, patch_size):
    """
    将图像分割成 patches

    Args:
        input: [B, C, H, W] 输入图像
        patch_size: patch 大小

    Returns:
        [N_patches, C * patch_size * patch_size] patches 张量
    """
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size)
    patches = patches.permute(0, 2, 1).reshape(-1, input.shape[1] * patch_size * patch_size)
    return patches


def margin_l2_loss(network_output, gt, margin, return_mask=False):
    """
    带 margin 的 L2 损失（只计算误差大于 margin 的部分）

    Args:
        network_output: 预测值
        gt: 真实值
        margin: 误差容忍度
        return_mask: 是否返回 mask

    Returns:
        L2 损失（标量）
    """
    mask = (network_output - gt).abs() > margin
    if mask.sum() == 0:
        # 没有超过 margin 的误差，返回 0
        if not return_mask:
            return torch.tensor(0.0, device=network_output.device, requires_grad=True)
        else:
            return torch.tensor(0.0, device=network_output.device, requires_grad=True), mask

    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask


def margin_l1_loss(network_output, gt, margin, return_mask=False):
    """
    带 margin 的 L1 损失（只计算误差大于 margin 的部分）
    """
    mask = (network_output - gt).abs() > margin
    if mask.sum() == 0:
        if not return_mask:
            return torch.tensor(0.0, device=network_output.device, requires_grad=True)
        else:
            return torch.tensor(0.0, device=network_output.device, requires_grad=True), mask

    if not return_mask:
        return ((network_output - gt)[mask].abs()).mean()
    else:
        return ((network_output - gt)[mask].abs()).mean(), mask


def patch_norm_mse_loss(input, target, patch_size, margin, return_mask=False):
    """
    局部归一化的 Patch MSE 损失

    每个 patch 独立归一化后计算 MSE

    Args:
        input: [B, C, H, W] 预测图像
        target: [B, C, H, W] 目标图像
        patch_size: patch 大小
        margin: 误差容忍度
        return_mask: 是否返回 mask

    Returns:
        patch norm MSE 损失
    """
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)


def patch_norm_mse_loss_global(input, target, patch_size, margin, return_mask=False):
    """
    全局归一化的 Patch MSE 损失

    使用全局标准差进行归一化，保持全局尺度信息

    Args:
        input: [B, C, H, W] 预测图像
        target: [B, C, H, W] 目标图像
        patch_size: patch 大小
        margin: 误差容忍度
        return_mask: 是否返回 mask

    Returns:
        patch norm MSE 损失（全局归一化）
    """
    input_patches = normalize(patchify(input, patch_size), std=input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std=target.std().detach())
    return margin_l2_loss(input_patches, target_patches, margin, return_mask)


def patch_norm_l1_loss_global(input, target, patch_size, margin, return_mask=False):
    """
    全局归一化的 Patch L1 损失
    """
    input_patches = normalize(patchify(input, patch_size), std=input.std().detach())
    target_patches = normalize(patchify(target, patch_size), std=target.std().detach())
    return margin_l1_loss(input_patches, target_patches, margin, return_mask)


def compute_projection_depth_prior(projection, method='inverse'):
    """
    从 X-ray 投影计算伪深度先验

    CT X-ray 投影是沿射线的衰减积分。我们可以从中推导出"等效深度"：
    - 投影值越大（衰减越强）→ 物体越"厚"或密度越高
    - 可以用投影值的反函数作为深度的代理

    Args:
        projection: [B, 1, H, W] X-ray 投影图像
        method: 深度计算方法
            - 'inverse': 1 - proj/max(proj)，投影越亮→深度越浅
            - 'log': -log(proj + eps)，对数变换增强对比度
            - 'linear': proj/max(proj)，直接线性映射

    Returns:
        [B, 1, H, W] 伪深度图
    """
    eps = 1e-6

    if method == 'inverse':
        # 投影越亮（衰减越大）→ 深度越浅（更接近相机）
        proj_max = projection.max()
        if proj_max > eps:
            depth = 1.0 - projection / proj_max
        else:
            depth = torch.zeros_like(projection)

    elif method == 'log':
        # 对数变换，增强对比度
        # 归一化到 [eps, 1] 范围
        proj_min = projection.min()
        proj_max = projection.max()
        if proj_max - proj_min > eps:
            proj_norm = (projection - proj_min) / (proj_max - proj_min)
            proj_norm = torch.clamp(proj_norm, eps, 1.0)
            depth = -torch.log(proj_norm)
            # 归一化到 [0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + eps)
        else:
            depth = torch.zeros_like(projection)

    elif method == 'linear':
        # 直接线性映射
        proj_max = projection.max()
        if proj_max > eps:
            depth = projection / proj_max
        else:
            depth = torch.zeros_like(projection)

    else:
        raise ValueError(f"Unknown depth prior method: {method}")

    return depth


def depth_smoothness_loss(depth, image):
    """
    深度平滑损失（边缘感知）

    在图像边缘处允许深度不连续，在平滑区域强制深度平滑

    Args:
        depth: [B, 1, H, W] 深度图
        image: [B, C, H, W] RGB/灰度图像

    Returns:
        平滑损失（标量）
    """
    # 图像梯度
    img_grad_x = image[:, :, :, :-1] - image[:, :, :, 1:]
    img_grad_y = image[:, :, :-1, :] - image[:, :, 1:, :]

    # 边缘感知权重：图像梯度大的地方权重小
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1, keepdim=True))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1, keepdim=True))

    # 深度梯度
    depth_grad_x = depth[:, :, :, :-1] - depth[:, :, :, 1:]
    depth_grad_y = depth[:, :, :-1, :] - depth[:, :, 1:, :]

    # 加权平滑损失
    loss = (
        (depth_grad_x.abs() * weight_x).sum() +
        (depth_grad_y.abs() * weight_y).sum()
    ) / (weight_x.sum() + weight_y.sum() + 1e-6)

    return loss
