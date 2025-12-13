#
# Volume rendering for NeRF-based methods
#

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple


def generate_rays_from_camera(
    viewpoint_camera,
    scanner_cfg: dict,
    n_rays: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 R²-Gaussian 相机生成射线

    R²-Gaussian 已对场景进行归一化 (sVoxel=[2,2,2], 场景在 [-1,1] 范围)
    相机位于 DSO 距离处 (如 7.8125)，看向原点

    Args:
        viewpoint_camera: R²-Gaussian Camera 对象
        scanner_cfg: 扫描仪配置
        n_rays: 采样射线数量（None 表示全部像素）

    Returns:
        rays: [N_rays, 8] (rays_o, rays_d, near, far)
        indices: [N_rays, 2] (i, j) 像素索引（用于与 GT 对比）
    """
    H = viewpoint_camera.image_height
    W = viewpoint_camera.image_width
    mode = viewpoint_camera.mode
    device = "cuda"

    # 从 scanner_cfg 获取几何参数
    # R²-Gaussian 已经归一化场景: sVoxel=[2,2,2], 场景在 [-1,1] 范围
    sVoxel = scanner_cfg["sVoxel"]
    DSO = scanner_cfg["DSO"]  # 源到原点距离 (已归一化)
    offOrigin = scanner_cfg.get("offOrigin", [0.0, 0.0, 0.0])

    # 使用原始 SAX-NeRF 的 near/far 计算方式
    # 计算场景在 xy 平面上的最大距离
    tolerance = 0.005
    dist1 = math.sqrt((offOrigin[0] - sVoxel[0] / 2)**2 + (offOrigin[1] - sVoxel[1] / 2)**2)
    dist2 = math.sqrt((offOrigin[0] - sVoxel[0] / 2)**2 + (offOrigin[1] + sVoxel[1] / 2)**2)
    dist3 = math.sqrt((offOrigin[0] + sVoxel[0] / 2)**2 + (offOrigin[1] - sVoxel[1] / 2)**2)
    dist4 = math.sqrt((offOrigin[0] + sVoxel[0] / 2)**2 + (offOrigin[1] + sVoxel[1] / 2)**2)
    dist_max = max(dist1, dist2, dist3, dist4)

    near_dist = max(0.0, DSO - dist_max - tolerance)
    far_dist = min(DSO * 2, DSO + dist_max + tolerance)

    # 生成像素网格
    i, j = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing='ij'
    )

    if mode == 0:
        # 平行投影 (parallel beam CT)
        # 像素坐标归一化到 [-1, 1]
        u = (j - W / 2) / W * 2
        v = (i - H / 2) / H * 2

        # 获取相机参数
        R = viewpoint_camera.world_view_transform[:3, :3].T  # 相机到世界旋转
        camera_center = viewpoint_camera.camera_center

        # 平行投影的射线方向是相机的 z 轴方向 (看向原点)
        # 对于平行投影，所有射线方向相同
        forward = -camera_center / torch.norm(camera_center)  # 指向原点
        rays_d = forward.unsqueeze(0).expand(H * W, -1)  # [H*W, 3]

        # 射线起点随像素位置变化
        # 需要根据探测器尺寸缩放
        dDetector = scanner_cfg.get("dDetector", [1.0, 1.0])
        sDetector = scanner_cfg.get("sDetector", [2.0, 2.0])

        # 在探测器平面上的偏移
        offset_x = u.reshape(-1, 1) * sDetector[0] / 2
        offset_y = v.reshape(-1, 1) * sDetector[1] / 2

        # 获取相机坐标系的 right 和 up 向量
        right = R[0:1, :]  # x 轴
        up = R[1:2, :]     # y 轴

        # 射线起点 = 相机位置 + 探测器平面偏移
        rays_o = camera_center.unsqueeze(0) + offset_x * right + offset_y * up

        # near/far 设置
        near = torch.ones(H * W, 1, device=device) * near_dist
        far = torch.ones(H * W, 1, device=device) * far_dist

    elif mode == 1:
        # 透视/锥形投影 (cone beam CT)
        # 使用原始 SAX-NeRF 的射线生成方式
        DSD = scanner_cfg.get("DSD", DSO * 1.5)  # 源到探测器距离
        dDetector = scanner_cfg.get("dDetector", [1.0, 1.0])
        offDetector = scanner_cfg.get("offDetector", [0.0, 0.0])

        # 像素坐标转探测器坐标 (与 SAX-NeRF tigre.py 一致)
        # uu, vv 是探测器平面上相对于中心的偏移
        uu = (j + 0.5 - W / 2) * dDetector[0] + offDetector[0]
        vv = (i + 0.5 - H / 2) * dDetector[1] + offDetector[1]

        # 射线方向（相机坐标系，z 轴指向前方）
        # dirs = [uu/DSD, vv/DSD, 1] 然后归一化
        dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], dim=-1)  # [H, W, 3]

        # c2w 旋转矩阵: world_view_transform[:3, :3] 就是 c2w 旋转
        # 注意: R²-Gaussian 存储方式使得 W2C[:3, :3] = c2w[:3, :3]
        c2w_rot = viewpoint_camera.world_view_transform[:3, :3]  # [3, 3]

        # 转换到世界坐标系: rays_d = c2w_rot @ dirs
        rays_d = torch.matmul(dirs.reshape(-1, 3), c2w_rot.T)  # [H*W, 3]

        # 射线起点 = 相机中心 (所有射线从同一点出发)
        rays_o = viewpoint_camera.camera_center.unsqueeze(0).expand(H * W, -1)

        # near/far 基于相机到场景的距离
        near = torch.ones(H * W, 1, device=device) * near_dist
        far = torch.ones(H * W, 1, device=device) * far_dist

    else:
        raise ValueError(f"Unsupported camera mode: {mode}")

    # 组合射线 (坐标已归一化)
    rays = torch.cat([rays_o, rays_d, near, far], dim=-1)  # [H*W, 8]
    indices = torch.stack([i.reshape(-1), j.reshape(-1)], dim=-1).long()  # [H*W, 2]

    # 随机采样
    if n_rays is not None and n_rays < H * W:
        select_inds = torch.randperm(H * W, device=device)[:n_rays]
        rays = rays[select_inds]
        indices = indices[select_inds]

    return rays, indices


def render_rays(
    rays: torch.Tensor,
    net: nn.Module,
    net_fine: Optional[nn.Module],
    n_samples: int,
    n_fine: int = 0,
    perturb: bool = True,
    netchunk: int = 409600,
    raw_noise_std: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    射线渲染（体积积分）

    Args:
        rays: 射线 [N_rays, 8] (rays_o, rays_d, near, far)
        net: 粗网络
        net_fine: 精细网络 (可选)
        n_samples: 粗采样点数
        n_fine: 精细采样点数
        perturb: 是否扰动采样
        netchunk: 网络批处理大小
        raw_noise_std: 密度噪声标准差

    Returns:
        dict:
            - acc: 累积密度 (投影图像) [N_rays, 1]
            - pts: 采样点 [N_rays, N_samples, 3]
            - raw: 原始密度输出
    """
    n_rays = rays.shape[0]

    # 解析射线
    rays_o = rays[..., :3]
    rays_d = rays[..., 3:6]
    near = rays[..., 6:7]
    far = rays[..., 7:]

    # 粗采样
    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=rays.device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand([n_rays, n_samples])

    # 扰动采样
    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand

    # 计算采样点
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    bound = net.bound - 1e-6
    pts = pts.clamp(-bound, bound)

    # 网络查询
    raw = run_network(pts, net, netchunk)
    acc, weights = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    # 精细采样
    if net_fine is not None and n_fine > 0:
        acc_0 = acc
        pts_0 = pts

        # 重要性采样
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_fine, det=(not perturb))
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.clamp(-bound, bound)

        raw = run_network(pts, net_fine, netchunk)
        acc, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std)

    return {
        "acc": acc,
        "pts": pts,
        "raw": raw,
    }


def run_network(
    inputs: torch.Tensor,
    fn: nn.Module,
    netchunk: int,
) -> torch.Tensor:
    """
    分批运行网络

    Args:
        inputs: 输入点 [N_rays, N_samples, 3]
        fn: 网络
        netchunk: 批处理大小

    Returns:
        outputs: 网络输出 [N_rays, N_samples, out_dim]
    """
    # 展平
    uvt_flat = inputs.reshape(-1, inputs.shape[-1])

    # 分批处理
    outputs = []
    for i in range(0, uvt_flat.shape[0], netchunk):
        outputs.append(fn(uvt_flat[i:i + netchunk]))
    out_flat = torch.cat(outputs, dim=0)

    # 恢复形状
    out = out_flat.reshape(*inputs.shape[:-1], -1)
    return out


def raw2outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将网络输出转换为 X-ray 投影

    Args:
        raw: 网络输出 (密度) [N_rays, N_samples, 1]
        z_vals: 采样深度 [N_rays, N_samples]
        rays_d: 射线方向 [N_rays, 3]
        raw_noise_std: 密度噪声标准差

    Returns:
        acc: 累积密度 (X-ray 投影) [N_rays, 1]
        weights: 权重 [N_rays, N_samples]
    """
    # 计算采样间隔
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, dists[..., -1:]], dim=-1)

    # 考虑射线方向长度
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # 密度噪声
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn_like(raw[..., 0]) * raw_noise_std

    # X-ray 衰减系数
    # raw 已经是 sigmoid 输出 [0, 1]
    alpha = raw[..., 0]

    # 计算累积密度（X-ray 投影是密度沿射线的积分）
    # acc = sum(alpha * dists)
    acc = torch.sum(alpha * dists, dim=-1, keepdim=True)

    # 权重用于重要性采样
    weights = alpha * dists
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)

    return acc, weights


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
) -> torch.Tensor:
    """
    重要性采样

    Args:
        bins: 采样区间 [N_rays, N_bins]
        weights: 权重 [N_rays, N_bins]
        N_samples: 采样数量
        det: 是否确定性采样

    Returns:
        samples: 采样深度 [N_rays, N_samples]
    """
    # 归一化权重
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # 采样
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=weights.device)

    # 反向查找
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], dim=-1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
