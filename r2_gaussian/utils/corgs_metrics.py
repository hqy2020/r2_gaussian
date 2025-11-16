"""
CoR-GS (Co-Regularization Gaussian Splatting) Metrics

实现 Point Disagreement 和 Rendering Disagreement 计算,用于衡量双模型差异。

基于论文: CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
https://jiaw-z.github.io/CoR-GS

实现日期: 2025-11-16
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

# PyTorch3D 加速 KNN (可选依赖)
try:
    from pytorch3d.ops import knn_points
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False


def compute_point_disagreement_pytorch3d(
    gaussians_1_xyz: torch.Tensor,
    gaussians_2_xyz: torch.Tensor,
    threshold: float = 0.3,
    max_points: int = 100000
) -> Tuple[float, float]:
    """
    使用 PyTorch3D 的 CUDA 加速 KNN 计算 Point Disagreement

    性能优势:
        - 10-100 倍速度提升 (相比原生 PyTorch cdist)
        - 不存储完整距离矩阵,内存效率高
        - 支持百万级点云处理
        - GPU 优化的 KNN 算法实现

    Args:
        gaussians_1_xyz: 第一个模型的 Gaussian 中心位置 [N1, 3]
        gaussians_2_xyz: 第二个模型的 Gaussian 中心位置 [N2, 3]
        threshold: 匹配距离阈值 (CT 场景建议 0.1~0.5)
        max_points: 最大处理点数 (避免显存不足,超过则随机采样)

    Returns:
        fitness: 匹配点比例 [0, 1], 越高表示两模型越一致
        rmse: 匹配点的均方根误差, 越低表示匹配越精确

    技术细节:
        - 使用 pytorch3d.ops.knn_points (CUDA 加速)
        - K=1 表示只找最近邻
        - 返回平方距离,需手动开方
    """
    print(f"[DEBUG-KNN-FAST-1] Using PyTorch3D KNN: N1={gaussians_1_xyz.shape[0]}, N2={gaussians_2_xyz.shape[0]}", flush=True)

    N1 = gaussians_1_xyz.shape[0]
    N2 = gaussians_2_xyz.shape[0]

    # 采样(如果需要)
    if N1 > max_points:
        print(f"[DEBUG-KNN-FAST-2] Sampling N1 from {N1} to {max_points}", flush=True)
        indices = torch.randperm(N1, device=gaussians_1_xyz.device)[:max_points]
        gaussians_1_xyz = gaussians_1_xyz[indices]
        N1 = max_points

    if N2 > max_points:
        print(f"[DEBUG-KNN-FAST-3] Sampling N2 from {N2} to {max_points}", flush=True)
        indices = torch.randperm(N2, device=gaussians_2_xyz.device)[:max_points]
        gaussians_2_xyz = gaussians_2_xyz[indices]
        N2 = max_points

    # PyTorch3D 需要 [B, N, 3] 格式
    xyz_1_batch = gaussians_1_xyz.unsqueeze(0)  # [1, N1, 3]
    xyz_2_batch = gaussians_2_xyz.unsqueeze(0)  # [1, N2, 3]

    print("[DEBUG-KNN-FAST-4] Computing KNN with PyTorch3D", flush=True)
    # K=1 表示只找最近邻
    # return_nn=False 表示不返回最近邻点坐标,只返回距离
    knn_result = knn_points(xyz_1_batch, xyz_2_batch, K=1, return_nn=False)

    # knn_result.dists: [1, N1, 1] 平方距离
    min_distances_sq = knn_result.dists.squeeze()  # [N1]
    min_distances = torch.sqrt(min_distances_sq)   # 转为欧式距离

    print("[DEBUG-KNN-FAST-5] Computing fitness and RMSE", flush=True)
    matched_mask = min_distances < threshold
    fitness = matched_mask.float().mean().item()

    if matched_mask.sum() > 0:
        rmse = min_distances[matched_mask].pow(2).mean().sqrt().item()
    else:
        rmse = float('inf')

    print(f"[DEBUG-KNN-FAST-6] KNN done: fitness={fitness:.4f}, rmse={rmse:.6f}", flush=True)
    return fitness, rmse


def compute_point_disagreement(
    gaussians_1_xyz: torch.Tensor,
    gaussians_2_xyz: torch.Tensor,
    threshold: float = 0.3,
    max_points: int = 100000
) -> Tuple[float, float]:
    """
    计算两个 Gaussian 点云的 Point Disagreement

    使用 KNN 匹配找到对应点,计算 Fitness 和 RMSE。
    Fitness 表示匹配点的比例,RMSE 表示匹配点的平均距离。

    Args:
        gaussians_1_xyz: 第一个模型的 Gaussian 中心位置 [N1, 3]
        gaussians_2_xyz: 第二个模型的 Gaussian 中心位置 [N2, 3]
        threshold: 匹配距离阈值 (论文原文 τ=5, CT 场景建议 0.1~0.5)
        max_points: 最大处理点数 (避免显存不足,超过则随机采样)

    Returns:
        fitness: 匹配点比例 [0, 1], 越高表示两模型越一致
        rmse: 匹配点的均方根误差, 越低表示匹配越精确

    实现策略:
        使用 PyTorch 原生操作避免 Open3D 依赖 (300MB)
        对大规模点云进行批处理以节省显存
    """
    print(f"[DEBUG-KNN-1] Input shapes: N1={gaussians_1_xyz.shape[0]}, N2={gaussians_2_xyz.shape[0]}", flush=True)
    N1 = gaussians_1_xyz.shape[0]
    N2 = gaussians_2_xyz.shape[0]

    # 如果点数过多,随机采样以节省显存
    if N1 > max_points:
        print(f"[DEBUG-KNN-2] Sampling N1 from {N1} to {max_points}", flush=True)
        indices = torch.randperm(N1, device=gaussians_1_xyz.device)[:max_points]
        gaussians_1_xyz = gaussians_1_xyz[indices]
        N1 = max_points

    if N2 > max_points:
        print(f"[DEBUG-KNN-3] Sampling N2 from {N2} to {max_points}", flush=True)
        indices = torch.randperm(N2, device=gaussians_2_xyz.device)[:max_points]
        gaussians_2_xyz = gaussians_2_xyz[indices]
        N2 = max_points

    # 计算最近邻距离 (方向: gaussians_1 → gaussians_2)
    # 分批计算以节省显存
    batch_size = 10000
    min_distances_list = []

    print(f"[DEBUG-KNN-4] Starting KNN batched computation (batch_size={batch_size})", flush=True)
    total_batches = (N1 + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, N1, batch_size)):
        batch_xyz = gaussians_1_xyz[i:i+batch_size]  # [batch, 3]
        # 计算距离矩阵 [batch, N2]
        distances = torch.cdist(batch_xyz, gaussians_2_xyz, p=2)
        # 找最近邻距离 [batch]
        min_dists, _ = distances.min(dim=1)
        min_distances_list.append(min_dists)

        if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
            print(f"[DEBUG-KNN-5] Processed batch {batch_idx+1}/{total_batches}", flush=True)

    print("[DEBUG-KNN-6] Concatenating results", flush=True)
    min_distances = torch.cat(min_distances_list, dim=0)  # [N1]

    # 计算 Fitness: 距离 < threshold 的点的比例
    print("[DEBUG-KNN-7] Computing fitness and RMSE", flush=True)
    matched_mask = min_distances < threshold
    fitness = matched_mask.float().mean().item()

    # 计算 RMSE: 匹配点的均方根误差
    if matched_mask.sum() > 0:
        rmse = min_distances[matched_mask].pow(2).mean().sqrt().item()
    else:
        rmse = float('inf')  # 无匹配点

    print(f"[DEBUG-KNN-8] KNN computation done", flush=True)
    return fitness, rmse


def compute_rendering_disagreement(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    epsilon: float = 1e-8
) -> float:
    """
    计算两个渲染图像的 PSNR 差异

    Args:
        image_1: 第一个模型渲染的图像 [C, H, W] 或 [H, W, C], 值域 [0, 1]
        image_2: 第二个模型渲染的图像 [C, H, W] 或 [H, W, C], 值域 [0, 1]
        epsilon: 数值稳定性常数

    Returns:
        psnr: 峰值信噪比 (dB), 越高表示两图像越相似
              论文观察: 双模型 PSNR 差异随训练降低,在 densification 时激增

    实现:
        PSNR = 10 * log10(1.0 / MSE)
        MSE = mean((image_1 - image_2)^2)
    """
    # 统一形状为 [C, H, W]
    if image_1.ndim == 3 and image_1.shape[-1] in [1, 3]:
        image_1 = image_1.permute(2, 0, 1)  # [H, W, C] → [C, H, W]
    if image_2.ndim == 3 and image_2.shape[-1] in [1, 3]:
        image_2 = image_2.permute(2, 0, 1)

    # 计算 MSE
    mse = torch.mean((image_1 - image_2) ** 2)

    # 计算 PSNR (值域假设为 [0, 1])
    psnr = 10.0 * torch.log10(1.0 / (mse + epsilon))

    return psnr.item()


def compute_ssim_disagreement(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> float:
    """
    计算两个渲染图像的 SSIM 差异 (可选,论文未使用)

    Args:
        image_1: 第一个模型渲染的图像 [C, H, W], 值域 [0, 1]
        image_2: 第二个模型渲染的图像 [C, H, W], 值域 [0, 1]
        window_size: SSIM 窗口大小
        size_average: 是否对所有通道求平均

    Returns:
        ssim: 结构相似性指数 [0, 1], 越高表示越相似

    注: 简化实现,完整版可使用 pytorch-msssim 库
    """
    # 简化版 SSIM (仅供参考,实际可用 pytorch-msssim)
    # 这里返回 1 - D-SSIM
    from r2_gaussian.utils.loss_utils import ssim as r2_ssim

    # 统一形状
    if image_1.ndim == 3 and image_1.shape[-1] in [1, 3]:
        image_1 = image_1.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    if image_2.ndim == 3 and image_2.shape[-1] in [1, 3]:
        image_2 = image_2.permute(2, 0, 1).unsqueeze(0)

    ssim_value = r2_ssim(image_1, image_2)
    return ssim_value.item()


def visualize_point_disagreement(
    gaussians_1_xyz: torch.Tensor,
    gaussians_2_xyz: torch.Tensor,
    threshold: float = 0.3,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    可视化 Point Disagreement (生成颜色标记的点云)

    Args:
        gaussians_1_xyz: 第一个模型的点 [N1, 3]
        gaussians_2_xyz: 第二个模型的点 [N2, 3]
        threshold: 匹配阈值
        output_path: 如果提供,保存可视化图像

    Returns:
        colors: 点云颜色标记 [N1, 3], RGB 值 [0, 255]
                绿色 = 匹配点, 红色 = 非匹配点

    用途: 阶段 1 概念验证时生成可视化,辅助调试
    """
    N1 = gaussians_1_xyz.shape[0]

    # 计算最近邻距离
    distances = torch.cdist(gaussians_1_xyz, gaussians_2_xyz, p=2)
    min_distances, _ = distances.min(dim=1)

    # 生成颜色: 绿色 = 匹配, 红色 = 非匹配
    colors = np.zeros((N1, 3), dtype=np.uint8)
    matched = (min_distances < threshold).cpu().numpy()
    colors[matched] = [0, 255, 0]  # 绿色
    colors[~matched] = [255, 0, 0]  # 红色

    if output_path:
        # 保存点云 (简化版,实际可用 Open3D 或 matplotlib)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        xyz = gaussians_1_xyz.cpu().numpy()
        ax.scatter(xyz[matched, 0], xyz[matched, 1], xyz[matched, 2],
                  c='green', s=1, alpha=0.5, label='Matched')
        ax.scatter(xyz[~matched, 0], xyz[~matched, 1], xyz[~matched, 2],
                  c='red', s=1, alpha=0.5, label='Non-matched')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title(f'Point Disagreement (threshold={threshold})')
        plt.savefig(output_path, dpi=150)
        plt.close()

    return colors


# ============================================================================
# 辅助函数: 批量计算 Disagreement (用于 TensorBoard 日志)
# ============================================================================

def log_corgs_metrics(
    gaussians_1,
    gaussians_2,
    test_camera,
    pipe,
    background,
    threshold: float = 0.3,
    device: str = 'cuda'
) -> dict:
    """
    计算所有 CoR-GS 指标并返回字典 (便于 TensorBoard 记录)

    Args:
        gaussians_1: 第一个 GaussianModel 实例
        gaussians_2: 第二个 GaussianModel 实例
        test_camera: 测试相机 (用于渲染 disagreement)
        pipe: Pipeline 参数
        background: 背景颜色
        threshold: Co-pruning 阈值
        device: 设备

    Returns:
        metrics: 包含所有指标的字典
            - point_fitness: float
            - point_rmse: float
            - render_psnr_diff: float
            - render_ssim_diff: float (可选)
    """
    print("[DEBUG-CORGS-6] Starting log_corgs_metrics", flush=True)
    from r2_gaussian.gaussian import render

    metrics = {}

    # 1. Point Disagreement
    print("[DEBUG-CORGS-7] Getting xyz coordinates", flush=True)
    xyz_1 = gaussians_1.get_xyz.detach()
    xyz_2 = gaussians_2.get_xyz.detach()
    print(f"[DEBUG-CORGS-8] Shapes: xyz_1={xyz_1.shape}, xyz_2={xyz_2.shape}", flush=True)

    print("[DEBUG-CORGS-9] Computing point disagreement (KNN)", flush=True)
    if HAS_PYTORCH3D:
        print("[DEBUG-CORGS-9.1] Using PyTorch3D accelerated KNN", flush=True)
        fitness, rmse = compute_point_disagreement_pytorch3d(xyz_1, xyz_2, threshold)
    else:
        print("[DEBUG-CORGS-9.1] Using fallback KNN (slow)", flush=True)
        fitness, rmse = compute_point_disagreement(xyz_1, xyz_2, threshold, max_points=10000)
    print(f"[DEBUG-CORGS-10] Point metrics computed: fitness={fitness:.4f}, rmse={rmse:.6f}", flush=True)
    metrics['point_fitness'] = fitness
    metrics['point_rmse'] = rmse

    # 2. Rendering Disagreement
    print("[DEBUG-CORGS-11] Starting rendering disagreement", flush=True)
    with torch.no_grad():
        print("[DEBUG-CORGS-12] Rendering model 1", flush=True)
        render_pkg_1 = render(test_camera, gaussians_1, pipe, scaling_modifier=1.0)
        print("[DEBUG-CORGS-13] Rendering model 2", flush=True)
        render_pkg_2 = render(test_camera, gaussians_2, pipe, scaling_modifier=1.0)

        print("[DEBUG-CORGS-14] Extracting rendered images", flush=True)
        image_1 = render_pkg_1["render"]
        image_2 = render_pkg_2["render"]

        print("[DEBUG-CORGS-15] Computing PSNR difference", flush=True)
        psnr_diff = compute_rendering_disagreement(image_1, image_2)
        print(f"[DEBUG-CORGS-16] PSNR diff computed: {psnr_diff:.2f} dB", flush=True)
        metrics['render_psnr_diff'] = psnr_diff

        # 可选: SSIM Disagreement
        try:
            print("[DEBUG-CORGS-17] Computing SSIM difference (optional)", flush=True)
            ssim_diff = compute_ssim_disagreement(image_1, image_2)
            metrics['render_ssim_diff'] = ssim_diff
            print(f"[DEBUG-CORGS-18] SSIM diff computed: {ssim_diff:.4f}", flush=True)
        except Exception as e:
            print(f"[DEBUG-CORGS-18] SSIM computation skipped: {e}", flush=True)
            pass  # SSIM 计算失败时跳过

    print("[DEBUG-CORGS-19] log_corgs_metrics completed successfully", flush=True)
    return metrics
