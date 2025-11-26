"""
CoR-GS Co-pruning Module (Stage 2)
==================================
基于论文 "CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization" (ECCV 2024)

Co-pruning 算法：
1. 将两个 3D Gaussian 模型视为点云
2. 对每个 Gaussian 找到在另一个模型中的最近邻
3. 如果最近邻距离 > threshold，标记为非匹配
4. 删除所有非匹配的 Gaussians

参考：论文 Section 4.1
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List


def compute_pairwise_distances(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
    chunk_size: int = 10000
) -> torch.Tensor:
    """
    计算两组点之间的成对距离（分块计算以节省显存）

    Args:
        points_a: (N, 3) 第一组点
        points_b: (M, 3) 第二组点
        chunk_size: 分块大小，用于处理大规模点云

    Returns:
        distances: (N,) 每个 points_a 到 points_b 的最小距离
    """
    device = points_a.device
    n_points = points_a.shape[0]
    min_distances = torch.zeros(n_points, device=device)

    # 分块计算以避免显存溢出
    for i in range(0, n_points, chunk_size):
        end_idx = min(i + chunk_size, n_points)
        chunk_a = points_a[i:end_idx]  # (chunk, 3)

        # 计算 L2 距离: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b
        # 使用 cdist 更高效
        if points_b.shape[0] <= 50000:
            # 小规模：直接计算
            dists = torch.cdist(chunk_a, points_b, p=2)  # (chunk, M)
            min_dists, _ = torch.min(dists, dim=1)  # (chunk,)
        else:
            # 大规模：进一步分块
            min_dists = torch.full((end_idx - i,), float('inf'), device=device)
            for j in range(0, points_b.shape[0], chunk_size):
                end_j = min(j + chunk_size, points_b.shape[0])
                chunk_b = points_b[j:end_j]
                dists = torch.cdist(chunk_a, chunk_b, p=2)
                chunk_min, _ = torch.min(dists, dim=1)
                min_dists = torch.minimum(min_dists, chunk_min)

        min_distances[i:end_idx] = min_dists

    return min_distances


def compute_unmatched_mask(
    gaussians_0,
    gaussians_1,
    threshold: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算两个 Gaussian 模型中不匹配的 Gaussians 掩码

    Args:
        gaussians_0: 第一个 GaussianModel
        gaussians_1: 第二个 GaussianModel
        threshold: 最大匹配距离 τ (论文默认 5.0，针对归一化场景 [-1,1]³)

    Returns:
        unmatched_mask_0: (N0,) bool tensor，True 表示该 Gaussian 应被删除
        unmatched_mask_1: (N1,) bool tensor
    """
    # 获取 Gaussian 位置
    pos_0 = gaussians_0.get_xyz.detach()  # (N0, 3)
    pos_1 = gaussians_1.get_xyz.detach()  # (N1, 3)

    # 计算每个点到另一组点的最小距离
    dist_0_to_1 = compute_pairwise_distances(pos_0, pos_1)  # (N0,)
    dist_1_to_0 = compute_pairwise_distances(pos_1, pos_0)  # (N1,)

    # 计算非匹配掩码：距离 > threshold 的点标记为需要删除
    unmatched_mask_0 = dist_0_to_1 > threshold
    unmatched_mask_1 = dist_1_to_0 > threshold

    return unmatched_mask_0, unmatched_mask_1


def co_pruning(
    gaussians_list: List,
    threshold: float = 5.0,
    min_points: int = 1000,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    执行 Co-pruning：修剪两个模型间不匹配的 Gaussians

    Args:
        gaussians_list: 包含至少两个 GaussianModel 的列表
        threshold: 最大匹配距离 τ (论文默认 5.0)
        min_points: 保留的最小点数，防止过度修剪
        verbose: 是否打印日志

    Returns:
        num_pruned_0: 模型 0 删除的点数
        num_pruned_1: 模型 1 删除的点数
    """
    if len(gaussians_list) < 2:
        return 0, 0

    gaussians_0 = gaussians_list[0]
    gaussians_1 = gaussians_list[1]

    # 记录初始点数
    n0_before = gaussians_0.get_xyz.shape[0]
    n1_before = gaussians_1.get_xyz.shape[0]

    # 计算不匹配掩码
    unmatched_mask_0, unmatched_mask_1 = compute_unmatched_mask(
        gaussians_0, gaussians_1, threshold
    )

    # 安全检查：确保不会删除太多点
    n_unmatched_0 = unmatched_mask_0.sum().item()
    n_unmatched_1 = unmatched_mask_1.sum().item()

    # 如果删除后点数少于 min_points，则调整掩码
    if n0_before - n_unmatched_0 < min_points:
        if verbose:
            print(f"[Co-pruning] Warning: Model 0 would have too few points "
                  f"({n0_before - n_unmatched_0} < {min_points}), skipping pruning")
        unmatched_mask_0 = torch.zeros_like(unmatched_mask_0, dtype=torch.bool)
        n_unmatched_0 = 0

    if n1_before - n_unmatched_1 < min_points:
        if verbose:
            print(f"[Co-pruning] Warning: Model 1 would have too few points "
                  f"({n1_before - n_unmatched_1} < {min_points}), skipping pruning")
        unmatched_mask_1 = torch.zeros_like(unmatched_mask_1, dtype=torch.bool)
        n_unmatched_1 = 0

    # 执行修剪
    if n_unmatched_0 > 0:
        gaussians_0.prune_points(unmatched_mask_0)
    if n_unmatched_1 > 0:
        gaussians_1.prune_points(unmatched_mask_1)

    # 记录最终点数
    n0_after = gaussians_0.get_xyz.shape[0]
    n1_after = gaussians_1.get_xyz.shape[0]

    if verbose:
        print(f"[Co-pruning] Model 0: {n0_before} -> {n0_after} "
              f"(pruned {n_unmatched_0}, {100*n_unmatched_0/max(n0_before,1):.1f}%)")
        print(f"[Co-pruning] Model 1: {n1_before} -> {n1_after} "
              f"(pruned {n_unmatched_1}, {100*n_unmatched_1/max(n1_before,1):.1f}%)")

    return n_unmatched_0, n_unmatched_1


def compute_point_disagreement(
    gaussians_list: List,
    max_distance: float = 5.0
) -> dict:
    """
    计算两个模型之间的 Point Disagreement 指标
    用于监控训练过程中模型的一致性

    Args:
        gaussians_list: GaussianModel 列表
        max_distance: Fitness 计算的最大距离阈值

    Returns:
        dict: 包含 fitness, rmse, unmatched_ratio 等指标
    """
    if len(gaussians_list) < 2:
        return {'fitness': 1.0, 'rmse': 0.0, 'unmatched_ratio_0': 0.0, 'unmatched_ratio_1': 0.0}

    gaussians_0 = gaussians_list[0]
    gaussians_1 = gaussians_list[1]

    pos_0 = gaussians_0.get_xyz.detach()  # (N0, 3)
    pos_1 = gaussians_1.get_xyz.detach()  # (N1, 3)

    # 计算距离
    dist_0_to_1 = compute_pairwise_distances(pos_0, pos_1)
    dist_1_to_0 = compute_pairwise_distances(pos_1, pos_0)

    # Fitness: 在 max_distance 内有匹配的点的比例
    matched_0 = (dist_0_to_1 <= max_distance).float().mean().item()
    matched_1 = (dist_1_to_0 <= max_distance).float().mean().item()
    fitness = (matched_0 + matched_1) / 2.0

    # RMSE: 匹配点对的平均距离
    all_distances = torch.cat([dist_0_to_1, dist_1_to_0])
    rmse = torch.sqrt((all_distances ** 2).mean()).item()

    # Unmatched ratio
    unmatched_ratio_0 = (dist_0_to_1 > max_distance).float().mean().item()
    unmatched_ratio_1 = (dist_1_to_0 > max_distance).float().mean().item()

    return {
        'fitness': fitness,
        'rmse': rmse,
        'unmatched_ratio_0': unmatched_ratio_0,
        'unmatched_ratio_1': unmatched_ratio_1,
        'n_points_0': pos_0.shape[0],
        'n_points_1': pos_1.shape[0]
    }
