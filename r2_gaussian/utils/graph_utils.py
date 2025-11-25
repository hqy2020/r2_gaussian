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

"""
Graph Utilities for GR-Gaussian (Graph Regularization)

实现 KNN 图构建和边权重计算，用于 Graph Laplacian 正则化
参考论文: GR-Gaussian (待补充具体论文信息)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class GaussianGraph:
    """
    高斯场的 KNN 图结构

    功能：
        - 构建 k-最近邻图
        - 计算基于距离的边权重
        - 支持动态更新（训练过程中高斯点位置变化）

    Args:
        k: KNN 邻居数量（默认 6，根据 GR-Gaussian 论文推荐）
        device: 'cuda' 或 'cpu'
        sigma: 高斯核带宽参数（用于边权重计算）
    """

    def __init__(self, k: int = 6, device: str = 'cuda', sigma: Optional[float] = None):
        self.k = k
        self.device = device
        self.sigma = sigma  # 如果为 None，将根据局部平均距离自动计算

        # 图结构
        self.edge_index = None  # (2, E) - 边索引 [src, dst]
        self.edge_weights = None  # (E,) - 边权重
        self.num_nodes = 0
        self.num_edges = 0

    def build_knn_graph(self, xyz: torch.Tensor, force_cpu: bool = False) -> None:
        """
        构建 k-最近邻图

        Args:
            xyz: (N, 3) 高斯点位置
            force_cpu: 强制使用 CPU 计算（当 GPU 内存不足时）
        """
        N = xyz.shape[0]
        self.num_nodes = N

        if N < self.k + 1:
            # 点数太少，无法构建 KNN 图
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.edge_weights = torch.empty((0,), dtype=torch.float32, device=self.device)
            self.num_edges = 0
            return

        try:
            # GPU 加速版本（优先）
            if not force_cpu and xyz.is_cuda:
                edge_index, distances = self._build_knn_gpu(xyz)
            else:
                # CPU fallback
                edge_index, distances = self._build_knn_cpu(xyz)

            self.edge_index = edge_index
            self.num_edges = edge_index.shape[1]

            # 边权重暂不计算，留给 compute_edge_weights()
            self.edge_weights = None

        except RuntimeError as e:
            # GPU OOM，回退到 CPU
            if not force_cpu:
                import warnings
                warnings.warn(f"GPU KNN 构建失败 ({str(e)})，回退到 CPU 版本")
                self.build_knn_graph(xyz, force_cpu=True)
            else:
                raise

    def _build_knn_gpu(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU 加速的 KNN 图构建

        Returns:
            edge_index: (2, E) - 边索引
            distances: (E,) - 边的欧氏距离
        """
        N = xyz.shape[0]

        # 计算所有点对之间的距离矩阵 (N, N)
        dists = torch.cdist(xyz, xyz)  # 使用 PyTorch 内置的高效实现

        # 获取每个点的 k+1 个最近邻（包括自己）
        _, indices = torch.topk(dists, self.k + 1, dim=1, largest=False, sorted=True)

        # 跳过自己（第一个邻居距离为 0）
        neighbor_indices = indices[:, 1:]  # (N, k)

        # 构建边索引
        src = torch.arange(N, device=xyz.device).unsqueeze(1).expand(-1, self.k)  # (N, k)
        dst = neighbor_indices  # (N, k)

        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)  # (2, N*k)

        # 提取对应的距离
        batch_indices = torch.arange(N, device=xyz.device).unsqueeze(1).expand(-1, self.k)
        distances = dists[batch_indices, neighbor_indices].reshape(-1)  # (N*k,)

        return edge_index, distances

    def _build_knn_cpu(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CPU 版本的 KNN 图构建（使用 sklearn）

        Returns:
            edge_index: (2, E) - 边索引
            distances: (E,) - 边的欧氏距离
        """
        from sklearn.neighbors import NearestNeighbors

        xyz_np = xyz.detach().cpu().numpy()
        N = xyz_np.shape[0]

        # 构建 KNN 搜索器
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='ball_tree').fit(xyz_np)
        distances_np, indices_np = nbrs.kneighbors(xyz_np)

        # 跳过自己（第一个邻居）
        neighbor_indices = indices_np[:, 1:]  # (N, k)
        neighbor_distances = distances_np[:, 1:]  # (N, k)

        # 构建边索引
        src = torch.arange(N, device=self.device).unsqueeze(1).expand(-1, self.k)
        dst = torch.tensor(neighbor_indices, dtype=torch.long, device=self.device)

        edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        distances = torch.tensor(neighbor_distances.reshape(-1), dtype=torch.float32, device=self.device)

        return edge_index, distances

    def compute_edge_weights(self, xyz: torch.Tensor) -> None:
        """
        计算边权重（基于高斯核）

        论文公式 (GR-Gaussian Eq. 12):
            w_ij = exp(-||p_i - p_j||² / k)

        其中 k 是 KNN 邻居数量，用作缩放因子

        Args:
            xyz: (N, 3) 高斯点位置（用于计算最新的距离）
        """
        if self.edge_index is None or self.num_edges == 0:
            self.edge_weights = torch.empty((0,), dtype=torch.float32, device=self.device)
            return

        src, dst = self.edge_index[0], self.edge_index[1]

        # 计算边的欧氏距离平方: ||p_i - p_j||²
        diff = xyz[src] - xyz[dst]  # (E, 3)
        distances_squared = (diff * diff).sum(dim=1)  # (E,)

        # 论文公式: w_ij = exp(-d² / k)，其中 k 是邻居数量
        # k 作为缩放因子，控制权重对距离的敏感度
        k = float(self.k)

        # 高斯核权重
        self.edge_weights = torch.exp(-distances_squared / k)

    def get_edge_index(self) -> torch.Tensor:
        """返回边索引 (2, E)"""
        return self.edge_index

    def get_edge_weights(self) -> torch.Tensor:
        """返回边权重 (E,)"""
        return self.edge_weights

    def __repr__(self):
        return (f"GaussianGraph(k={self.k}, num_nodes={self.num_nodes}, "
                f"num_edges={self.num_edges}, device={self.device})")


def apply_pga_gradient(density_param: torch.nn.Parameter,
                      graph: GaussianGraph,
                      lambda_g: float = 1e-4) -> None:
    """
    应用 PGA (Pixel-Graph-Aware Gradient Strategy)

    功能：
        在梯度中引入邻居密度差异信息，引导高斯点向合理的密度值调整

    公式：
        g_i += λ_g · Σ(Δρ_ij) / k
        其中 Δρ_ij = ρ_j - ρ_i (邻居密度差异)

    Args:
        density_param: 密度参数 (N, 1) 或 (N,) - 必须有 grad
        graph: GaussianGraph 实例（包含邻居信息）
        lambda_g: PGA 强度系数（论文推荐 1e-4）

    参考：
        GR-Gaussian 论文 Section 3.3: Pixel-Graph-Aware Gradient Strategy
    """
    # 边界情况检查
    if graph is None or graph.num_edges == 0:
        return

    if density_param.grad is None:
        return

    # 确保 density 是 1D
    density = density_param.data.squeeze()  # (N,)
    grad = density_param.grad.squeeze()     # (N,)

    if density.dim() != 1:
        raise ValueError(f"density must be 1D after squeeze, got shape {density.shape}")

    # 获取图结构
    edge_index = graph.get_edge_index()  # (2, E)
    if edge_index is None:
        return

    src = edge_index[0].long()  # (E,) - 源节点索引
    dst = edge_index[1].long()  # (E,) - 目标节点索引

    # 计算邻居密度差异 Δρ_ij = ρ_j - ρ_i
    density_src = density[src]  # (E,)
    density_dst = density[dst]  # (E,)
    density_diff = density_dst - density_src  # (E,)

    # 聚合每个节点的邻居密度差异（使用 scatter_add）
    # neighbor_density_sum[i] = Σ(Δρ_ij) for all neighbors j of node i
    neighbor_density_sum = torch.zeros_like(density)  # (N,)
    neighbor_count = torch.zeros_like(density)        # (N,)

    neighbor_density_sum.scatter_add_(0, src, density_diff)
    neighbor_count.scatter_add_(0, src, torch.ones_like(density_diff))

    # 计算平均密度差异 Σ(Δρ_ij) / k
    # 避免除以零
    neighbor_count = torch.clamp(neighbor_count, min=1.0)
    avg_density_diff = neighbor_density_sum / neighbor_count  # (N,)

    # 应用 PGA：g_i += λ_g · avg(Δρ_ij)
    pga_gradient = lambda_g * avg_density_diff

    # 更新梯度（原地操作）
    if grad.shape != pga_gradient.shape:
        raise ValueError(f"Gradient shape mismatch: grad {grad.shape} vs pga {pga_gradient.shape}")

    grad.add_(pga_gradient)
