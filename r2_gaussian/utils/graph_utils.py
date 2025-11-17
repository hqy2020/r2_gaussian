"""
GR-Gaussian Graph Utilities
构建和管理高斯核的 KNN 图结构

依赖:
    - PyTorch Geometric (优先): 高效 GPU KNN 搜索
    - PyTorch (回退): 纯 CPU/GPU 实现 (性能较低)

参考论文: GR-Gaussian: Bridging 2D Priors with 3D Generation using Graph Laplacian
"""

import torch
import torch.nn.functional as F

# 尝试导入 PyTorch Geometric
HAS_TORCH_GEOMETRIC = False
try:
    from torch_geometric.nn import knn_graph
    # 测试是否真正可用 (某些情况下导入成功但缺少依赖)
    import torch
    _ = knn_graph(torch.randn(10, 3), k=3)
    HAS_TORCH_GEOMETRIC = True
except (ImportError, RuntimeError) as e:
    HAS_TORCH_GEOMETRIC = False
    # 只在第一次导入时打印警告
    import warnings
    warnings.warn("PyTorch Geometric KNN not available, using fallback implementation. "
                  f"Install torch-cluster for better performance. Error: {e}")


def build_knn_graph(xyz: torch.Tensor, k: int = 6) -> torch.Tensor:
    """
    构建 k 近邻图

    Args:
        xyz: 点云坐标 [N, 3]
        k: 近邻数量

    Returns:
        edges: 边的索引 [2, E]
    """
    if HAS_TORCH_GEOMETRIC:
        # 使用 PyTorch Geometric 的高效 KNN 实现
        edge_index = knn_graph(
            xyz,
            k=k,
            loop=False,  # 不包含自环
            flow='source_to_target'
        )
    else:
        # Fallback: 纯 PyTorch 实现 (较慢)
        edge_index = _pytorch_knn_graph(xyz, k)

    return edge_index


def _pytorch_knn_graph(positions: torch.Tensor, k: int) -> torch.Tensor:
    """
    纯 PyTorch 实现 KNN (Fallback)
    复杂度: O(N²) - 仅在没有 PyG 时使用

    Args:
        positions: 点云坐标 [N, 3]
        k: 近邻数量

    Returns:
        edge_index: 边的索引 [2, E]
    """
    N = positions.shape[0]

    # 计算所有点对距离矩阵 (N, N)
    dist_matrix = torch.cdist(positions, positions, p=2)

    # 找到每个点的 k 个最近邻 (不包括自身)
    # topk 返回 (values, indices), shape: (N, k)
    knn_dists, knn_indices = torch.topk(
        dist_matrix,
        k=k + 1,  # +1 因为第一个是自身
        largest=False,  # 最小的 k 个
        dim=1
    )

    # 移除自环 (第一列是自身,距离为 0)
    knn_indices = knn_indices[:, 1:]  # (N, k)

    # 构建边索引
    src = torch.arange(N, device=positions.device).unsqueeze(1).repeat(1, k)  # (N, k)
    dst = knn_indices  # (N, k)

    edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)  # (2, N*k)
    return edge_index


def compute_graph_laplacian(xyz: torch.Tensor,
                            edges: torch.Tensor,
                            normalized: bool = True) -> torch.Tensor:
    """
    计算 Graph Laplacian 矩阵

    Args:
        xyz: 点云坐标 [N, 3]
        edges: 边的索引 [2, E]
        normalized: 是否使用归一化 Laplacian

    Returns:
        laplacian: Graph Laplacian 矩阵 [N, N] (稀疏)
    """
    N = xyz.shape[0]
    E = edges.shape[1]

    src, dst = edges[0], edges[1]

    # 计算边权重 (基于欧氏距离)
    edge_weights = torch.exp(-torch.sum((xyz[src] - xyz[dst]) ** 2, dim=1))

    # 构建邻接矩阵 A
    A = torch.zeros(N, N, device=xyz.device)
    A[src, dst] = edge_weights

    # 度矩阵 D
    D = torch.diag(A.sum(dim=1))

    if normalized:
        # 归一化 Laplacian: L = I - D^(-1/2) A D^(-1/2)
        D_sqrt_inv = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-8))
        L = torch.eye(N, device=xyz.device) - D_sqrt_inv @ A @ D_sqrt_inv
    else:
        # 非归一化 Laplacian: L = D - A
        L = D - A

    return L


class GaussianGraph:
    """
    管理高斯核的 KNN 图结构

    图构建策略:
        - 使用 KNN 双向连接确保对称性
        - 边权重基于欧氏距离的高斯衰减

    属性:
        k: int, 邻居数量 (论文推荐 6)
        device: str, 计算设备
        edge_index: (2, E) 边索引 (src, dst)
        edge_weights: (E,) 边权重
        num_nodes: int, 节点数量
    """

    def __init__(self, k=6, device='cuda'):
        """
        Args:
            k: 邻居数量 (论文推荐 6)
            device: 计算设备
        """
        self.k = k
        self.device = device
        self.edge_index = None  # (2, E) 边索引
        self.edge_weights = None  # (E,) 边权重
        self.num_nodes = 0

    def build_knn_graph(self, positions):
        """
        构建 KNN 双向图

        Args:
            positions: (M, 3) 高斯核位置 (已归一化到 [-1, 1]³)

        Returns:
            edge_index: (2, E) 边索引 (src, dst)
        """
        self.num_nodes = positions.shape[0]

        # 使用全局函数构建图
        edge_index = build_knn_graph(positions, k=self.k)

        # 强制双向连接 (对称化)
        edge_index = self._symmetrize_edges(edge_index)

        self.edge_index = edge_index.to(self.device)
        return self.edge_index

    def _symmetrize_edges(self, edge_index):
        """
        强制双向连接:仅保留互为 KNN 的边

        条件: (i, j) ∈ E 且 (j, i) ∈ E
        """
        src, dst = edge_index[0], edge_index[1]

        # 将边转为集合 (使用元组作为键)
        edge_set = set(zip(src.cpu().tolist(), dst.cpu().tolist()))

        # 过滤双向边
        symmetric_edges = []
        for i, j in edge_set:
            if (j, i) in edge_set:
                symmetric_edges.append((i, j))

        # 转回张量
        if len(symmetric_edges) == 0:
            print("⚠️  Warning: No symmetric edges found, falling back to asymmetric graph")
            return edge_index

        symmetric_edges = torch.tensor(symmetric_edges, dtype=torch.long, device=self.device).t()

        print(f"[Graph] Symmetrized edges: {edge_index.shape[1]} → {symmetric_edges.shape[1]}")
        return symmetric_edges

    def compute_edge_weights(self, positions):
        """
        计算边权重 w_ij = exp(-||p_i - p_j||² / k)

        Args:
            positions: (M, 3) 高斯核位置

        Returns:
            edge_weights: (E,) 边权重
        """
        if self.edge_index is None:
            raise ValueError("Must build graph first!")

        src, dst = self.edge_index[0], self.edge_index[1]

        # 计算边的欧氏距离平方
        pos_src = positions[src]  # (E, 3)
        pos_dst = positions[dst]  # (E, 3)
        dist_sq = torch.sum((pos_src - pos_dst) ** 2, dim=1)  # (E,)

        # 高斯衰减权重
        weights = torch.exp(-dist_sq / self.k)

        self.edge_weights = weights
        return weights

    def get_neighbors(self, node_idx):
        """
        查询指定节点的邻居索引

        Args:
            node_idx: int, 节点索引

        Returns:
            neighbors: (N_neighbors,) 邻居索引列表
        """
        if self.edge_index is None:
            raise ValueError("Must build graph first!")

        mask = self.edge_index[0] == node_idx
        neighbors = self.edge_index[1][mask]
        return neighbors

    def compute_density_differences(self, densities):
        """
        计算所有边的密度差异 Δρ_ij = |ρ_i - ρ_j|

        Args:
            densities: (M,) 高斯核密度值

        Returns:
            density_diffs: (E,) 密度差异
        """
        if self.edge_index is None:
            raise ValueError("Must build graph first!")

        src, dst = self.edge_index[0], self.edge_index[1]
        density_diffs = torch.abs(densities[src] - densities[dst])
        return density_diffs
