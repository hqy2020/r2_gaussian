"""
SAGP (Structure-Aware Gaussian Pruning) for CT Reconstruction

来源: SeCuRe 论文 (Sparse-view 3D Curve Reconstruction with Gaussian Splatting)
适配: CT 重建场景

核心策略:
1. Spatial Coherence (空间一致性): DBSCAN + KNN 动态邻域半径
2. Visibility Consistency (可见性一致性): 多视角投影贡献分数
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN


class SAGPPruner:
    """Structure-Aware Gaussian Pruning for CT reconstruction"""

    def __init__(
        self,
        # 空间一致性参数
        spatial_k: int = 10,
        spatial_scale: float = 2.0,
        min_cluster_size: int = 5,
        # 可见性一致性参数
        visibility_threshold: float = 0.3,
        min_views_visible: int = 2,
    ):
        """
        初始化 SAGP 剪枝器

        Args:
            spatial_k: KNN 邻居数，用于计算动态邻域半径
            spatial_scale: DBSCAN eps 缩放因子 γ
            min_cluster_size: DBSCAN 最小聚类大小
            visibility_threshold: 最低可见性阈值 η (0-1)
            min_views_visible: 最少可见视角数
        """
        self.spatial_k = spatial_k
        self.spatial_scale = spatial_scale
        self.min_cluster_size = min_cluster_size
        self.visibility_threshold = visibility_threshold
        self.min_views_visible = min_views_visible

    def compute_spatial_outliers(
        self,
        positions: Tensor,
        chunk_size: int = 10000,
    ) -> Tensor:
        """
        空间一致性剪枝 - DBSCAN 聚类

        算法:
        1. KNN 计算每个点到 k 个最近邻的平均距离
        2. 动态估计 DBSCAN eps = median(distances) * spatial_scale
        3. DBSCAN 聚类，标记噪声点（label == -1）

        Args:
            positions: 高斯位置 [N, 3]
            chunk_size: 分块处理大小（节省显存）

        Returns:
            outlier_mask: 离群点 mask [N]，True = 需要剪枝
        """
        device = positions.device
        positions_np = positions.detach().cpu().numpy()
        n_points = len(positions_np)

        if n_points == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        # 1. 计算 KNN 距离
        # 使用分块处理避免内存爆炸
        k = min(self.spatial_k, n_points - 1)
        if k <= 0:
            return torch.zeros(n_points, dtype=torch.bool, device=device)

        distances = []
        for i in range(0, n_points, chunk_size):
            end_idx = min(i + chunk_size, n_points)
            chunk = positions_np[i:end_idx]

            # 计算到所有点的距离
            diff = chunk[:, None, :] - positions_np[None, :, :]  # [chunk, N, 3]
            dist = np.linalg.norm(diff, axis=2)  # [chunk, N]

            # 对每个点，排序找到 k 个最近邻（排除自己）
            sorted_dist = np.sort(dist, axis=1)
            # 跳过第一个（自己），取 k 个最近邻的平均
            knn_dist = sorted_dist[:, 1 : k + 1].mean(axis=1)
            distances.append(knn_dist)

        distances = np.concatenate(distances)

        # 2. 动态估计 DBSCAN eps
        eps = np.median(distances) * self.spatial_scale

        # 3. DBSCAN 聚类
        dbscan = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
        labels = dbscan.fit_predict(positions_np)

        # 标记噪声点（label == -1）为离群点
        outlier_mask = torch.tensor(labels == -1, dtype=torch.bool, device=device)

        return outlier_mask

    def compute_visibility_scores(
        self,
        gaussians,  # GaussianModel
        cameras: List,  # List[Camera]
        pipe,  # PipelineParams
    ) -> Tensor:
        """
        CT 多视角投影贡献分数

        核心思想:
        - 对每个视角执行渲染，获取 visibility_filter (radii > 0)
        - 累积每个高斯在所有视角中被"看到"的次数
        - 归一化为 [0, 1] 的可见性分数

        Args:
            gaussians: GaussianModel 实例
            cameras: 训练相机列表
            pipe: PipelineParams

        Returns:
            visibility_scores: 每个高斯的平均可见性分数 [N]
        """
        # 延迟导入避免循环依赖
        from r2_gaussian.gaussian import render

        n_gaussians = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device

        if n_gaussians == 0 or len(cameras) == 0:
            return torch.ones(n_gaussians, device=device)

        visibility_counts = torch.zeros(n_gaussians, device=device)

        with torch.no_grad():
            for camera in cameras:
                # 复用现有的 render 函数
                render_result = render(camera, gaussians, pipe)
                visibility_filter = render_result["visibility_filter"]  # [N] bool
                visibility_counts += visibility_filter.float()

        # 归一化：可见视角数 / 总视角数
        visibility_scores = visibility_counts / len(cameras)
        return visibility_scores

    def get_prune_mask(
        self,
        gaussians,  # GaussianModel
        cameras: List,  # List[Camera]
        pipe,  # PipelineParams
        use_spatial: bool = True,
        use_visibility: bool = True,
    ) -> Tuple[Tensor, dict]:
        """
        综合剪枝 mask

        Args:
            gaussians: GaussianModel 实例
            cameras: 训练相机列表
            pipe: PipelineParams
            use_spatial: 是否启用空间一致性剪枝
            use_visibility: 是否启用可见性一致性剪枝

        Returns:
            prune_mask: 需要剪枝的高斯 mask [N]，True = 需要剪枝
            stats: 统计信息字典
        """
        positions = gaussians.get_xyz
        n_gaussians = positions.shape[0]
        device = positions.device

        mask = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
        stats = {
            "total": n_gaussians,
            "spatial_outliers": 0,
            "visibility_outliers": 0,
            "total_pruned": 0,
        }

        if use_spatial:
            spatial_mask = self.compute_spatial_outliers(positions)
            stats["spatial_outliers"] = spatial_mask.sum().item()
            mask |= spatial_mask

        if use_visibility:
            vis_scores = self.compute_visibility_scores(gaussians, cameras, pipe)
            # 低可见性 = 需要剪枝
            visibility_mask = vis_scores < self.visibility_threshold
            stats["visibility_outliers"] = visibility_mask.sum().item()
            mask |= visibility_mask

        stats["total_pruned"] = mask.sum().item()
        return mask, stats
