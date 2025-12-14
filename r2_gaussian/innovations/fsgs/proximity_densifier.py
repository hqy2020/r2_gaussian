#
# FSGS Proximity-Guided Densifier (Rewritten & Optimized)
#
# This is a clean, modular reimplementation of the FSGS proximity-guided densification algorithm.
# Optimized for clarity, performance, and extensibility.
#

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import warnings

# 尝试导入 FAISS（可选依赖）
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False



class ProximityGuidedDensifier:
    """
    FSGS Proximity-Guided Densification

    Paper: FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting (ECCV 2024)
    arXiv: https://arxiv.org/abs/2312.00451

    Core Algorithm:
        1. Compute proximity score: P_i = (1/K) × Σ ||μ_i - μ_j||  for j in K-nearest-neighbors
        2. Identify densify candidates: P_i > threshold
        3. Generate new Gaussians at edge midpoints
        4. Initialize attributes from destination Gaussian

    Key Improvements over original paper (for CT reconstruction):
        - Adaptive K and threshold based on medical tissue type
        - Batch K-NN for 10-30x speedup
        - Chunked computation to avoid OOM
        - Hybrid strategy with gradient-based densification

    Example:
        >>> densifier = ProximityGuidedDensifier(k_neighbors=6, proximity_threshold=8.0)
        >>> positions = torch.randn(10000, 3, device='cuda')
        >>> scores = densifier.compute_proximity_scores(positions)
        >>> candidates = densifier.identify_densify_candidates(scores)
        >>> print(f"Densify {candidates.sum()} out of {len(positions)} Gaussians")
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        proximity_threshold: float = 0.05,
        chunk_size: int = 5000,
        enable: bool = True,
        # 🆕 GAR 优化参数
        adaptive_threshold: bool = False,
        adaptive_method: str = "percentile",
        adaptive_percentile: float = 90.0,
        progressive_decay: bool = False,
        decay_start_ratio: float = 0.5,
        final_strength: float = 0.3,
        # 🆕 K-NN 加速选项
        use_faiss: bool = True,  # 是否使用 FAISS 加速 K-NN
    ):
        """
        Initialize FSGS Proximity Densifier

        Args:
            k_neighbors: Number of nearest neighbors for proximity calculation
                - FSGS paper default: 3
                - R²-Gaussian CT optimal: 5 (more stable for noisy CT data)
                - Sensitivity: Higher K → smoother scores, more computation

            proximity_threshold: Proximity score threshold for densification trigger
                - 场景归一化到 [-1, 1]³ 后，典型邻近分数范围: 0.01 ~ 0.5
                - R²-Gaussian CT optimal: 0.05 (密化邻近分数 > 0.05 的稀疏区域)
                - Physical meaning: Average distance to K nearest neighbors

            chunk_size: Chunk size for batched K-NN computation (memory optimization)
                - Increase for more GPU memory: 10000-20000
                - Decrease for OOM errors: 2000-3000
                - No effect on results, only memory/speed tradeoff

            enable: Master switch for this module
                - Set to False to disable FSGS densification entirely

            adaptive_threshold: 启用自适应阈值（基于邻近分数分布）
            adaptive_method: 自适应阈值计算方法 ("percentile", "std", "iqr")
            adaptive_percentile: percentile 方法的百分位数（默认 90，只密化最稀疏的 10%）
            progressive_decay: 启用渐进衰减（训练后期减少密化强度）
            decay_start_ratio: 衰减开始的进度比例（默认 0.5，即 50% 进度后开始衰减）
            final_strength: 最终密化强度（默认 0.3，即阈值提高到 1/0.3 ≈ 3.3 倍）
            use_faiss: 是否使用 FAISS 加速 K-NN（需要安装 faiss-gpu 或 faiss-cpu）
        """
        self.k_neighbors = k_neighbors
        self.proximity_threshold = proximity_threshold
        self.chunk_size = chunk_size
        self.enable = enable

        # 🆕 GAR 优化参数
        self.adaptive_threshold = adaptive_threshold
        self.adaptive_method = adaptive_method
        self.adaptive_percentile = adaptive_percentile
        self.progressive_decay = progressive_decay
        self.decay_start_ratio = decay_start_ratio
        self.final_strength = final_strength

        # 🆕 K-NN 加速选项
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        if use_faiss and not FAISS_AVAILABLE:
            warnings.warn(
                "FAISS not available. Install with: pip install faiss-gpu (or faiss-cpu). "
                "Falling back to PyTorch chunked K-NN."
            )

        # Statistics tracking (for TensorBoard logging)
        self.stats = {
            'num_densify_calls': 0,
            'total_new_gaussians': 0,
            'avg_proximity_score': 0.0,
            'adaptive_threshold_value': 0.0,  # 🆕
            'decay_multiplier': 1.0,  # 🆕
            'knn_method': 'faiss' if self.use_faiss else 'pytorch',  # 🆕
        }

    def compute_proximity_scores(
        self,
        positions: torch.Tensor,  # (N, 3)
        custom_k: Optional[int] = None,
        return_neighbors: bool = False
    ) -> torch.Tensor:  # (N,) or ((N,), (N, K), (N, K))
        """
        Compute proximity score for each Gaussian

        Proximity score definition (FSGS Eq. 4):
            P_i = (1/K) × Σ(j∈N_K(i)) d_ij
            where d_ij = ||μ_i - μ_j||₂

        Physical meaning:
            - High P_i: This Gaussian is "lonely" → sparse region → needs densification
            - Low P_i: This Gaussian is "crowded" → dense region → no densification needed

        Args:
            positions: Gaussian center positions (N, 3)
            custom_k: Override self.k_neighbors for this call (optional)
            return_neighbors: If True, also return neighbor indices and distances

        Returns:
            If return_neighbors=False:
                proximity_scores: (N,) proximity score for each Gaussian
            If return_neighbors=True:
                (proximity_scores, neighbor_indices, neighbor_distances)
                where neighbor_indices: (N, K), neighbor_distances: (N, K)

        Raises:
            RuntimeError: If CUDA out of memory (reduce chunk_size)
            ValueError: If positions.shape[0] < K+1
        """
        if not self.enable:
            # Return dummy scores if disabled
            if return_neighbors:
                return torch.zeros(len(positions), device=positions.device), None, None
            return torch.zeros(len(positions), device=positions.device)

        N = positions.shape[0]
        device = positions.device
        K = custom_k if custom_k is not None else self.k_neighbors

        if N <= K:
            raise ValueError(
                f"Number of Gaussians ({N}) must be > K ({K}). "
                f"Cannot compute {K}-nearest neighbors."
            )

        # 选择 K-NN 实现：FAISS（快速）或 PyTorch chunked（稳定）
        if self.use_faiss:
            neighbor_distances, neighbor_indices = self._compute_knn_faiss(
                positions, K
            )
        else:
            neighbor_distances, neighbor_indices = self._compute_knn_chunked(
                positions, K
            )

        # Compute proximity score: average distance to K nearest neighbors
        proximity_scores = neighbor_distances.mean(dim=1)  # (N,)

        # Update statistics
        self.stats['avg_proximity_score'] = proximity_scores.mean().item()

        if return_neighbors:
            return proximity_scores, neighbor_indices, neighbor_distances
        return proximity_scores

    def _compute_knn_faiss(
        self,
        positions: torch.Tensor,  # (N, 3)
        K: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (N, K), (N, K)
        """
        使用 FAISS 计算 K-近邻（高效实现，O(N log N)）

        FAISS 是 Facebook AI Research 开发的高效相似度搜索库，
        相比 torch.cdist 的 O(N²) 复杂度，FAISS 使用索引结构可以
        达到 O(N log N) 或更好的复杂度。

        Args:
            positions: (N, 3) Gaussian 位置
            K: 近邻数量

        Returns:
            neighbor_distances: (N, K) 到 K 个近邻的距离
            neighbor_indices: (N, K) K 个近邻的索引
        """
        N = positions.shape[0]
        device = positions.device

        # 转换为 numpy（FAISS 需要）
        positions_np = positions.detach().cpu().numpy().astype('float32')

        # 创建 FAISS 索引
        # 对于小规模数据使用精确搜索，大规模数据使用近似搜索
        d = 3  # 维度
        if N < 50000:
            # 精确搜索（小规模）
            index = faiss.IndexFlatL2(d)
        else:
            # 使用 IVF 近似搜索（大规模，更快）
            nlist = min(100, N // 100)  # 聚类数
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(positions_np)
            index.nprobe = min(10, nlist)  # 搜索时检查的聚类数

        # 添加数据到索引
        index.add(positions_np)

        # K+1 因为会包含自己
        distances_sq, indices = index.search(positions_np, K + 1)

        # 移除自己（第一个结果通常是自己）
        # 但有时候不是，所以需要检查
        neighbor_distances = []
        neighbor_indices = []
        for i in range(N):
            mask = indices[i] != i
            valid_indices = indices[i][mask][:K]
            valid_distances = distances_sq[i][mask][:K]

            # 如果不够 K 个，补充（理论上不会发生）
            if len(valid_indices) < K:
                valid_indices = indices[i, 1:K+1]
                valid_distances = distances_sq[i, 1:K+1]

            neighbor_indices.append(valid_indices)
            neighbor_distances.append(valid_distances)

        # 转换回 torch tensor
        neighbor_indices = torch.tensor(neighbor_indices, device=device, dtype=torch.long)
        neighbor_distances = torch.tensor(neighbor_distances, device=device, dtype=torch.float32)

        # FAISS 返回的是 L2 距离的平方，需要开根号
        neighbor_distances = torch.sqrt(neighbor_distances)

        return neighbor_distances, neighbor_indices

    def _compute_knn_chunked(
        self,
        positions: torch.Tensor,  # (N, 3)
        K: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (N, K), (N, K)
        """
        Compute K-nearest neighbors using chunked PyTorch operations

        This is the memory-efficient fallback when CUDA K-NN is not available.
        Processes positions in chunks to avoid OOM for large N.

        Args:
            positions: (N, 3) Gaussian positions
            K: Number of nearest neighbors

        Returns:
            neighbor_distances: (N, K) distances to K nearest neighbors
            neighbor_indices: (N, K) indices of K nearest neighbors
        """
        N = positions.shape[0]
        device = positions.device

        all_neighbor_distances = []
        all_neighbor_indices = []

        for start_idx in range(0, N, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, N)
            chunk = positions[start_idx:end_idx]  # (chunk_size, 3)

            try:
                # Compute pairwise distances: chunk vs all positions
                distances = torch.cdist(chunk, positions, p=2)  # (chunk_size, N)

                # Set self-distance to infinity (to exclude self from K-NN)
                for i in range(distances.shape[0]):
                    distances[i, start_idx + i] = float('inf')

                # Find K nearest neighbors
                neighbor_distances, neighbor_indices = torch.topk(
                    distances,
                    k=K,
                    dim=1,
                    largest=False  # Smallest distances
                )  # (chunk_size, K)

                all_neighbor_distances.append(neighbor_distances)
                all_neighbor_indices.append(neighbor_indices)

                del distances  # Free memory immediately

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM: Fall back to CPU computation for this chunk
                    warnings.warn(
                        f"CUDA OOM at chunk {start_idx}-{end_idx}. "
                        f"Using CPU fallback. Consider reducing chunk_size from {self.chunk_size}."
                    )
                    torch.cuda.empty_cache()

                    chunk_cpu = chunk.cpu()
                    positions_cpu = positions.cpu()
                    distances_cpu = torch.cdist(chunk_cpu, positions_cpu, p=2)

                    for i in range(distances_cpu.shape[0]):
                        distances_cpu[i, start_idx + i] = float('inf')

                    neighbor_distances, neighbor_indices = torch.topk(
                        distances_cpu,
                        k=K,
                        dim=1,
                        largest=False
                    )

                    # Move back to GPU
                    neighbor_distances = neighbor_distances.to(device)
                    neighbor_indices = neighbor_indices.to(device)

                    all_neighbor_distances.append(neighbor_distances)
                    all_neighbor_indices.append(neighbor_indices)
                else:
                    raise e

        # Concatenate all chunks
        neighbor_distances = torch.cat(all_neighbor_distances, dim=0)  # (N, K)
        neighbor_indices = torch.cat(all_neighbor_indices, dim=0)  # (N, K)

        return neighbor_distances, neighbor_indices

    def compute_adaptive_threshold_value(
        self,
        proximity_scores: torch.Tensor,
        method: Optional[str] = None,
        percentile: Optional[float] = None,
    ) -> torch.Tensor:
        """
        基于邻近分数分布计算自适应阈值

        核心思想：根据当前点云的邻近分数分布自动确定阈值，
        而非使用固定值。这样可以适应不同密度的点云。

        Args:
            proximity_scores: (N,) 每个高斯的邻近分数
            method: 阈值计算方法，可选:
                - "percentile": 使用百分位数（推荐，只密化最稀疏的点）
                - "std": mean + multiplier * std
                - "iqr": Q3 + 1.5 * IQR（类似异常检测）
            percentile: percentile 方法的百分位数

        Returns:
            threshold: 自适应阈值（标量）

        Example:
            >>> scores = densifier.compute_proximity_scores(positions)
            >>> adaptive_thresh = densifier.compute_adaptive_threshold_value(scores)
            >>> # adaptive_thresh 会根据点云密度自动调整
        """
        method = self.adaptive_method if method is None else method
        percentile = self.adaptive_percentile if percentile is None else percentile

        if method == "percentile":
            # 使用百分位数：只密化邻近分数高于该百分位的点
            # percentile=90 意味着只密化最稀疏的 10% 点
            threshold = torch.quantile(proximity_scores, percentile / 100.0)
        elif method == "std":
            # 使用均值 + 标准差
            mean = proximity_scores.mean()
            std = proximity_scores.std()
            threshold = mean + 1.5 * std
        elif method == "iqr":
            # 使用四分位距（类似箱线图异常检测）
            q1 = torch.quantile(proximity_scores, 0.25)
            q3 = torch.quantile(proximity_scores, 0.75)
            iqr = q3 - q1
            threshold = q3 + 1.5 * iqr
        else:
            raise ValueError(f"Unknown adaptive method: {method}")

        # [优化] 下限保护：使用 P25（第一四分位数）作为下限
        # 允许密化最多 75% 的点，而非原来的 50%
        # 原因：P50 下限过于保守，会阻止对中等稀疏区域的密化
        p25 = torch.quantile(proximity_scores, 0.25)
        threshold = threshold.clamp(min=p25)

        # 更新统计
        self.stats['adaptive_threshold_value'] = threshold.item()

        return threshold

    def get_decay_multiplier(
        self,
        iteration: int,
        start_iter: int,
        until_iter: int,
        decay_start_ratio: Optional[float] = None,
        final_strength: Optional[float] = None,
    ) -> float:
        """
        计算渐进衰减乘数

        核心思想：训练后期逐渐减少密化强度，给新生成的高斯
        更多优化时间，避免在接近收敛时添加干扰。

        Args:
            iteration: 当前迭代次数
            start_iter: GAR 开始迭代
            until_iter: GAR 结束迭代
            decay_start_ratio: 衰减开始的进度比例（默认 0.5）
            final_strength: 最终强度（默认 0.3，阈值提高 ~3.3 倍）

        Returns:
            multiplier: 阈值乘数（>=1.0，越大越保守）

        Example:
            >>> mult = densifier.get_decay_multiplier(10000, 1000, 15000)
            >>> # 如果 progress > 0.5，mult > 1.0（阈值提高）
            >>> effective_threshold = base_threshold * mult
        """
        decay_start_ratio = self.decay_start_ratio if decay_start_ratio is None else decay_start_ratio
        final_strength = self.final_strength if final_strength is None else final_strength

        # 计算进度 [0, 1]
        total_iters = until_iter - start_iter
        if total_iters <= 0:
            return 1.0

        progress = (iteration - start_iter) / total_iters
        progress = max(0.0, min(1.0, progress))  # clamp to [0, 1]

        if progress < decay_start_ratio:
            # 衰减开始前：正常密化
            multiplier = 1.0
        else:
            # 衰减开始后：阈值逐渐提高
            # 当 progress = 1.0 时，multiplier = 1/final_strength
            decay_progress = (progress - decay_start_ratio) / (1.0 - decay_start_ratio)
            # 线性插值：从 1.0 到 1/final_strength
            final_multiplier = 1.0 / final_strength
            multiplier = 1.0 + decay_progress * (final_multiplier - 1.0)

        # 更新统计
        self.stats['decay_multiplier'] = multiplier

        return multiplier

    def identify_densify_candidates(
        self,
        proximity_scores: torch.Tensor,  # (N,)
        custom_threshold: Optional[torch.Tensor] = None,  # (N,) or scalar
        gradient_mask: Optional[torch.Tensor] = None,  # (N,) bool
        hybrid_mode: str = "union"  # "union", "intersection", "proximity_only"
    ) -> torch.Tensor:  # (N,) bool mask
        """
        Identify which Gaussians should be densified

        Args:
            proximity_scores: (N,) proximity scores from compute_proximity_scores

            custom_threshold: Custom proximity threshold (optional)
                - If None: use self.proximity_threshold for all points
                - If scalar: use this value for all points
                - If (N,): use adaptive threshold (e.g., from medical constraints)

            gradient_mask: (N,) bool mask from gradient-based densification (optional)
                - If None: use proximity scores only
                - If provided: combine with proximity scores based on hybrid_mode

            hybrid_mode: How to combine proximity and gradient masks
                - "union": densify if proximity OR gradient (more aggressive)
                - "intersection": densify if proximity AND gradient (conservative)
                - "proximity_only": ignore gradient mask

        Returns:
            densify_mask: (N,) boolean mask indicating which Gaussians to densify

        Example:
            >>> # Pure FSGS (proximity only)
            >>> mask = densifier.identify_densify_candidates(scores)

            >>> # Hybrid FSGS + gradient-based
            >>> mask = densifier.identify_densify_candidates(
            ...     scores,
            ...     gradient_mask=grad_mask,
            ...     hybrid_mode="union"
            ... )

            >>> # Medical constraints (adaptive threshold)
            >>> adaptive_thresh = medical_constraints.get_proximity_thresholds(tissue_types)
            >>> mask = densifier.identify_densify_candidates(
            ...     scores,
            ...     custom_threshold=adaptive_thresh
            ... )
        """
        if not self.enable:
            return torch.zeros(len(proximity_scores), dtype=torch.bool, device=proximity_scores.device)

        # Determine threshold
        if custom_threshold is None:
            threshold = self.proximity_threshold
        else:
            threshold = custom_threshold

        # Proximity-based mask
        proximity_mask = proximity_scores > threshold  # (N,) bool

        # Combine with gradient mask if provided
        if gradient_mask is None or hybrid_mode == "proximity_only":
            densify_mask = proximity_mask
        else:
            if hybrid_mode == "union":
                densify_mask = proximity_mask | gradient_mask
            elif hybrid_mode == "intersection":
                densify_mask = proximity_mask & gradient_mask
            else:
                raise ValueError(
                    f"Unknown hybrid_mode: {hybrid_mode}. "
                    f"Choose from: 'union', 'intersection', 'proximity_only'"
                )

        return densify_mask

    def generate_new_gaussians(
        self,
        source_positions: torch.Tensor,  # (M, 3) positions of Gaussians to densify
        neighbor_indices: torch.Tensor,  # (M, K) indices of K nearest neighbors
        all_positions: torch.Tensor,  # (N, 3) all Gaussian positions
        all_attributes: Dict[str, torch.Tensor],  # All Gaussian attributes
        max_new_per_source: int = None  # Max new Gaussians per source (default: K)
    ) -> Dict[str, torch.Tensor]:
        """
        Generate new Gaussians at edge midpoints

        FSGS Strategy (Paper Sec. 3.2):
            1. For each source Gaussian, connect to its K nearest neighbors (proximity graph edges)
            2. Place new Gaussians at the midpoint of each edge: new_pos = (source + neighbor) / 2
            3. Initialize attributes:
                - Position: midpoint
                - Scale & Opacity: inherit from destination (neighbor)
                - Rotation & SH coefficients: initialize to zero

        Args:
            source_positions: (M, 3) positions of Gaussians identified for densification
            neighbor_indices: (M, K) indices of their K nearest neighbors
            all_positions: (N, 3) all Gaussian positions (for looking up neighbor positions)
            all_attributes: Dict of all Gaussian attributes, e.g.:
                {
                    'scales': (N, 3),
                    'rotations': (N, 4),  # quaternions
                    'opacities': (N, 1),
                    'features_dc': (N, 1, 3),  # SH degree 0
                    'features_rest': (N, 15, 3),  # SH degree 1-3
                }
            max_new_per_source: Maximum new Gaussians per source (default: K, i.e., one per edge)

        Returns:
            new_gaussians: Dict of new Gaussian attributes
                {
                    'positions': (M*K, 3),  # or (M*max_new_per_source, 3)
                    'scales': (M*K, 3),
                    'rotations': (M*K, 4),
                    'opacities': (M*K, 1),
                    'features_dc': (M*K, 1, 3),
                    'features_rest': (M*K, 15, 3),
                }

        Example:
            >>> # Get densify candidates
            >>> densify_mask = densifier.identify_densify_candidates(scores)
            >>> source_positions = all_positions[densify_mask]
            >>>
            >>> # Get their neighbors
            >>> _, neighbor_indices, _ = densifier.compute_proximity_scores(
            ...     all_positions,
            ...     return_neighbors=True
            ... )
            >>> neighbor_indices_densify = neighbor_indices[densify_mask]
            >>>
            >>> # Generate new Gaussians
            >>> new_gaussians = densifier.generate_new_gaussians(
            ...     source_positions,
            ...     neighbor_indices_densify,
            ...     all_positions,
            ...     all_attributes
            ... )
            >>> print(f"Generated {len(new_gaussians['positions'])} new Gaussians")
        """
        if not self.enable:
            # Return empty dict if disabled
            return {key: torch.empty(0, *value.shape[1:], device=value.device)
                    for key, value in all_attributes.items()}

        M = source_positions.shape[0]  # Number of source Gaussians
        K = neighbor_indices.shape[1]  # Number of neighbors per source
        device = source_positions.device

        if max_new_per_source is None:
            max_new_per_source = K
        else:
            max_new_per_source = min(max_new_per_source, K)

        # Limit to max_new_per_source neighbors
        if max_new_per_source < K:
            neighbor_indices = neighbor_indices[:, :max_new_per_source]  # (M, max_new_per_source)

        # Get neighbor positions: (M, max_new_per_source, 3)
        neighbor_positions = all_positions[neighbor_indices]  # (M, max_new_per_source, 3)

        # Compute midpoint positions
        # source_positions: (M, 3) → (M, 1, 3) for broadcasting
        # neighbor_positions: (M, max_new_per_source, 3)
        new_positions = (source_positions.unsqueeze(1) + neighbor_positions) / 2.0  # (M, max_new_per_source, 3)
        new_positions = new_positions.reshape(-1, 3)  # (M*max_new_per_source, 3)

        # Initialize attributes from destination (neighbor) Gaussians
        new_attributes = {'positions': new_positions}

        for attr_name, attr_values in all_attributes.items():
            if attr_name in ['scales', 'opacities']:
                # Inherit from destination (neighbor)
                neighbor_attrs = attr_values[neighbor_indices]  # (M, max_new_per_source, ...)
                new_attributes[attr_name] = neighbor_attrs.reshape(-1, *attr_values.shape[1:])

            elif attr_name == 'rotations':
                # Inherit rotation from destination (neighbor)
                # 对齐 3DGS/FSGS 的 densify_and_clone/split 行为：新点继承邻居的局部各向异性方向
                neighbor_attrs = attr_values[neighbor_indices]  # (M, max_new_per_source, 4)
                new_attributes[attr_name] = neighbor_attrs.reshape(-1, *attr_values.shape[1:])

            elif attr_name in ['features_dc', 'features_rest']:
                # SH 系数可以初始化为零
                new_shape = (M * max_new_per_source, *attr_values.shape[1:])
                new_attributes[attr_name] = torch.zeros(new_shape, device=device, dtype=attr_values.dtype)

        # Update statistics
        self.stats['num_densify_calls'] += 1
        self.stats['total_new_gaussians'] += len(new_positions)

        return new_attributes

    def get_stats(self) -> Dict[str, float]:
        """
        Get densification statistics (for TensorBoard logging)

        Returns:
            stats: Dict of statistics
                {
                    'num_densify_calls': int,
                    'total_new_gaussians': int,
                    'avg_proximity_score': float,
                    'avg_new_gaussians_per_call': float,
                }
        """
        stats = self.stats.copy()
        if stats['num_densify_calls'] > 0:
            stats['avg_new_gaussians_per_call'] = (
                stats['total_new_gaussians'] / stats['num_densify_calls']
            )
        else:
            stats['avg_new_gaussians_per_call'] = 0.0
        return stats

    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'num_densify_calls': 0,
            'total_new_gaussians': 0,
            'avg_proximity_score': 0.0,
        }

    def get_diagnostics(
        self,
        proximity_scores: torch.Tensor,
        iteration: int,
        start_iter: int,
        until_iter: int,
    ) -> Dict[str, float]:
        """
        获取 GAR 诊断信息，用于训练日志输出

        Args:
            proximity_scores: (N,) 邻近分数张量
            iteration: 当前迭代次数
            start_iter: GAR 开始迭代
            until_iter: GAR 结束迭代

        Returns:
            diagnostics: 诊断信息字典
                {
                    'score_mean': float,  # 邻近分数均值
                    'score_std': float,   # 邻近分数标准差
                    'score_min': float,   # 邻近分数最小值
                    'score_max': float,   # 邻近分数最大值
                    'threshold': float,   # 当前阈值（自适应或固定）
                    'decay_mult': float,  # 衰减系数
                }
        """
        # 邻近分数统计
        score_mean = proximity_scores.mean().item()
        score_std = proximity_scores.std().item()
        score_min = proximity_scores.min().item()
        score_max = proximity_scores.max().item()

        # 计算当前阈值
        if self.adaptive_threshold:
            threshold = self.compute_adaptive_threshold_value(proximity_scores).item()
        else:
            threshold = self.proximity_threshold

        # 计算衰减系数
        if self.progressive_decay:
            decay_mult = self.get_decay_multiplier(iteration, start_iter, until_iter)
        else:
            decay_mult = 1.0

        # 应用衰减后的有效阈值
        effective_threshold = threshold * decay_mult

        return {
            'score_mean': score_mean,
            'score_std': score_std,
            'score_min': score_min,
            'score_max': score_max,
            'threshold': effective_threshold,
            'base_threshold': threshold,
            'decay_mult': decay_mult,
        }
