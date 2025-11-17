# GR-Gaussian 技术实现方案

## 核心策略总结 (3-5 句话)

本方案将 GR-Gaussian 的三项核心技术完整迁移到 R²-Gaussian baseline：(1) **De-Init** 通过在初始化脚本中添加 `scipy.ndimage.gaussian_filter` 三维滤波实现，修改 `r2_gaussian/gaussian/initialize.py` 并新增 `--sigma_d` 参数；(2) **Graph 构建与 PGA 梯度增强**通过新建 `r2_gaussian/utils/graph_utils.py` 模块提供 KNN 图管理，依赖 PyTorch Geometric 的 `knn_graph` API，并在 `gaussian_model.py` 的密集化逻辑中注入邻域密度差异项；(3) **Graph Laplacian Regularization** 在 `loss_utils.py` 中新增 `compute_graph_laplacian_loss` 函数并集成到 `train.py` 的损失计算流程。整体设计遵循向后兼容原则，使用 `--use_gr_gaussian` 参数开关控制新功能，预期工期 7-10 天。

---

## 1. 架构设计概览

### 1.1 模块依赖关系图
```
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                       │
│                      (train.py)                              │
└─────┬───────────────────────────────┬────────────────────────┘
      │                               │
      │ 1. Initialization             │ 3. Training Loop
      ▼                               ▼
┌──────────────────┐           ┌──────────────────────┐
│  De-Init Module  │           │  GaussianModel       │
│  (initialize.py) │           │  (gaussian_model.py) │
│                  │           │                      │
│ - gaussian_filter│           │ - build_graph()      │
│ - denoise_fdk()  │           │ - compute_pga_grad() │
└──────────────────┘           │ - densify_and_prune()│
                               └───────┬──────────────┘
                                       │
      ┌────────────────────────────────┴───────────┐
      │ 2. Graph Construction (每 100 iters)       │
      ▼                                            │
┌──────────────────────┐                          │
│  Graph Utils         │                          │
│  (graph_utils.py)    │                          │
│                      │                          │
│ - GaussianGraph      │                          │
│ - build_knn_graph()  │                          │
│ - compute_weights()  │                          │
└──────┬───────────────┘                          │
       │                                           │
       │ 4. Loss Calculation                       │
       ▼                                           ▼
┌────────────────────────────────────────────────────┐
│            Loss Functions (loss_utils.py)          │
│                                                    │
│ - l1_loss()                   (existing)           │
│ - ssim()                      (existing)           │
│ - tv_3d_loss()                (existing)           │
│ + compute_graph_laplacian_loss()  (new)            │
└────────────────────────────────────────────────────┘
```

### 1.2 数据流向
```
FDK Volume (128³)
    │
    ├─> [De-Init] gaussian_filter(σ_d=3) → Denoised Volume
    │                                         │
    │                                         ├─> Thresholding (τ=0.001)
    │                                         │
    │                                         └─> Random Sampling (M=50k points)
    │                                                   │
    ▼                                                   ▼
Gaussians Initialization ◄──────────────────────── (xyz, density)
    │
    ├─> [Training Loop] (iteration 1~30000)
    │       │
    │       ├─> [Graph Construction] (every 100 iters)
    │       │       │
    │       │       └─> KNN Graph (k=6, bidirectional) → Edge Index (2, E)
    │       │
    │       ├─> [Forward Rendering] → Rendered Images
    │       │
    │       ├─> [Backward Pass] → Pixel Gradients
    │       │       │
    │       │       └─> [PGA Enhancement]
    │       │               │
    │       │               ├─> Compute Density Diff (Δρ_ij)
    │       │               │
    │       │               └─> Augmented Grad = Pixel Grad + λ_g * Δρ
    │       │
    │       ├─> [Loss Calculation]
    │       │       │
    │       │       ├─> L1 + SSIM + TV (existing)
    │       │       │
    │       │       └─> + λ_lap * L_lap(Graph)  (new)
    │       │
    │       └─> [Densify & Prune] (using Augmented Grad)
    │
    └─> Final Gaussians → Render CT Volume
```

---

## 2. De-Init 实现方案

### 2.1 修改文件
**主要文件:** `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/gaussian/initialize.py`

### 2.2 核心函数实现
```python
# 在 initialize.py 中添加以下函数

import numpy as np
from scipy.ndimage import gaussian_filter

def denoise_fdk_pointcloud(fdk_volume, sigma_d=3.0, tau=0.001, M=50000, seed=42):
    """
    使用高斯滤波对 FDK 重建的点云进行降噪

    Args:
        fdk_volume: (D, H, W) ndarray, FDK 重建的密度体积
        sigma_d: float, 高斯滤波标准差 (论文推荐 3.0)
        tau: float, 密度阈值，用于过滤空气区域 (论文推荐 0.001)
        M: int, 采样点数量 (论文默认 50000)
        seed: int, 随机种子，确保可复现性

    Returns:
        xyz: (M, 3) ndarray, 高斯核位置 (归一化到 [-1, 1]³)
        density: (M,) ndarray, 对应的中心密度值
    """
    np.random.seed(seed)

    # Step 1: 三维高斯滤波
    print(f"[De-Init] Applying Gaussian filter with σ_d={sigma_d}...")
    denoised_volume = gaussian_filter(fdk_volume, sigma=sigma_d, mode='constant')

    # Step 2: 阈值过滤
    valid_mask = denoised_volume > tau
    num_valid = np.sum(valid_mask)
    print(f"[De-Init] Valid voxels after thresholding (τ={tau}): {num_valid}")

    # Step 3: 提取有效体素坐标
    valid_indices = np.argwhere(valid_mask)  # (N, 3)
    valid_densities = denoised_volume[valid_mask]  # (N,)

    # Step 4: 随机采样 M 个点
    if num_valid <= M:
        print(f"[De-Init] Warning: Only {num_valid} valid voxels, using all")
        xyz = valid_indices.astype(np.float32)
        density = valid_densities
    else:
        sample_indices = np.random.choice(num_valid, M, replace=False)
        xyz = valid_indices[sample_indices].astype(np.float32)
        density = valid_densities[sample_indices]

    # Step 5: 坐标归一化到 [-1, 1]³ (R²-GS 约定)
    volume_shape = np.array(fdk_volume.shape, dtype=np.float32)
    xyz = (xyz / volume_shape - 0.5) * 2.0

    print(f"[De-Init] Sampled {len(xyz)} points from denoised FDK volume")
    print(f"[De-Init] Density range: [{density.min():.4f}, {density.max():.4f}]")

    return xyz, density
```

### 2.3 修改现有初始化逻辑
在 `initialize_gaussian()` 函数中集成 De-Init：

```python
def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, loaded_iter=None):
    # ... (现有代码保持不变，直到加载点云部分)

    if ply_type == "npy":
        point_cloud = np.load(ply_path)

        # 🌟 GR-Gaussian: De-Init 降噪
        if args.use_gr_gaussian and args.enable_denoise_init:
            print("\n" + "="*60)
            print("🌟 [GR-Gaussian] De-Init Enabled")
            print("="*60)

            # 假设 point_cloud 是从 FDK volume 生成的
            # 需要重新加载原始 FDK volume 进行降噪
            fdk_volume_path = ply_path.replace("init_", "fdk_volume_")
            fdk_volume_path = fdk_volume_path.replace(".npy", "_volume.npy")

            if os.path.exists(fdk_volume_path):
                fdk_volume = np.load(fdk_volume_path)
                xyz, density = denoise_fdk_pointcloud(
                    fdk_volume,
                    sigma_d=args.sigma_d,
                    tau=args.denoise_tau,
                    M=args.denoise_num_points,
                    seed=args.seed
                )
                density = density[:, np.newaxis]  # (M,) → (M, 1)
            else:
                print(f"⚠️  Warning: FDK volume not found at {fdk_volume_path}")
                print("   Falling back to standard initialization")
                xyz = point_cloud[:, :3]
                density = point_cloud[:, 3:4]
        else:
            # 标准初始化流程
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]

    # ... (后续代码不变)
    gaussians.create_from_pcd(xyz, density, 1.0)
    return loaded_iter
```

### 2.4 新增参数 (arguments.py)
在 `ModelParams` 类中添加：

```python
class ModelParams:
    # ... (现有参数)

    # GR-Gaussian: De-Init 参数
    use_gr_gaussian: bool = False
    enable_denoise_init: bool = True  # 默认开启（当 use_gr_gaussian=True 时）
    sigma_d: float = 3.0  # 高斯滤波标准差
    denoise_tau: float = 0.001  # 密度阈值
    denoise_num_points: int = 50000  # 采样点数量
```

### 2.5 实现复杂度
- **开发时间:** 1-2 天
- **测试重点:**
  - 验证滤波后体积的噪声抑制效果（可视化对比）
  - 确认采样点的空间分布合理性
  - 检查训练初期的收敛速度对比

---

## 3. Graph 构建模块

### 3.1 新建文件
**文件路径:** `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/graph_utils.py`

### 3.2 完整实现代码
```python
"""
GR-Gaussian Graph Utilities
构建和管理高斯核的 KNN 图结构
"""

import torch
import torch.nn.functional as F
try:
    from torch_geometric.nn import knn_graph
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("⚠️  PyTorch Geometric not found, using fallback KNN implementation")


class GaussianGraph:
    """
    管理高斯核的 KNN 图结构

    图构建策略:
        - 使用 KNN 双向连接确保对称性
        - 边权重基于欧氏距离的高斯衰减
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

        if HAS_TORCH_GEOMETRIC:
            # 使用 PyTorch Geometric 的高效 KNN 实现
            edge_index = knn_graph(
                positions,
                k=self.k,
                loop=False,  # 不包含自环
                flow='source_to_target'
            )
        else:
            # Fallback: 纯 PyTorch 实现 (较慢)
            edge_index = self._pytorch_knn_graph(positions)

        # 强制双向连接 (对称化)
        edge_index = self._symmetrize_edges(edge_index)

        self.edge_index = edge_index.to(self.device)
        return self.edge_index

    def _pytorch_knn_graph(self, positions):
        """
        纯 PyTorch 实现 KNN (Fallback)
        复杂度: O(M²) - 仅在没有 PyG 时使用
        """
        # 计算所有点对距离矩阵 (M, M)
        dist_matrix = torch.cdist(positions, positions, p=2)

        # 找到每个点的 k 个最近邻 (不包括自身)
        # topk 返回 (values, indices), shape: (M, k)
        knn_dists, knn_indices = torch.topk(
            dist_matrix,
            k=self.k + 1,  # +1 因为第一个是自身
            largest=False,  # 最小的 k 个
            dim=1
        )

        # 移除自环 (第一列是自身，距离为 0)
        knn_indices = knn_indices[:, 1:]  # (M, k)

        # 构建边索引
        src = torch.arange(self.num_nodes, device=positions.device).unsqueeze(1).repeat(1, self.k)  # (M, k)
        dst = knn_indices  # (M, k)

        edge_index = torch.stack([src.flatten(), dst.flatten()], dim=0)  # (2, M*k)
        return edge_index

    def _symmetrize_edges(self, edge_index):
        """
        强制双向连接：仅保留互为 KNN 的边

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
```

### 3.3 依赖安装脚本
创建 `/home/qyhu/Documents/r2_ours/r2_gaussian/scripts/install_torch_geometric.sh`:

```bash
#!/bin/bash
# PyTorch Geometric 安装脚本
# 根据当前 PyTorch 版本自动选择兼容的 PyG 版本

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

echo "Detected PyTorch version: $TORCH_VERSION"
echo "Detected CUDA version: $CUDA_VERSION"

# 安装 PyG (使用官方推荐的方式)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION//.}.html

# 验证安装
python -c "from torch_geometric.nn import knn_graph; print('✅ PyTorch Geometric installed successfully')"
```

---

## 4. PGA (Pixel-Graph-Aware Gradient) 实现

### 4.1 修改文件
**主要文件:** `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/gaussian/gaussian_model.py`

### 4.2 在 GaussianModel 类中添加图管理

```python
# 在 GaussianModel.__init__() 中添加

from r2_gaussian.utils.graph_utils import GaussianGraph

class GaussianModel:
    def __init__(self, scale_bound=None, use_student_t=False):
        # ... (现有初始化代码)

        # GR-Gaussian: Graph 管理
        self.graph = None  # GaussianGraph 对象
        self.graph_update_interval = 100  # 每 100 iterations 重建图
        self.last_graph_update = 0
        self.pga_lambda_g = 1e-4  # PGA 正则化权重

    def setup_gr_gaussian_graph(self, k=6, lambda_g=1e-4, update_interval=100):
        """
        初始化 GR-Gaussian 图结构

        Args:
            k: KNN 邻居数
            lambda_g: PGA 梯度增强权重
            update_interval: 图重建间隔（iterations）
        """
        self.graph = GaussianGraph(k=k, device=self._xyz.device)
        self.pga_lambda_g = lambda_g
        self.graph_update_interval = update_interval
        print(f"[GR-Gaussian] Graph initialized: k={k}, λ_g={lambda_g}")

    def update_graph_if_needed(self, iteration):
        """
        根据迭代次数决定是否重建图

        在以下情况重建:
            1. 图从未构建
            2. 经过 update_interval 次迭代
            3. 刚执行过密集化/剪枝（高斯核数量变化）
        """
        if self.graph is None:
            return

        should_update = (
            self.graph.edge_index is None or
            iteration - self.last_graph_update >= self.graph_update_interval
        )

        if should_update:
            print(f"[GR-Gaussian] Rebuilding graph at iteration {iteration}...")
            self.graph.build_knn_graph(self._xyz.detach())
            self.graph.compute_edge_weights(self._xyz.detach())
            self.last_graph_update = iteration
```

### 4.3 修改密集化逻辑中的梯度计算

找到 `GaussianModel` 中处理梯度累积的部分（通常在 `densify_and_prune` 或相关函数中），添加 PGA 增强：

```python
def compute_pga_augmented_gradient(self, pixel_gradients):
    """
    计算 PGA 增强后的梯度

    增强公式:
        g_aug = g_pixel + λ_g * (Σ Δρ_ij / k)

    Args:
        pixel_gradients: (M,) 原始像素梯度范数

    Returns:
        augmented_gradients: (M,) 增强后梯度
    """
    if self.graph is None or self.graph.edge_index is None:
        return pixel_gradients

    densities = self.get_density.detach()  # (M,)

    # 计算边的密度差异
    density_diffs = self.graph.compute_density_differences(densities)  # (E,)

    # 聚合到每个节点
    src, dst = self.graph.edge_index
    avg_density_diff = torch.zeros_like(densities)
    avg_density_diff.scatter_add_(0, src, density_diffs)

    # 归一化（每个节点最多 k 个邻居）
    node_degree = torch.zeros_like(densities)
    node_degree.scatter_add_(0, src, torch.ones_like(density_diffs))
    avg_density_diff = avg_density_diff / (node_degree + 1e-8)

    # 增强梯度
    augmented_gradients = pixel_gradients + self.pga_lambda_g * avg_density_diff

    return augmented_gradients

# 在密集化函数中调用 (修改现有代码)
def densify_and_prune(self, ...):
    # ... (现有代码获取 pixel_gradients)

    # 🌟 GR-Gaussian: PGA 梯度增强
    if hasattr(self, 'graph') and self.graph is not None:
        pixel_gradients = self.compute_pga_augmented_gradient(pixel_gradients)

    # ... (后续使用 augmented gradients 进行密集化判断)
```

### 4.4 在训练循环中集成
在 `train.py` 的主循环中添加图更新逻辑：

```python
# train.py 训练循环

for iteration in range(first_iter, opt.iterations + 1):
    # ... (前向渲染、损失计算等)

    # 🌟 GR-Gaussian: 更新图结构
    if args.use_gr_gaussian:
        gaussians.update_graph_if_needed(iteration)

    # ... (反向传播、密集化等)
```

---

## 5. Graph Laplacian Regularization

### 5.1 修改文件
**主要文件:** `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/loss_utils.py`

### 5.2 新增损失函数

```python
# 在 loss_utils.py 中添加

def compute_graph_laplacian_loss(gaussians, graph, lambda_lap=8e-4):
    """
    计算图拉普拉斯正则化损失

    公式:
        L_lap = Σ_{(i,j)∈E} w_ij * (ρ_i - ρ_j)²

    Args:
        gaussians: GaussianModel 对象
        graph: GaussianGraph 对象
        lambda_lap: 正则化权重 (论文推荐 8e-4)

    Returns:
        lap_loss: 标量损失
    """
    if graph is None or graph.edge_index is None:
        return torch.tensor(0.0, device='cuda')

    densities = gaussians.get_density  # (M,)
    src, dst = graph.edge_index  # (E,), (E,)

    # 密度差异平方
    density_diff_sq = (densities[src] - densities[dst]) ** 2  # (E,)

    # 边权重（如果已计算）
    if graph.edge_weights is not None:
        weights = graph.edge_weights
    else:
        # 如果未计算权重，使用均匀权重
        weights = torch.ones_like(density_diff_sq)

    # 加权求和
    lap_loss = lambda_lap * torch.sum(weights * density_diff_sq)

    return lap_loss
```

### 5.3 集成到训练损失
在 `train.py` 的损失计算部分添加：

```python
# train.py 损失计算部分

# 现有损失项
Ll1 = l1_loss(image, gt_image)
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

# TV 损失 (如果启用)
if use_tv:
    tv_loss = tv_3d_loss(volume, reduction="sum")
    loss += opt.lambda_tv * tv_loss

# 🌟 GR-Gaussian: Graph Laplacian 损失
if args.use_gr_gaussian and hasattr(gaussians, 'graph'):
    lap_loss = compute_graph_laplacian_loss(
        gaussians,
        gaussians.graph,
        lambda_lap=args.lambda_lap
    )
    loss += lap_loss

    # 日志记录
    if iteration % 10 == 0:
        tb_writer.add_scalar('Loss/graph_laplacian', lap_loss.item(), iteration)
```

---

## 6. 配置文件模板

### 6.1 创建配置文件
**文件路径:** `/home/qyhu/Documents/r2_ours/r2_gaussian/configs/gr_gaussian_foot3.yaml`

```yaml
# GR-Gaussian 配置文件 - Foot 3 Views

# 基础训练参数
iterations: 30000
position_lr_init: 0.00016
position_lr_final: 0.0000016
scaling_lr: 0.005
rotation_lr: 0.001
density_lr: 0.05

# GR-Gaussian 开关
use_gr_gaussian: true

# De-Init 参数
enable_denoise_init: true
sigma_d: 3.0  # 高斯滤波标准差
denoise_tau: 0.001  # 密度阈值
denoise_num_points: 50000  # 采样点数

# Graph 构建参数
k_neighbors: 6  # KNN 邻居数
graph_update_interval: 100  # 图重建间隔（iterations）

# PGA 参数
lambda_g: 1.0e-4  # 梯度增强权重

# Graph Laplacian 参数
lambda_lap: 8.0e-4  # 图拉普拉斯权重

# 现有损失权重
lambda_dssim: 0.25
lambda_tv: 0.05

# 数据集配置
source_path: "data/369/foot"
num_views: 3
resolution: 1
```

### 6.2 参数加载脚本
在 `arguments.py` 中添加从 YAML 加载配置的函数：

```python
import yaml

def load_gr_gaussian_config(config_path):
    """
    从 YAML 文件加载 GR-Gaussian 配置

    Args:
        config_path: YAML 配置文件路径

    Returns:
        config_dict: 配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 在命令行参数解析后调用
if args.gr_config is not None:
    gr_config = load_gr_gaussian_config(args.gr_config)
    for key, value in gr_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
```

---

## 7. 代码修改清单

### 7.1 新建文件
| 文件路径 | 用途 | 核心内容 |
|---------|------|---------|
| `r2_gaussian/utils/graph_utils.py` | Graph 构建与管理 | `GaussianGraph` 类，KNN 图构建 |
| `configs/gr_gaussian_foot3.yaml` | 超参数配置 | 所有 GR-Gaussian 参数 |
| `scripts/install_torch_geometric.sh` | 依赖安装 | PyG 自动安装脚本 |

### 7.2 修改现有文件
| 文件路径 | 修改内容 | 函数/类 |
|---------|---------|---------|
| `r2_gaussian/gaussian/initialize.py` | 添加 `denoise_fdk_pointcloud()` | 降噪初始化函数 |
| `r2_gaussian/gaussian/initialize.py` | 修改 `initialize_gaussian()` | 集成 De-Init 逻辑 |
| `r2_gaussian/gaussian/gaussian_model.py` | 添加 `setup_gr_gaussian_graph()` | 图初始化 |
| `r2_gaussian/gaussian/gaussian_model.py` | 添加 `update_graph_if_needed()` | 图更新逻辑 |
| `r2_gaussian/gaussian/gaussian_model.py` | 添加 `compute_pga_augmented_gradient()` | PGA 梯度增强 |
| `r2_gaussian/gaussian/gaussian_model.py` | 修改 `densify_and_prune()` | 使用增强梯度 |
| `r2_gaussian/utils/loss_utils.py` | 添加 `compute_graph_laplacian_loss()` | 图拉普拉斯损失 |
| `train.py` | 导入新模块 | `from r2_gaussian.utils.graph_utils import GaussianGraph` |
| `train.py` | 初始化图结构 | 在高斯模型初始化后调用 `setup_gr_gaussian_graph()` |
| `train.py` | 训练循环添加图更新 | `gaussians.update_graph_if_needed(iteration)` |
| `train.py` | 损失计算添加 L_lap | `loss += compute_graph_laplacian_loss(...)` |
| `r2_gaussian/arguments.py` | 添加 GR-Gaussian 参数 | `use_gr_gaussian`, `sigma_d`, `k_neighbors` 等 |

### 7.3 向后兼容性保证

**关键策略：使用 `try-except` 和参数开关**

```python
# 示例 1: 可选依赖加载
try:
    from r2_gaussian.utils.graph_utils import GaussianGraph
    HAS_GRAPH_UTILS = True
except ImportError:
    HAS_GRAPH_UTILS = False

# 示例 2: 条件功能启用
if args.use_gr_gaussian and HAS_GRAPH_UTILS:
    gaussians.setup_gr_gaussian_graph(k=args.k_neighbors)
else:
    print("📦 Running without GR-Gaussian enhancements")

# 示例 3: 损失计算防护
lap_loss = (
    compute_graph_laplacian_loss(gaussians, gaussians.graph, args.lambda_lap)
    if args.use_gr_gaussian and hasattr(gaussians, 'graph')
    else torch.tensor(0.0, device='cuda')
)
```

---

## 8. 依赖库检查清单

### 8.1 必需依赖
| 库名称 | 版本要求 | 用途 | 安装命令 |
|-------|---------|------|---------|
| `scipy` | ≥1.7.0 | 高斯滤波 (De-Init) | `pip install scipy` |
| `torch-geometric` | ≥2.3.0 | KNN 图构建 | 见下方详细脚本 |
| `torch-scatter` | 匹配 PyTorch 版本 | 图操作加速 | PyG 依赖项 |
| `torch-sparse` | 匹配 PyTorch 版本 | 稀疏矩阵操作 | PyG 依赖项 |

### 8.2 PyTorch Geometric 安装

**步骤 1: 检查当前环境**
```bash
conda activate r2_gaussian_new
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
```

**步骤 2: 安装 PyG**
```bash
# 假设输出: PyTorch 1.13.0, CUDA 11.7
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

**步骤 3: 验证安装**
```bash
python -c "from torch_geometric.nn import knn_graph; import torch; x=torch.randn(100,3).cuda(); e=knn_graph(x, k=6); print('✅ PyG working, edges:', e.shape)"
```

### 8.3 CUDA 兼容性验证脚本

创建 `/home/qyhu/Documents/r2_ours/r2_gaussian/scripts/verify_gr_dependencies.py`:

```python
"""
验证 GR-Gaussian 所需依赖的安装和兼容性
"""

import sys

def check_scipy():
    try:
        import scipy
        from scipy.ndimage import gaussian_filter
        print(f"✅ scipy {scipy.__version__}")
        return True
    except ImportError as e:
        print(f"❌ scipy not found: {e}")
        return False

def check_torch_geometric():
    try:
        import torch
        from torch_geometric.nn import knn_graph

        # 测试 CUDA 兼容性
        x = torch.randn(100, 3).cuda()
        edge_index = knn_graph(x, k=6)

        print(f"✅ PyTorch Geometric (CUDA compatible)")
        print(f"   Test: 100 points → {edge_index.shape[1]} edges")
        return True
    except ImportError as e:
        print(f"❌ PyTorch Geometric not found: {e}")
        return False
    except RuntimeError as e:
        print(f"⚠️  PyG installed but CUDA test failed: {e}")
        return False

def check_yaml():
    try:
        import yaml
        print(f"✅ PyYAML")
        return True
    except ImportError:
        print(f"❌ PyYAML not found (needed for config files)")
        return False

if __name__ == "__main__":
    print("="*60)
    print("GR-Gaussian Dependency Check")
    print("="*60)

    checks = {
        "scipy": check_scipy(),
        "torch_geometric": check_torch_geometric(),
        "yaml": check_yaml()
    }

    print("\n" + "="*60)
    if all(checks.values()):
        print("🎉 All dependencies satisfied!")
        sys.exit(0)
    else:
        print("⚠️  Some dependencies missing, please install:")
        if not checks["scipy"]:
            print("   pip install scipy")
        if not checks["torch_geometric"]:
            print("   See scripts/install_torch_geometric.sh")
        if not checks["yaml"]:
            print("   pip install pyyaml")
        sys.exit(1)
```

---

## 9. 风险评估与缓解方案

### 9.1 PyTorch Geometric 版本兼容性
**风险等级:** 🟡 中等

**潜在问题:**
- PyG 依赖 PyTorch 的特定版本
- CUDA 版本不匹配会导致运行时错误
- 安装过程可能失败

**缓解方案:**
1. **主方案:** 使用 PyG 官方推荐的安装命令，自动匹配 CUDA 版本
2. **备用方案:** 实现纯 PyTorch 的 KNN (已在 `graph_utils.py` 中提供 fallback)
   - 性能损失：约 10-20% (仅图构建阶段)
   - 总训练时间影响：< 5%

**测试计划:**
```bash
# 在 r2_gaussian_new 环境中测试
conda activate r2_gaussian_new
python scripts/verify_gr_dependencies.py
```

### 9.2 图构建计算开销
**风险等级:** 🟢 低

**性能分析:**
- 图构建频率：每 100 iterations
- 单次 KNN 时间（50k 点）：约 50-100 ms (PyG) / 200-300 ms (PyTorch)
- 总训练时间增加：< 1%

**优化策略:**
1. 缓存边索引，仅在密集化后重建
2. 使用 GPU 加速 KNN 搜索
3. 如果内存允许，降低重建频率（200 iterations）

### 9.3 内存占用增加
**风险等级:** 🟢 低

**内存估算:**
```
高斯核数量: M = 50,000
邻居数: k = 6
边数: E ≈ k * M = 300,000

边索引 (2, E): 2 * 300k * 4 bytes (int32) = 2.4 MB
边权重 (E,): 300k * 4 bytes (float32) = 1.2 MB
总增加: ~4 MB (可忽略)
```

### 9.4 超参数敏感性
**风险等级:** 🟡 中等

**关键超参数:**
- `k` (邻居数): 论文推荐 6，需验证在 3 视角下是否最优
- `λ_g` (PGA 权重): 1e-4，可能需要调整到 5e-5 ~ 2e-4
- `λ_lap` (Laplacian 权重): 8e-4，需平衡平滑性和边界保留

**缓解方案:**
1. **阶段 1:** 使用论文默认值快速验证功能
2. **阶段 2:** 在 foot 数据集上进行小范围网格搜索
   ```python
   # 搜索空间
   k_values = [4, 6, 8]
   lambda_g_values = [5e-5, 1e-4, 2e-4]
   lambda_lap_values = [4e-4, 8e-4, 1.2e-3]
   ```
3. **预留时间:** 2-3 天用于超参数调优

---

## 10. 验证测试计划

### 10.1 单元测试：KNN 图构建
**测试文件:** `tests/test_graph_utils.py`

```python
import torch
from r2_gaussian.utils.graph_utils import GaussianGraph

def test_knn_graph_construction():
    """测试 KNN 图构建的正确性"""
    # 创建简单的 3x3x3 网格点
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ], device='cuda')

    graph = GaussianGraph(k=2, device='cuda')
    edge_index = graph.build_knn_graph(positions)

    # 验证：每个点应该有 2 个邻居（双向）
    src, dst = edge_index
    for i in range(4):
        num_neighbors = (src == i).sum().item()
        assert num_neighbors <= 2, f"Node {i} has {num_neighbors} neighbors (expected ≤2)"

    print("✅ KNN graph construction test passed")

def test_symmetry():
    """测试边的双向对称性"""
    positions = torch.randn(100, 3).cuda()
    graph = GaussianGraph(k=6, device='cuda')
    edge_index = graph.build_knn_graph(positions)

    # 验证：(i,j) ∈ E ⟹ (j,i) ∈ E
    src, dst = edge_index
    edge_set = set(zip(src.cpu().tolist(), dst.cpu().tolist()))

    for i, j in edge_set:
        assert (j, i) in edge_set, f"Edge ({i},{j}) is not symmetric"

    print("✅ Edge symmetry test passed")

if __name__ == "__main__":
    test_knn_graph_construction()
    test_symmetry()
```

### 10.2 集成测试：梯度增强效果
**测试脚本:** `tests/test_pga_gradient.py`

```python
import torch
from r2_gaussian.gaussian.gaussian_model import GaussianModel

def test_pga_gradient_enhancement():
    """测试 PGA 梯度增强的数值正确性"""
    # 创建简单的高斯模型
    gaussians = GaussianModel()
    # ... (初始化高斯核)

    # 构建图
    gaussians.setup_gr_gaussian_graph(k=6, lambda_g=1e-4)
    gaussians.update_graph_if_needed(iteration=0)

    # 模拟像素梯度
    pixel_gradients = torch.randn(gaussians.get_xyz.shape[0]).cuda()

    # 计算增强梯度
    aug_gradients = gaussians.compute_pga_augmented_gradient(pixel_gradients)

    # 验证：增强梯度应该 ≥ 原始梯度（因为加了正项）
    assert (aug_gradients >= pixel_gradients - 1e-6).all(), "Augmented gradients should be ≥ pixel gradients"

    print("✅ PGA gradient enhancement test passed")
```

### 10.3 性能测试：训练时间对比
**测试脚本:** `scripts/benchmark_gr_gaussian.sh`

```bash
#!/bin/bash
# 对比 baseline 和 GR-Gaussian 的训练时间

# Baseline (1000 iterations)
echo "Testing baseline..."
python train.py \
    --config configs/baseline_foot3.yaml \
    --iterations 1000 \
    --eval \
    --test_iterations 1000 \
    > logs/baseline_1k.log 2>&1

# GR-Gaussian (1000 iterations)
echo "Testing GR-Gaussian..."
python train.py \
    --config configs/gr_gaussian_foot3.yaml \
    --iterations 1000 \
    --eval \
    --test_iterations 1000 \
    > logs/gr_gaussian_1k.log 2>&1

# 提取训练时间
baseline_time=$(grep "Total training time" logs/baseline_1k.log | awk '{print $4}')
gr_time=$(grep "Total training time" logs/gr_gaussian_1k.log | awk '{print $4}')

echo "Baseline: ${baseline_time}s"
echo "GR-Gaussian: ${gr_time}s"
echo "Overhead: $(echo "scale=2; ($gr_time - $baseline_time) / $baseline_time * 100" | bc)%"
```

---

## 11. 实施时间表

### 第 1-2 天：De-Init 实现与验证
- [ ] 实现 `denoise_fdk_pointcloud()` 函数
- [ ] 修改 `initialize_gaussian()` 集成降噪
- [ ] 添加命令行参数
- [ ] 可视化对比：FDK vs. Denoised FDK
- [ ] 运行 1000 iterations 验证收敛速度

### 第 3 天：PyTorch Geometric 环境搭建
- [ ] 安装 PyG 并验证 CUDA 兼容性
- [ ] 运行 `verify_gr_dependencies.py`
- [ ] 测试 KNN 性能（PyG vs. PyTorch fallback）

### 第 4-5 天：Graph 构建与 PGA 实现
- [ ] 实现 `graph_utils.py` 完整代码
- [ ] 在 `GaussianModel` 中添加图管理
- [ ] 实现 `compute_pga_augmented_gradient()`
- [ ] 修改密集化逻辑使用增强梯度
- [ ] 单元测试：图构建正确性

### 第 6 天：Graph Laplacian 损失
- [ ] 实现 `compute_graph_laplacian_loss()`
- [ ] 集成到 `train.py` 损失计算
- [ ] 添加 TensorBoard 日志记录

### 第 7 天：集成测试与调试
- [ ] 完整训练 1000 iterations
- [ ] 检查损失曲线是否收敛
- [ ] 验证 PSNR/SSIM 指标
- [ ] 性能分析（训练时间、内存占用）

### 第 8-10 天：超参数调优与实验
- [ ] 使用论文默认值训练 30000 iterations
- [ ] 对比 baseline 结果
- [ ] 网格搜索调优 `k`, `λ_g`, `λ_lap`
- [ ] 生成可视化切片和定量报告

---

## 12. 交付物检查清单

### 代码交付
- [ ] 所有新文件已创建并通过语法检查
- [ ] 所有修改的文件已备份原始版本
- [ ] Git commit 记录清晰，包含 `[GR-Gaussian]` 标签
- [ ] 代码中包含详细的中文注释

### 测试交付
- [ ] 单元测试脚本 `tests/test_graph_utils.py`
- [ ] 集成测试脚本 `tests/test_pga_gradient.py`
- [ ] 性能基准测试 `scripts/benchmark_gr_gaussian.sh`
- [ ] 依赖验证脚本 `scripts/verify_gr_dependencies.py`

### 文档交付
- [ ] 配置文件 `configs/gr_gaussian_foot3.yaml`
- [ ] 安装指南 `docs/gr_gaussian_setup.md`
- [ ] 超参数调优记录 `cc-agent/experiments/gr_gaussian_tuning.md`
- [ ] 实验结果报告 `cc-agent/experiments/gr_gaussian_results.md`

### 向后兼容性
- [ ] `--use_gr_gaussian=false` 时程序正常运行
- [ ] 不依赖 PyG 时 fallback 正常工作
- [ ] 现有 checkpoint 可正常加载

---

## 需要您的批准

### 实施确认
- [ ] 是否批准上述技术方案？
- [ ] 是否同意安装 PyTorch Geometric？
- [ ] 预计工期 7-10 天是否可接受？

### 优先级调整
如果需要加速实施，可选择以下简化方案：
- **方案 A (推荐):** 完整实施，工期 7-10 天
- **方案 B (快速):** 仅 De-Init + Graph Laplacian，跳过 PGA，工期 4-5 天
- **方案 C (最小):** 仅 De-Init，工期 2-3 天

### 技术疑问
1. 是否需要在其他数据集（liver/pancreas）上同步测试？
2. 是否需要医学专家评估视觉质量？
3. 是否需要与 CoR-GS 功能集成（如果已实施）？

---

**文档版本:** v1.0
**生成时间:** 2025-11-17
**作者:** 3DGS Expert
**状态:** 等待用户批准
**字数:** 2487 字
