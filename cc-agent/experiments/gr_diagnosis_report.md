# GR-Gaussian 实验失败诊断报告

**生成时间:** 2025-11-18
**诊断 Agent:** 深度学习调参与分析专家
**实验版本:** commit 290569d
**诊断状态:** ✅ 根因已确认

---

## 【核心结论】

**GR-Gaussian 实验效果远低于 baseline 的根本原因：Graph Regularization 功能完全没有生效。**

关键证据：
1. **train.py:154 行硬编码 `gr_graph = None`**，导致图结构从未被初始化
2. **loss_utils.py:299-301 行**：当 `graph=None` 时直接返回 0 损失
3. **缺失 GaussianGraph 类实现**：整个代码库中不存在 `GaussianGraph` 类定义
4. **即使参数配置正确**（`enable_graph_laplacian=True`, `graph_lambda_lap=0.0008`），损失函数始终返回 0，对训练无任何影响

实际情况：GR 实验本质上就是 baseline 实验，但由于配置了不同的超参数（如 `iterations=30000`, `densify_until_iter=15000`），导致结果略有差异，但这与 Graph Laplacian 正则化毫无关系。

---

## 【详细分析】

### 1. 代码缺陷清单

#### **缺陷 #1: Graph 结构被硬编码禁用（致命）**

**文件:** `train.py`
**位置:** 第 153-155 行
**严重程度:** 🔴 致命 (P0)

```python
# ❌ 禁用 GR-Gaussian（不确定实现是否正确）
gr_graph = None
print("⚠️ [R²] Graph Regularization disabled (focus on FSGS)")
```

**影响:** 即使用户配置 `enable_graph_laplacian=True`，`gr_graph` 始终为 `None`，导致后续所有依赖图结构的代码失效。

**历史原因:** 根据注释，开发者对 GR-Gaussian 实现的正确性存疑，因此硬编码禁用。

---

#### **缺陷 #2: Graph Laplacian 损失函数提前返回（致命）**

**文件:** `r2_gaussian/utils/loss_utils.py`
**位置:** 第 299-301 行
**严重程度:** 🔴 致命 (P0)

```python
# 🚨 [GR-Gaussian 优化] 如果没有预构建图,直接返回零损失,避免昂贵的 KNN 计算
# 在 iteration 1000 前,graph 尚未构建,此时跳过 Graph Laplacian 损失
return torch.tensor(0.0, device=xyz.device, requires_grad=True)

# 下面的 GPU fallback 代码被禁用,因为它太慢了
# 尝试GPU加速版本（优先）
```

**影响:** 当 `graph=None` 时，损失函数立即返回 0，不执行任何 Graph Laplacian 计算。这意味着：
- 在 train.py 第 666-674 行调用 `compute_graph_laplacian_loss()` 时，实际返回值永远是 0
- 即使添加到总损失中（`LossDict[f"loss_gs{i}"] += graph_laplacian_loss`），对梯度和优化过程无任何影响

---

#### **缺陷 #3: GaussianGraph 类完全缺失（致命）**

**文件:** 无
**位置:** N/A
**严重程度:** 🔴 致命 (P0)

**发现:**
- 使用 `grep -r "class GaussianGraph"` 搜索整个代码库，未找到任何定义
- 使用 serena MCP `find_symbol` 搜索，返回空结果
- train.py 和 loss_utils.py 中引用了 `GaussianGraph` 对象，但该类从未被实现

**影响:**
- train.py:656-663 行尝试调用 `gr_graph.build_knn_graph()` 和 `gr_graph.compute_edge_weights()`，但由于 `gr_graph=None`，这些代码永远不会执行
- 缺少图结构初始化代码（应该在训练初始化阶段创建 `GaussianGraph` 实例）

---

#### **缺陷 #4: Graph 更新逻辑形同虚设（高优先级）**

**文件:** `train.py`
**位置:** 第 654-663 行
**严重程度:** 🟠 高 (P1)

```python
# 🌟 [GR-Gaussian] 图更新与图拉普拉斯正则化
if dataset.enable_graph_laplacian:
    # 更新图结构 (每 graph_update_interval 次迭代,从 iteration 100 开始)
    if gr_graph is not None and iteration > 0 and iteration % dataset.graph_update_interval == 0:
        with torch.no_grad():
            xyz = gaussians.get_xyz.detach()
            gr_graph.build_knn_graph(xyz)
            gr_graph.compute_edge_weights(xyz)
            if iteration % 500 == 0:
                print(f"[GR-Gaussian] Rebuilt graph at iteration {iteration}: "
                      f"{gr_graph.num_nodes} nodes, {gr_graph.edge_index.shape[1]} edges")
```

**影响:** 由于 `gr_graph=None`，条件 `gr_graph is not None` 永远为 `False`，此段代码永不执行。

---

#### **缺陷 #5: Graph Laplacian 损失计算频率过低（次要）**

**文件:** `train.py`
**位置:** 第 666 行
**严重程度:** 🟡 中 (P2)

```python
# 计算图拉普拉斯损失 - 添加延迟启动和频率限制
if iteration > 5000 and iteration % 500 == 0:  # 延迟启动 + 每500次迭代计算一次
```

**问题分析:**
- **延迟启动过晚:** iteration > 5000 意味着前 5000 步（约占总训练的 16.7%）完全没有 Graph Laplacian 正则化
- **计算频率过低:** 每 500 次迭代才计算一次，在 30000 步训练中仅计算 50 次
- **与论文不符:** GR-Gaussian 论文建议每次迭代或至少每 100 次迭代计算一次

**实际影响:** 即使修复了前面的致命缺陷，这个配置也会导致正则化效果大幅削弱。

---

#### **缺陷 #6: 缺少 Tensorboard 日志记录（次要）**

**文件:** `train.py`
**位置:** 第 666-674 行
**严重程度:** 🟢 低 (P3)

**问题:** 没有将 `graph_laplacian_loss` 记录到 tensorboard，导致无法在实验日志中验证损失是否生效。

**建议添加:**
```python
if iteration % 100 == 0:
    tb_writer.add_scalar("Loss/graph_laplacian", graph_laplacian_loss.item(), iteration)
```

---

### 2. 与 GR-Gaussian 论文的差异对比

| 组件 | 论文要求 | 当前实现 | 状态 |
|------|---------|---------|------|
| **图结构初始化** | 在训练开始时用 KNN 构建图 | ❌ `gr_graph=None`，从未初始化 | 完全缺失 |
| **图更新频率** | 每 1000 次迭代重建一次 | ✅ 配置正确 (`graph_update_interval=1000`) | 配置正确，但未生效 |
| **KNN 邻居数** | k=6 | ✅ 配置正确 (`graph_k=6`) | 配置正确，但未生效 |
| **正则化权重** | λ_lap = 8e-4 | ✅ 配置正确 (`graph_lambda_lap=0.0008`) | 配置正确，但未生效 |
| **损失计算频率** | 每次迭代或每 100 次 | ❌ 每 500 次，且延迟到 5000 步后 | 过于保守 |
| **GaussianGraph 类** | 需实现 `build_knn_graph()`, `compute_edge_weights()` | ❌ 类不存在 | 完全缺失 |
| **边权重计算** | 基于高斯点间距离的 RBF 核 | ❌ 未实现 | 完全缺失 |
| **Laplacian 矩阵** | 使用 edge_index 计算稀疏 Laplacian | ⚠️ 代码存在但不可达 | 已实现但被禁用 |

---

### 3. 实验结果再解读

#### **原假设（错误）:**
"GR-Gaussian 技术导致了性能下降，需要调整超参数。"

#### **实际真相:**
"GR 实验"实际上就是一个配置了不同超参数的 baseline 实验，Graph Laplacian 正则化从未参与训练。

#### **性能差异的真实原因:**

**GR 实验配置:**
- `iterations=30000` (比 baseline 的 10k 多 3 倍)
- `densify_until_iter=15000` (比 baseline 的 5k 多 3 倍)
- `position_lr_init=0.0002` (baseline 未知，但可能不同)

**Baseline 配置:**
- `iterations=10000`
- `densify_until_iter=5000`

**推测:** GR 实验性能差的原因可能是：
1. **过拟合:** 30k 次迭代对于 3 视角稀疏场景可能过多，导致过拟合训练集，泛化性能下降
2. **密化时间过长:** `densify_until_iter=15000` 导致高斯点数量持续增长过久，可能引入噪声点
3. **学习率衰减:** 30k 步的学习率调度策略可能不适合 3 视角场景

---

### 4. 根因诊断总结

**问题根源:** 开发者在实现 GR-Gaussian 时未完成核心组件（`GaussianGraph` 类），出于谨慎选择硬编码禁用，但忘记在配置参数中同步禁用 `enable_graph_laplacian`，导致用户误以为功能已实现。

**关键失误:**
1. ❌ **缺少实现验证:** 未检查 `GaussianGraph` 类是否存在即引用
2. ❌ **缺少功能测试:** 没有单元测试验证 Graph Laplacian 损失是否非零
3. ❌ **缺少日志监控:** 未在 tensorboard 记录损失，导致问题被掩盖
4. ❌ **配置不一致:** 代码禁用了功能，但参数文件仍允许启用

---

## 【修复方案】

### 修复优先级

#### **阶段 1: 紧急修复（P0 - 致命缺陷）**

**目标:** 恢复 Graph Regularization 基本功能

---

#### **Step 1.1: 实现 GaussianGraph 类**

**文件:** 新建 `r2_gaussian/utils/graph_utils.py`

**代码框架:**
```python
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np

class GaussianGraph:
    """
    高斯点云图结构 - 用于 GR-Gaussian 正则化

    功能:
        - 构建 KNN 图 (k 近邻)
        - 计算边权重 (基于 RBF 核)
        - 提供边索引供 Laplacian 损失计算
    """

    def __init__(self, k=6, sigma=None):
        """
        Args:
            k: KNN 邻居数量 (默认 6, 根据论文)
            sigma: RBF 核带宽 (None 时自动估计)
        """
        self.k = k
        self.sigma = sigma
        self.edge_index = None  # (2, E) - [源节点, 目标节点]
        self.edge_weights = None  # (E,) - 边权重
        self.num_nodes = 0

    def build_knn_graph(self, xyz):
        """
        构建 KNN 图

        Args:
            xyz: (N, 3) 高斯点位置
        """
        N = xyz.shape[0]
        self.num_nodes = N

        # 转换为 numpy 用于 sklearn KNN
        xyz_np = xyz.detach().cpu().numpy()

        # 构建 KNN 图 (k+1 因为包含自身)
        nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto').fit(xyz_np)
        distances, indices = nbrs.kneighbors(xyz_np)

        # 移除自身连接 (第 0 列)
        distances = distances[:, 1:]  # (N, k)
        indices = indices[:, 1:]      # (N, k)

        # 构建边索引 (双向边)
        src = np.repeat(np.arange(N), self.k)  # [0,0,...,0, 1,1,...,1, ..., N-1,...]
        dst = indices.flatten()                 # [邻居0_0, ..., 邻居0_k, 邻居1_0, ...]

        # 转换为 PyTorch tensor
        device = xyz.device
        self.edge_index = torch.stack([
            torch.from_numpy(src).long().to(device),
            torch.from_numpy(dst).long().to(device)
        ], dim=0)  # (2, E) where E = N * k

        # 存储距离用于权重计算
        self._distances = torch.from_numpy(distances.flatten()).float().to(device)  # (E,)

    def compute_edge_weights(self, xyz):
        """
        计算边权重 - RBF 核函数

        w_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))

        Args:
            xyz: (N, 3) 高斯点位置
        """
        if self.edge_index is None:
            raise RuntimeError("Must call build_knn_graph() before compute_edge_weights()")

        # 自动估计 sigma (如果未指定)
        if self.sigma is None:
            # 使用 KNN 距离的中位数作为 sigma
            self.sigma = torch.median(self._distances).item()

        # 计算 RBF 权重
        sigma_sq = self.sigma ** 2
        self.edge_weights = torch.exp(-self._distances ** 2 / (2 * sigma_sq))  # (E,)

        # 归一化权重 (可选, 使损失稳定)
        # self.edge_weights = self.edge_weights / self.edge_weights.sum()
```

---

#### **Step 1.2: 修复 train.py - 启用图结构初始化**

**文件:** `train.py`
**修改位置:** 第 153-155 行

**修改前:**
```python
# ❌ 禁用 GR-Gaussian（不确定实现是否正确）
gr_graph = None
print("⚠️ [R²] Graph Regularization disabled (focus on FSGS)")
```

**修改后:**
```python
# 🌟 [GR-Gaussian] 图结构初始化
gr_graph = None
if dataset.enable_graph_laplacian:
    from r2_gaussian.utils.graph_utils import GaussianGraph

    gr_graph = GaussianGraph(
        k=dataset.graph_k,
        sigma=None  # 自动估计
    )

    # 初始化图结构
    with torch.no_grad():
        xyz = gaussians.get_xyz.detach()
        gr_graph.build_knn_graph(xyz)
        gr_graph.compute_edge_weights(xyz)

    print(f"✅ [GR-Gaussian] Initialized graph: {gr_graph.num_nodes} nodes, "
          f"{gr_graph.edge_index.shape[1]} edges, k={dataset.graph_k}")
else:
    print("⚠️ [R²] Graph Regularization disabled")
```

---

#### **Step 1.3: 修复 loss_utils.py - 移除提前返回**

**文件:** `r2_gaussian/utils/loss_utils.py`
**修改位置:** 第 299-301 行

**修改前:**
```python
# 🚨 [GR-Gaussian 优化] 如果没有预构建图,直接返回零损失,避免昂贵的 KNN 计算
# 在 iteration 1000 前,graph 尚未构建,此时跳过 Graph Laplacian 损失
return torch.tensor(0.0, device=xyz.device, requires_grad=True)
```

**修改后:**
```python
# 🚨 [Fallback] 如果没有预构建图且点数过少,返回零损失
if N < 100:  # 点数过少时跳过
    return torch.tensor(0.0, device=xyz.device, requires_grad=True)

# ⚠️ 警告: 动态 KNN 计算非常昂贵, 建议预构建图
print(f"⚠️ [GR-Gaussian] Warning: No pre-built graph, using expensive CPU KNN (N={N})")

# 使用 CPU fallback 进行 KNN (下面的代码已经实现但被注释)
```

**注意:** 此修改允许在没有预构建图时回退到 CPU KNN，但性能会很差。更好的做法是确保 `gr_graph` 始终在启用时被初始化（通过 Step 1.2）。

---

#### **阶段 2: 性能优化（P1 - 高优先级）**

#### **Step 2.1: 优化损失计算频率**

**文件:** `train.py`
**修改位置:** 第 666 行

**修改前:**
```python
if iteration > 5000 and iteration % 500 == 0:  # 延迟启动 + 每500次迭代计算一次
```

**修改后:**
```python
if iteration >= 0:  # 从第 0 步开始计算
```

**理由:** Graph Laplacian 正则化应该从训练开始就生效，延迟启动和低频计算会削弱正则化效果。

---

#### **Step 2.2: 添加 Tensorboard 日志**

**文件:** `train.py`
**修改位置:** 第 674 行后

**添加代码:**
```python
# 记录 Graph Laplacian 损失到 tensorboard
if iteration % 100 == 0:
    tb_writer.add_scalar(f"Loss/graph_laplacian_gs{i}", graph_laplacian_loss.item(), iteration)

# 每 1000 步打印一次
if iteration % 1000 == 0 and graph_laplacian_loss.item() > 0:
    print(f"[GR-Gaussian] Iteration {iteration}: Graph Laplacian Loss = {graph_laplacian_loss.item():.6f}")
```

---

#### **阶段 3: 健壮性增强（P2 - 中优先级）**

#### **Step 3.1: 添加功能验证检查**

**文件:** `train.py`
**修改位置:** 第 600 行后 (训练循环开始前)

**添加代码:**
```python
# 🔍 [验证] 检查 Graph Laplacian 是否正常工作
if dataset.enable_graph_laplacian and iteration == 100:
    # 测试损失计算
    test_loss = compute_graph_laplacian_loss(
        gaussians,
        graph=gr_graph,
        k=dataset.graph_k,
        Lambda_lap=dataset.graph_lambda_lap
    )

    if test_loss.item() == 0.0:
        print("❌ [GR-Gaussian] ERROR: Graph Laplacian loss is 0.0! Check implementation!")
    else:
        print(f"✅ [GR-Gaussian] Validation passed: Loss = {test_loss.item():.6f}")
```

---

#### **Step 3.2: 添加单元测试**

**文件:** 新建 `tests/test_graph_laplacian.py`

**代码框架:**
```python
import torch
import pytest
from r2_gaussian.utils.graph_utils import GaussianGraph
from r2_gaussian.utils.loss_utils import compute_graph_laplacian_loss

def test_gaussian_graph_init():
    """测试图结构初始化"""
    # 创建随机点云
    xyz = torch.randn(100, 3).cuda()

    # 构建图
    graph = GaussianGraph(k=6)
    graph.build_knn_graph(xyz)
    graph.compute_edge_weights(xyz)

    # 验证
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == 100 * 6  # N * k
    assert graph.edge_weights.shape[0] == 100 * 6
    assert torch.all(graph.edge_weights > 0)  # 权重应为正

def test_graph_laplacian_loss_nonzero():
    """测试 Graph Laplacian 损失非零"""
    from r2_gaussian.gaussian_model import GaussianModel

    # 创建虚拟高斯模型
    gaussians = GaussianModel(sh_degree=0)
    gaussians._xyz = torch.randn(100, 3).cuda()
    gaussians._density = torch.randn(100).cuda()

    # 构建图
    graph = GaussianGraph(k=6)
    graph.build_knn_graph(gaussians.get_xyz)
    graph.compute_edge_weights(gaussians.get_xyz)

    # 计算损失
    loss = compute_graph_laplacian_loss(gaussians, graph=graph, Lambda_lap=8e-4)

    # 验证
    assert loss.item() > 0, "Graph Laplacian loss should be non-zero!"
    print(f"✅ Loss = {loss.item():.6f}")

if __name__ == "__main__":
    test_gaussian_graph_init()
    test_graph_laplacian_loss_nonzero()
    print("✅ All tests passed!")
```

---

### 修复后的完整工作流

```
1. 实现 GaussianGraph 类 (graph_utils.py)
   ↓
2. 修改 train.py 启用图初始化 (第 153 行)
   ↓
3. 修改 loss_utils.py 移除提前返回 (第 299 行)
   ↓
4. 修改 train.py 优化损失计算频率 (第 666 行)
   ↓
5. 添加 Tensorboard 日志 (train.py 第 674 行)
   ↓
6. 添加验证检查 (train.py 第 600 行)
   ↓
7. 编写并运行单元测试 (tests/test_graph_laplacian.py)
   ↓
8. 重新运行实验验证修复效果
```

---

## 【修复后实验配置建议】

### 实验 1: 验证功能修复（10k 快速验证）

**目的:** 确认 Graph Laplacian 正则化已经生效

**配置:**
```bash
python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path output/2025_11_18_gr_gaussian_10k_fixed \
  --enable_graph_laplacian \
  --graph_k 6 \
  --graph_lambda_lap 0.0008 \
  --graph_update_interval 1000 \
  --iterations 10000 \
  --densify_until_iter 5000 \
  --eval
```

**验证步骤:**
1. 检查训练日志是否打印 "✅ [GR-Gaussian] Initialized graph"
2. 检查 iteration 100 是否打印 "✅ [GR-Gaussian] Validation passed"
3. 打开 tensorboard 检查 `Loss/graph_laplacian_gs0` 曲线是否非零
4. 对比 PSNR/SSIM 是否与 baseline (10k) 有差异

**预期结果:**
- Graph Laplacian 损失曲线在 [1e-5, 1e-3] 范围内波动
- PSNR 应该接近或略高于 baseline (28.5 dB)

---

### 实验 2: 超参数扫描（调优 λ_lap）

**目的:** 找到最优的正则化权重

**配置:** 固定其他参数，扫描 `graph_lambda_lap`

| 实验名称 | graph_lambda_lap | 预期效果 |
|---------|------------------|---------|
| gr_lambda_1e-4 | 0.0001 | 正则化过弱，可能无明显改进 |
| gr_lambda_4e-4 | 0.0004 | 论文推荐值的一半 |
| gr_lambda_8e-4 | 0.0008 | 论文推荐值（基线） |
| gr_lambda_1.6e-3 | 0.0016 | 论文推荐值的 2 倍 |
| gr_lambda_3.2e-3 | 0.0032 | 强正则化，可能过平滑 |

**运行命令示例:**
```bash
for lambda in 0.0001 0.0004 0.0008 0.0016 0.0032; do
  python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_18_gr_lambda_sweep/lambda_${lambda} \
    --enable_graph_laplacian \
    --graph_lambda_lap ${lambda} \
    --iterations 10000 \
    --densify_until_iter 5000 \
    --eval
done
```

**分析指标:**
- PSNR, SSIM (定量)
- 渲染图像质量 (定性)
- Graph Laplacian 损失曲线 (查看正则化强度)
- 高斯点数量 (检查是否抑制过度密化)

---

### 实验 3: 与 Baseline 对比（30k 完整训练）

**目的:** 在完整训练设置下验证 GR-Gaussian 的改进效果

**配置对比:**

| 配置项 | Baseline | GR-Gaussian (修复后) |
|-------|----------|---------------------|
| enable_graph_laplacian | False | True |
| graph_lambda_lap | N/A | 0.0008 (或从实验2选最优) |
| graph_k | N/A | 6 |
| graph_update_interval | N/A | 1000 |
| iterations | 30000 | 30000 |
| densify_until_iter | 15000 | 15000 |

**运行命令:**
```bash
# Baseline (作为参照)
python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path output/2025_11_18_baseline_30k \
  --iterations 30000 \
  --densify_until_iter 15000 \
  --eval

# GR-Gaussian (修复后)
python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path output/2025_11_18_gr_gaussian_30k_fixed \
  --enable_graph_laplacian \
  --graph_lambda_lap 0.0008 \
  --graph_k 6 \
  --graph_update_interval 1000 \
  --iterations 30000 \
  --densify_until_iter 15000 \
  --eval
```

**分析要点:**
1. **收敛速度:** 比较两者的损失曲线下降速度
2. **最终性能:** PSNR/SSIM 对比（至少运行 3 次取平均）
3. **高斯点分布:** 可视化点云，检查 GR 是否减少噪声点
4. **训练稳定性:** 查看损失曲线是否更平滑

---

### 实验 4: 消融实验（验证图更新频率）

**目的:** 验证 `graph_update_interval` 对性能的影响

**配置:**

| 实验名称 | graph_update_interval | 说明 |
|---------|----------------------|------|
| gr_update_500 | 500 | 更新频繁，计算开销大 |
| gr_update_1000 | 1000 | 论文推荐（基线） |
| gr_update_2000 | 2000 | 更新稀疏，节省计算 |
| gr_update_inf | -1 (仅初始化一次) | 不更新，仅用初始图 |

**分析:** 找到性能与计算成本的最佳平衡点

---

## 【需要您的决策】

根据以上诊断和修复方案，请选择下一步行动：

### **选项 A: 立即修复并验证（推荐）**
- 按照修复方案实现 `GaussianGraph` 类
- 修改 train.py 和 loss_utils.py
- 运行单元测试验证功能
- 执行实验 1 (10k 快速验证)
- **预计时间:** 2-3 小时（编码 + 测试 + 运行）

### **选项 B: 先运行单元测试，再决定是否修复**
- 仅实现 `GaussianGraph` 类和单元测试
- 验证功能是否可行
- 根据测试结果决定是否继续修复
- **预计时间:** 1 小时

### **选项 C: 暂缓修复，先分析其他技术**
- 将 GR-Gaussian 标记为"待修复"
- 优先分析其他论文技术（如 FSGS, SAX-NeRF）
- 等待更充足的开发时间再修复 GR
- **预计时间:** 立即转向其他任务

### **选项 D: 完全放弃 GR-Gaussian**
- 从代码库中移除所有 GR 相关代码
- 更新参数配置，禁用 `enable_graph_laplacian` 选项
- 专注于其他更成熟的技术
- **预计时间:** 30 分钟清理代码

---

**推荐选择:** **选项 A**

**理由:**
1. GR-Gaussian 论文的核心思想（Graph Laplacian 正则化）在理论上对稀疏视角场景有益
2. 修复难度不高（主要是实现 `GaussianGraph` 类，约 100 行代码）
3. 可以为后续研究提供有价值的对比实验数据
4. 修复后可以验证论文结论，为知识库增加宝贵经验

---

## 【附录: 相关文件路径清单】

### 需要修改的文件
1. `/home/qyhu/Documents/r2_ours/r2_gaussian/train.py` (第 153-155, 666 行)
2. `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/loss_utils.py` (第 299-301 行)

### 需要新建的文件
1. `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/graph_utils.py` (GaussianGraph 类)
2. `/home/qyhu/Documents/r2_ours/r2_gaussian/tests/test_graph_laplacian.py` (单元测试)

### 相关实验输出
1. `/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_gr_gaussian_30k_optimized/` (失败的 GR 实验)
2. `/home/qyhu/Documents/r2_ours/r2_gaussian/output/foot_3_1013/` (Baseline 对照)

---

**✋ 等待用户确认：请选择 A/B/C/D 中的一个选项，或提出其他想法。**
