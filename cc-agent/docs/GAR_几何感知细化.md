# GAR 几何感知细化技术文档

> **GAR**: **G**eometry-**A**ware **R**efinement
>
> 使用邻近分数识别稀疏区域并智能密化

---

## 1. Motivation：为什么需要几何感知密化？

### 1.1 问题：梯度驱动密化在稀疏视角下失效

标准 3DGS 使用**梯度驱动密化**：累积梯度超过阈值的点会被密化。

但在稀疏视角（3/6/9 views）CT 重建中，这种方法存在严重问题：

```
梯度驱动密化的问题
├── 梯度信号弱
│   ├── 只有 3-9 个视角提供监督
│   ├── 大部分区域梯度接近零
│   └── 难以识别真正需要密化的位置
│
├── 梯度信号不可靠
│   ├── 稀疏视角下梯度方向可能有偏
│   ├── ��易在错误位置密化
│   └── 产生伪影
│
└── 无法感知几何稀疏性
    ├── 梯度只反映"优化需求"
    ├── 不反映"空间分布"
    └── 孤立的点可能梯度很小
```

### 1.2 核心洞察：直接感知���何稀疏性

**关键观察**：如果一个高斯点距离它的邻居很远，那它所在的区域就是**稀疏**的。

这个观察不依赖梯度，而是直接从**几何结构**出发。

### 1.3 GAR 的解决方案

**核心思想**：用**邻近分数**（proximity score）量化每个点的孤立程度

$$
P_i = \frac{1}{K} \sum_{j \in \text{KNN}(i)} \|μ_i - μ_j\|_2
$$

- $P_i$ 越大 → 该点越孤立 → 越需要密化
- $P_i$ 越小 → 该点周围很密集 → 不需要密化

---

## 2. 核心公式

### 2.1 邻近分数计算

```
邻近分数 = 该点到 K 个最近邻的平均距离
```

代码实现（`proximity_densifier.py`）：

```python
def compute_proximity_scores(
    self,
    positions: torch.Tensor,  # (N, 3)
    custom_k: Optional[int] = None,
) -> torch.Tensor:
    """
    计算每个高斯的邻近分数
    """
    K = custom_k if custom_k is not None else self.k_neighbors

    # 计算 K-NN（使用 FAISS 或 PyTorch）
    neighbor_distances, neighbor_indices = self._compute_knn(positions, K)

    # 邻近分数 = K 个近邻的平均距离
    proximity_scores = neighbor_distances.mean(dim=1)  # (N,)

    return proximity_scores
```

### 2.2 密化判断

```python
# 邻近分数 > 阈值 → 需要密化
densify_mask = proximity_scores > proximity_threshold
```

### 2.3 新高斯生成

在源点和其最近邻之间生成新点：

```python
# 新位置 = (源点位置 + 邻点位置) / 2
new_positions = (source_positions + neighbor_positions) / 2.0
```

---

## 3. 三种优化策略

GAR 提供三种可选的优化策略，用于提升密化效果。

### 3.1 自适应阈值（Adaptive Threshold）

**问题**：固定阈值无法适应不同的数据分布

**解决方案**：根据邻近分数的分布自动确定阈值

```python
# 百分位数方法（推荐）
threshold = torch.quantile(proximity_scores, percentile / 100.0)

# 例：percentile=85 → 只密化最稀疏的 15% 点
```

**可用方法**：
| 方法 | 计算方式 | 适用场景 |
|------|---------|---------|
| `percentile` | 第 N 百分位 | 通用，推荐 |
| `std` | mean + 1.5×std | 正态分布数据 |
| `iqr` | Q3 + 1.5×IQR | 有离群点的数据 |

### 3.2 渐进衰减（Progressive Decay）

**问题**：训练后期继续密化可能破坏已收敛的结构

**解决方案**：训练后期逐渐减少密化强度

```
时间线：
[iter 1000] ──────────────────[iter 10500]──────────[iter 15000]
     │                              │                     │
     │←───── 正常密化 ─────→│←── 逐渐衰减 ──→│
     │     (mult = 1.0)            │     (1.0 → 0.5)      │
```

代码实现：

```python
def get_decay_multiplier(self, iteration, start_iter, until_iter):
    progress = (iteration - start_iter) / (until_iter - start_iter)

    if progress < decay_start_ratio:  # 例如 0.7
        return 1.0  # 正常密化
    else:
        # 线性衰减：阈值逐渐提高
        decay_progress = (progress - decay_start_ratio) / (1.0 - decay_start_ratio)
        final_multiplier = 1.0 / final_strength  # 例如 1/0.5 = 2
        return 1.0 + decay_progress * (final_multiplier - 1.0)
```

### 3.3 梯度过滤（Gradient Filter）

**目的**：结合梯度信息，避免在完全不需要的位置密化

```python
# 同时满足两个条件才密化：
# 1. 邻近分数高（几何稀疏）
# 2. 梯度高（优化有需求）

if gar_gradient_filter:
    gradients = xyz_gradient_accum / (denom + 1e-7)
    gradient_mask = gradients > gradient_threshold
    densify_mask = densify_mask & gradient_mask
```

---

## 4. 训练中的调用流程

GAR 在训练循环中定期执行（每 `proximity_interval` 次迭代）：

```python
# train.py 中的 GAR 流程（简化）

if iteration >= proximity_start_iter and \
   iteration <= proximity_until_iter and \
   iteration % proximity_interval == 0:

    # 1. 计算邻近分数
    positions = gaussians.get_xyz
    proximity_scores, neighbor_indices, _ = \
        proximity_densifier.compute_proximity_scores(positions, return_neighbors=True)

    # 2. 计算有效阈值（可选优化策略）
    if gar_adaptive_threshold:
        threshold = proximity_densifier.compute_adaptive_threshold_value(proximity_scores)
    if gar_progressive_decay:
        decay_mult = proximity_densifier.get_decay_multiplier(iteration, ...)
        threshold = threshold * decay_mult

    # 3. 识别需要密化的点
    densify_mask = proximity_scores > threshold

    # 4. 可选：梯度过滤
    if gar_gradient_filter:
        densify_mask = densify_mask & (gradients > gradient_threshold)

    # 5. 限制候选点数
    if densify_mask.sum() > gar_max_candidates:
        # 随机选择 gar_max_candidates 个点
        ...

    # 6. 生成新高斯
    new_gaussians = proximity_densifier.generate_new_gaussians(
        source_positions, source_neighbor_indices, positions, all_attributes
    )

    # 7. 添加到模型
    gaussians.densification_postfix(new_gaussians)
```

---

## 5. 超参数设置

### 5.1 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_gar` / `enable_fsgs_proximity` | False | 启用 GAR |
| `gar_proximity_threshold` | 0.05 | 邻近分数阈值 |
| `gar_proximity_k` | 5 | K 近邻数量 |
| `proximity_start_iter` | 1000 | 密化开始迭代 |
| `proximity_interval` | 500 | 密化执行间隔 |
| `proximity_until_iter` | 15000 | 密化结束迭代 |

### 5.2 优化策略参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gar_adaptive_threshold` | False | 启用自适应阈值 |
| `gar_adaptive_method` | "percentile" | 自适应方法 |
| `gar_adaptive_percentile` | 85.0 | 百分位数（只密化最稀疏的 15%） |
| `gar_progressive_decay` | False | 启用渐进衰减 |
| `gar_decay_start_ratio` | 0.7 | 衰减开始进度 |
| `gar_final_strength` | 0.5 | 最终密化强度 |
| `gar_gradient_filter` | False | 启用梯度过滤 |
| `gar_gradient_threshold` | 0.0002 | 梯度过滤阈值 |
| `gar_max_candidates` | 5000 | 每次密化最大点数 |

### 5.3 推荐配置

```bash
# SPAGS 推荐的 GAR 配置
--enable_fsgs_proximity \
--gar_adaptive_threshold \
--gar_adaptive_percentile 85 \
--gar_progressive_decay \
--gar_decay_start_ratio 0.7 \
--gar_final_strength 0.5
```

---

## 6. 使用方法

### 6.1 在训练中启用 GAR

```bash
# 方法 1：手动指定参数
python train.py \
    -s data/369/foot_50_3views.pickle \
    --enable_fsgs_proximity \
    --gar_proximity_threshold 0.05 \
    --gar_proximity_k 5 \
    --proximity_start_iter 1000 \
    --proximity_interval 500

# 方法 2：使用消融脚本（推荐）
./cc-agent/scripts/run_spags_ablation.sh gar foot 3 0
```

### 6.2 与 SPS 组合使用

```bash
# SPS + GAR
./cc-agent/scripts/run_spags_ablation.sh sps_gar foot 3 0
```

### 6.3 监控 GAR 效果

GAR 会在 TensorBoard 中记录诊断信息：

| 指标 | 说明 |
|------|------|
| `gar/score_mean` | 邻近分数均值 |
| `gar/score_std` | 邻近分数标准差 |
| `gar/threshold` | 当前有效阈值 |
| `gar/decay_mult` | 衰减系数 |
| `gar/candidates` | 本次密化的候选点数 |

---

## 7. 代码位置索引

| 功能 | 文件 | 位置 |
|------|------|------|
| 参数定义 | `r2_gaussian/arguments/__init__.py` | ModelParams 中的 GAR 参数 |
| ProximityGuidedDensifier 类 | `r2_gaussian/innovations/fsgs/proximity_densifier.py` | 完整实现 |
| 邻近分数计算 | 同上 | `compute_proximity_scores()` |
| 自适应阈值 | 同上 | `compute_adaptive_threshold_value()` |
| 渐进衰减 | 同上 | `get_decay_multiplier()` |
| 新高斯生成 | 同上 | `generate_new_gaussians()` |
| 训练循环集成 | `train.py` | 第 313-414 行 |

---

## 8. 注意事项

### 8.1 场景归一化

GAR 的阈值（如 0.05）是针对归一化到 $[-1, 1]^3$ 的场景设计的。

如果场景尺度不同，需要相应调整 `gar_proximity_threshold`。

### 8.2 内存考虑

K-NN 计算需要额外内存。对于大规模点云：
- 使用 FAISS 可加速计算
- 设置 `gar_max_candidates` 限制每次密化的点数

### 8.3 与标准密化的关系

GAR 与标准梯度驱动密化**并行执行**：
- 标准密化：每 100 次迭代，基于梯度
- GAR 密化：每 500 次迭代，基于邻近分数

两者互补，不冲突。

---

*文档版本：v1.0 | 更新日期：2025-12-10*
