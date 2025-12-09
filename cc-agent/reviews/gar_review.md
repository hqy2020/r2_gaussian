# GAR (Geometry-Aware Refinement) 代码审查报告

> 审查日期: 2025-12-09
> 审查人: Claude Code

## 1. 概述

GAR (几何感知细化) 基于 FSGS 的 Proximity-guided Densification，通过 K-NN 计算邻近分数识别稀疏区域，在边界中点生成新高斯，是 SPAGS 的密化模块。

## 2. 相关文件

| 文件 | 用途 |
|------|------|
| `r2_gaussian/innovations/fsgs/proximity_densifier.py` | **核心实现** - ProximityGuidedDensifier 类 |
| `r2_gaussian/innovations/fsgs/__init__.py` | 模块导出 |
| `r2_gaussian/innovations/fsgs/utils.py` | 工具函数 |
| `r2_gaussian/arguments/__init__.py` | 参数定义 (ModelParams) |
| `train.py` | GAR 调用逻辑 (第 96-410 行) |
| `cc-agent/scripts/run_spags_ablation.sh` | 训练配置 (第 86-91 行) |

## 3. 超参数列表

### 3.1 ModelParams 中的参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_gar` | bool | `False` | **主开关**: 启用 GAR |
| `gar_proximity_threshold` | float | `0.05` | 邻近分数阈值 (场景归一化到 [-1,1]³) |
| `gar_proximity_k` | int | `5` | K-邻近的邻居数量 |
| `proximity_start_iter` | int | `1000` | 密化开始迭代 |
| `proximity_interval` | int | `500` | 密化间隔 |
| `proximity_until_iter` | int | `15000` | 密化结束迭代 |

### 3.2 优化参数 (v2 新增)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gar_adaptive_threshold` | bool | `False` | 启用自适应阈值 |
| `gar_adaptive_method` | str | `"percentile"` | 自适应方法 |
| `gar_adaptive_percentile` | float | `90.0` | 百分位数 (90=只密化最稀疏的10%) |
| `gar_progressive_decay` | bool | `False` | 启用渐进衰减 |
| `gar_decay_start_ratio` | float | `0.5` | 衰减开始进度 |
| `gar_final_strength` | float | `0.3` | 最终强度 |
| `gar_gradient_filter` | bool | `False` | 启用梯度过滤 |
| `gar_gradient_threshold` | float | `0.0002` | 梯度阈值 |

### 3.3 向下兼容参数

| 新参数 | 旧参数 |
|--------|--------|
| `enable_gar` | `enable_gar_proximity` / `enable_fsgs_proximity` |
| `gar_proximity_threshold` | `proximity_threshold` |
| `gar_proximity_k` | `proximity_k_neighbors` |

## 4. 核心算法

### 4.1 邻近分数计算 (FSGS Eq. 4)

```
P_i = (1/K) × Σ(j∈N_K(i)) ||μ_i - μ_j||₂
```

高斯点 i 到其 K 个最近邻的平均距离。

### 4.2 候选点识别

```python
# 基础阈值
densify_mask = proximity_scores > threshold

# 自适应阈值 (percentile 方法)
threshold = torch.quantile(proximity_scores, percentile / 100)

# 渐进衰减
decay_mult = 1.0 + progress * (final_mult - 1.0)
effective_threshold = threshold * decay_mult
```

### 4.3 新高斯生成

在源点与其最近邻的中点生成新高斯:
```
new_xyz = (source_xyz + neighbor_xyz) / 2
new_density = (source_density + neighbor_density) / 2
new_scale = source_scale * 0.8  # 略微缩小
```

## 5. 执行流程

```
iteration ∈ [start_iter, until_iter] && iteration % interval == 0
    ↓
compute_proximity_scores(positions) → scores [N]
    ↓
compute_adaptive_threshold_value(scores, percentile) → threshold
    ↓
apply_progressive_decay(threshold, iteration) → effective_threshold
    ↓
identify_densify_candidates(scores, effective_threshold) → mask [N]
    ↓
[可选] gradient_filter(mask, grad_threshold) → filtered_mask
    ↓
[可选] limit_candidates(filtered_mask, max=5000) → limited_mask
    ↓
generate_new_gaussians(source_positions, neighbor_indices) → new_gaussians
    ↓
gaussians.densification_postfix(new_gaussians)
```

## 6. 训练脚本配置

`run_spags_ablation.sh:86-91`:
```bash
GAR_FLAGS_COMPAT="--enable_fsgs_proximity \
    --gar_adaptive_threshold \
    --gar_adaptive_percentile 85 \
    --gar_progressive_decay \
    --gar_decay_start_ratio 0.7 \
    --gar_final_strength 0.5"
```

**当前实验配置**:
- 自适应阈值: percentile=85 (密化最稀疏的 15%)
- 渐进衰减: start=70%, final=0.5 (阈值最终提高 ~2 倍)

## 7. 发现的问题

### 7.1 K-NN 计算效率问题 (高风险) - ✅ 已修复

**位置**: `proximity_densifier.py:195-265` (`_compute_knn_faiss`)

**原问题**: 使用 `torch.cdist` 计算距离矩阵，复杂度 O(N²)。

**修复方案** (2025-12-09):
- 添加 FAISS 支持，复杂度降至 O(N log N)
- 小规模 (<50K) 使用精确搜索 `IndexFlatL2`
- 大规模 (≥50K) 使用近似搜索 `IndexIVFFlat`
- FAISS 不可用时自动回退到 PyTorch chunked 实现

```python
# 新增参数
use_faiss: bool = True  # 是否使用 FAISS 加速

# 安装 FAISS
pip install faiss-gpu  # GPU 版本
# 或
pip install faiss-cpu  # CPU 版本
```

### 7.2 自适应阈值边界保护过于保守 (中等风险)

**位置**: `train.py` 阈值计算逻辑

```python
p50 = torch.quantile(proximity_scores, 0.5)
threshold = threshold.clamp(min=p50)
```

**问题**: percentile=90 计算的阈值被中位数 P50 限制，实际密化点数可能比预期多。

**示例**:
- 预期: 密化最稀疏的 10% (P90 以上)
- 实际: 如果 P90 < P50 (这不可能)，但限制逻辑意味着永远不会低于 P50

**注**: 这个限制逻辑实际上是防止阈值过低的保护，但命名和注释可能造成混淆。

### 7.3 梯度过滤 NaN 风险 (低风险)

**位置**: `train.py:342-346`

```python
grads = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-7)
gradient_mask = grads.squeeze() > gar_gradient_threshold
```

**问题**: 早期迭代 `xyz_gradient_accum` 可能包含未初始化值。

**现状**: 代码有 NaN 处理:
```python
gradient_mask[torch.isnan(grads.squeeze())] = False
```

**建议**: 增加梯度预热期或更鲁棒的归一化

### 7.4 内存管理中的 Mask 复用 (低风险)

**位置**: `train.py:362`

```python
densify_mask.fill_(False)
densify_mask[selected_indices] = True
```

**问题**: 原地修改 mask 张量，如果 `selected_indices` 有重复可能导致问题。

**建议**: 使用新张量而非原地修改

### 7.5 新高斯旋转初始化 (低风险)

**位置**: `proximity_densifier.py:556-557`

```python
new_rotations = torch.zeros(new_shape, device=device)
new_rotations[:, 0] = 1.0  # w=1, x=y=z=0
```

**问题**: 使用 w-first 四元数格式，需确保与其他代码一致。

**现状**: 项目整体使用 w-first 格式，此处正确。

### 7.6 候选点数量限制硬编码 (低风险) - ✅ 已修复

**位置**: `train.py:356-367`, `arguments/__init__.py:103`

**原问题**: 5000 硬编码，大场景可能需要更多密化。

**修复方案** (2025-12-09):
- 新增参数 `gar_max_candidates` 控制每次密化最大候选点数
- 默认值仍为 5000，但可通过命令行调整
- 修复了原地修改 mask 的潜在问题，改用新张量

```python
# 新增参数
self.gar_max_candidates = 5000  # [GAR] 每次密化最大候选点数（避免 OOM）

# 使用示例
--gar_max_candidates 10000  # 大场景可增加
```

## 8. 渐进衰减曲线

```
multiplier
2.0  |            ___________
     |           /
1.0  |__________/
     |
     0%   70%   100%  (progress)
         ↑
    decay_start_ratio
```

| 进度 | multiplier | 有效阈值 (base=0.05) |
|------|------------|---------------------|
| 0-70% | 1.0 | 0.05 |
| 85% | 1.5 | 0.075 |
| 100% | 2.0 | 0.10 |

## 9. 诊断输出

`train.py` 每 1000 次迭代输出:
```
[GAR Iter 5000] 密化诊断:
  邻近分数: mean=0.0234, std=0.0156, range=[0.002, 0.312]
  阈值: base=0.050, effective=0.050 (decay_mult=1.00)
  候选点: 2341 / 50000 (4.68%)
  新增高斯: 2341
  总高斯数: 52341
```

## 10. 代码质量评估

### 优点
- 模块化设计 (ProximityGuidedDensifier 独立类)
- 完善的参数化 (v2 优化参数)
- 错误处理和 OOM 降级
- 详细的诊断输出

### 改进点
- K-NN 效率需要优化
- 参数名过多 (3 套兼容名)
- 缺少单元测试

## 11. 当前配置状态

**训练脚本中 GAR 配置**:
```bash
--enable_fsgs_proximity         # 启用 (旧参数名)
--gar_adaptive_threshold        # 自适应阈值
--gar_adaptive_percentile 85    # 密化最稀疏 15%
--gar_progressive_decay         # 渐进衰减
--gar_decay_start_ratio 0.7     # 70% 进度后开始衰减
--gar_final_strength 0.5        # 阈值最终提高 ~2x
```

**状态**: 功能正常，v2 优化已解决 "iter 9000 后停止密化" 问题。

## 12. 性能改进建议

### 12.1 短期 (参数调优)
- 监控实际密化点数，确保符合预期
- 考虑按器官调整 percentile

### 12.2 中期 (代码优化)
- 替换 torch.cdist 为 FAISS
- 参数名统一化

### 12.3 长期 (架构改进)
- 视角自适应 GAR (类似 ADM)
- 基于梯度的自适应密化频率
