# 项目进度记录

---

## [2025-11-18 14:30] GR-Gaussian 实验失败根因诊断

**执行者:** 深度学习调参与分析专家
**状态:** ✅ 已完成
**版本:** commit 290569d

### 任务目标
深入分析为什么 GR-Gaussian 实验效果远低于 baseline，并提供修复方案。

### 核心发现

**致命问题：Graph Regularization 功能完全未生效**

1. **train.py:154** - `gr_graph = None` 被硬编码禁用
2. **缺失 GaussianGraph 类** - 整个代码库中不存在该类实现
3. **loss_utils.py:299** - 当 graph=None 时直接返回 0 损失
4. **实验真相** - "GR 实验"本质上是配置了不同超参数的 baseline

### 根本原因
开发者在实现 GR-Gaussian 时未完成核心组件，出于谨慎选择硬编码禁用，但配置参数仍允许 `enable_graph_laplacian=True`，导致用户误以为功能已实现。

### 交付物
- **诊断报告:** `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/gr_diagnosis_report.md` (700 行)
  - 6 个代码缺陷（3 个致命 P0, 2 个高优先级 P1, 1 个中优先级 P2）
  - 完整修复方案（包含代码框架）
  - 4 组实验配置建议（验证、调优、对比、消融）

### 修复方案概要

**阶段 1: 紧急修复 (P0)**
1. 实现 `GaussianGraph` 类（新建 `graph_utils.py`）
2. 修改 `train.py` 启用图初始化（第 153 行）
3. 修改 `loss_utils.py` 移除提前返回（第 299 行）

**阶段 2: 性能优化 (P1)**
4. 优化损失计算频率（移除延迟启动）
5. 添加 Tensorboard 日志监控

**阶段 3: 健壮性增强 (P2)**
6. 添加功能验证检查
7. 编写单元测试

### 推荐实验计划

**实验 1:** 10k 快速验证（确认修复生效）
**实验 2:** 超参数扫描（调优 λ_lap ∈ [1e-4, 3.2e-3]）
**实验 3:** 30k 完整对比（vs. baseline）
**实验 4:** 消融实验（验证图更新频率）

### 决策选项
- **选项 A (推荐):** 立即修复并验证（预计 2-3 小时）
- **选项 B:** 先运行单元测试（预计 1 小时）
- **选项 C:** 暂缓修复，先分析其他技术
- **选项 D:** 完全放弃 GR-Gaussian

**✋ 等待用户确认下一步行动**

---

## [2025-11-18 15:45] GR-Gaussian 代码实现与调试（进行中）

**执行者:** PyTorch/CUDA 编程专家
**状态:** ⚠️ 进行中（遇阻）
**版本:** 基于 commit 290569d

### 任务目标
根据诊断报告实现 GR-Gaussian 核心功能，修复 PSNR/SSIM 远低于 baseline 的问题（17.26 vs 28.31）。

### 完成的工作

#### 1️⃣ 核心代码实现（2025-11-18 15:15）

**新建文件:**
- **`/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/graph_utils.py`** (180行)
  - 实现 `GaussianGraph` 类
  - 核心功能：K-NN 图构建、拉普拉斯矩阵计算、稀疏化
  - 优化：支持批处理、内存高效、GPU 加速

**修改文件:**
- **`train.py`**
  - Line 153-160: 启用图初始化逻辑（移除 `gr_graph = None` 硬编码）
  - Line 670-689: 添加 Tensorboard 日志（图节点数、边数、平均度数）
  - Line 674: 只对 gs0 应用图正则化（修复多 GS 集合索引越界）

- **`r2_gaussian/utils/loss_utils.py`**
  - Line 281-310: 移除提前返回，添加完整损失计算
  - Line 284-287: 节点数验证（防止 shape 不匹配）
  - Line 292-294: 边界检查（防止索引越界）

#### 2️⃣ Bug 修复迭代（4 轮）

| Bug | 现象 | 根因 | 修复方案 |
|-----|------|------|----------|
| #1 | edge_weights 形状不匹配 | 图重建后未更新权重 | 移除权重缓存，始终重新计算 |
| #2 | RuntimeError: num_nodes=61769 ≠ means.size(0)=61793 | 图构建时使用错误的 gaussians 变量 | 添加节点数验证（loss_utils.py:284-287）|
| #3 | IndexError: index 61793 超出边索引范围 [0, 61768] | 多 GS 集合共享图，gs1 索引超界 | 只对 gs0 应用正则化（train.py:674）|
| #4 | 边索引越界（重复出现）| 异步更新导致图与 Gaussians 不一致 | 添加边界检查（loss_utils.py:292-294）|

#### 3️⃣ 验证测试

**测试日志分析（iteration 1-999）:**
```
Graph Info:
- Nodes: 61769
- Edges: 370614
- Avg degree: 6.00
- Loss computation: PASSED
```

**✅ 成功部分:**
- 图构建逻辑正常工作（K-NN=6）
- 拉普拉斯损失计算通过验证
- Tensorboard 日志正常记录

### 当前阻塞问题（CRITICAL）

**现象:**
- 训练持续在 **iteration 1000** 崩溃
- 错误：`CUDA out of memory. Tried to allocate 511.69 GiB`

**日志证据:**
```
[ITER 999] Graph Info: nodes=61769, edges=370614
[ITER 1000] RuntimeError: CUDA out of memory
```

**可疑线索:**
1. Iteration 1000 是 densification 发生点（`densify_and_prune`）
2. 图在 densification 后未同步更新
3. 损失计算时使用过期的图结构
4. 索引不匹配触发大规模内存分配

**已尝试的修复（均失败）:**
- ✗ 添加节点数验证（仍崩溃）
- ✗ 只对 gs0 应用正则化（仍崩溃）
- ✗ 添加边界检查（仍崩溃）
- ✗ 移除权重缓存（仍崩溃）

### 深层技术分析

**怀疑根因：**
1. **图更新时机问题**
   - 当前设计：每 `opt.graph_update_interval` 次迭代重建图
   - Densification：每 500 次迭代动态增删 Gaussians
   - **冲突点：** iteration 1000 图重建与 densification 同时发生

2. **变量作用域混淆**
   - `loss_utils.py` 中的 `gaussians` 参数可能不是最新状态
   - 图构建时使用 `gaussians.get_xyz` 但损失计算时使用 `means`
   - **潜在不一致：** xyz 和 means 可能来自不同的 GS 集合

3. **内存分配异常**
   - 511.69 GiB 是 512 GB，明显是索引计算错误（如 `num_edges * max_index`）
   - 可能触发因素：edge_index 中出现异常大的索引值

### 关键代码位置

**需要排查的执行路径:**
```python
train.py:668 - densify_and_prune()  # 修改 Gaussians
train.py:674 - gr_graph.rebuild()   # 重建图
train.py:254 - loss_utils.graph_laplacian_loss()  # 计算损失
```

**核心疑问:**
- `graph_laplacian_loss()` 中传入的 `gaussians` 是否是 densification 后的最新状态？
- `means` 参数是从哪个 GS 集合提取的？
- 边索引的最大值是否与当前节点数一致？

### 下一步行动计划

**紧急任务（优先级 P0）:**
1. **在 iteration 1000 前后添加详细日志**
   - 打印 `gaussians` 对象 ID（验证是否同一对象）
   - 打印 `means.size(0)` vs `edge_index.max()`
   - 打印图重建前后的内存占用

2. **临时绕过方案**
   - 选项 1：禁用 iteration 1000 的图重建（`if iteration % 500 == 0 and iteration % 1000 != 0`）
   - 选项 2：在 densification 后强制重建图（无论 interval）

3. **长期修复**
   - 重构图更新逻辑，确保与 densification 同步
   - 添加图有效性检查（`assert edge_index.max() < num_nodes`）

### 知识沉淀

**失败教训:**
- ⚠️ **动态拓扑与静态图不兼容** - Gaussian Splatting 的动态增删与固定图结构存在根本性冲突
- ⚠️ **异步更新需要显式同步机制** - 图的重建时机必须与 Gaussians 修改严格绑定
- ⚠️ **内存错误可能是索引错误的表象** - 511 GB 分配请求实际是索引越界的副作用

**成功经验:**
- ✅ **稀疏矩阵是必需的** - 密集拉普拉斯矩阵无法处理 6 万节点规模
- ✅ **K-NN=6 是合理选择** - 与诊断报告一致，计算开销可控
- ✅ **节点数验证有效** - 虽未完全解决问题，但暴露了数据不一致

### 交付物清单

- ✅ `r2_gaussian/utils/graph_utils.py` (180 行)
- ✅ 修改 `train.py` (3 处，共 40 行)
- ✅ 修改 `loss_utils.py` (1 处，30 行)
- ⚠️ 实验结果：**失败**（训练崩溃）

### 决策选项

**选项 A (快速验证):**
临时禁用 iteration 1000 的图重建，先跑通 30k 实验，观察指标是否改善

**选项 B (深度调试):**
添加详细日志定位内存爆炸的确切原因，彻底修复后再实验

**选项 C (架构重构):**
重新设计图更新策略（如每次 densification 后立即重建图），确保同步一致性

**选项 D (功能降级):**
暂时只在初始化时构建图，训练过程中冻结图结构（避免动态更新）

**✋ 等待用户确认下一步行动**

---
