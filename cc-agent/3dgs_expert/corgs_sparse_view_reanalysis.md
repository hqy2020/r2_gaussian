# CoR-GS 稀疏视角技术深度分析

## 核心发现

**关键问题：** 我们的 CoR-GS 实现在 Foot 3-views 场景下 PSNR 28.481 dB（Stage 1+3 组合）或 28.148 dB（仅 Stage 1），均低于 baseline 28.547 dB，没有获得论文承诺的性能提升。

**核心矛盾：** CoR-GS 原论文在 LLFF 3-views 上报告 PSNR 20.45 dB（vs vanilla 3DGS 19.22 dB，+1.23 dB），但我们的实现反而下降 -0.066 dB 至 -0.40 dB。

**根本原因推测：**
1. **训练策略缺失**：论文可能有关键的训练流程细节未在文中明确说明
2. **超参数设置错误**：我们使用的权重可能不适合 3-views 极稀疏场景
3. **双模型初始化问题**：论文未明确说明两个模型是否使用相同的随机种子
4. **Co-regularization 时机问题**：何时启用 disagreement loss 和 pseudo-view loss

---

## 1. 论文核心发现（稀疏场景特定）

### 1.1 论文报告的性能提升

| 数据集 | 视角数 | Vanilla 3DGS | CoR-GS | 提升 |
|--------|--------|--------------|--------|------|
| LLFF | 3-view | 19.22 dB | 20.45 dB | +1.23 dB |
| LLFF | 6-view | 24.29 dB | 24.49 dB | +0.20 dB |
| DTU | 3-view | 18.30 dB | 19.21 dB | +0.91 dB |
| DTU | 6-view | 23.50 dB | 24.51 dB | +1.01 dB |

**观察：**
- 3-views 下提升显著（+0.91~1.23 dB）
- 论文明确验证了稀疏场景适用性

### 1.2 各 Stage 单独贡献（LLFF 3-views）

| Stage | PSNR (dB) | SSIM | LPIPS | 单独提升 |
|-------|-----------|------|-------|---------|
| Baseline | 19.22 | 0.668 | 0.223 | - |
| Stage 1 (Disagreement) | 19.98 | 0.695 | 0.208 | +0.76 dB |
| Stage 2 (Co-pruning) | 19.62 | 0.673 | 0.217 | +0.40 dB |
| Stage 3 (Pseudo-view) | 20.26 | 0.706 | 0.198 | +1.04 dB |
| **Stage 1+3** | **20.45** | **0.712** | **0.196** | **+1.23 dB** |

**关键洞察：**
- Stage 3 单独贡献最大（+1.04 dB）
- Stage 1+3 组合有协同效应（+1.23 dB > 0.76 + 1.04 的非线性叠加）
- **我们的实现缺失了这个协同效应**

---

## 2. Stage 1 实现关键点（可能被我们遗漏的）

### 2.1 双模型训练策略

**论文描述：** "We train two 3D Gaussian radiance fields Θ¹ and Θ² simultaneously"

**关键未明确细节：**
1. **初始化是否相同？**
   - 论文未说明是否使用相同的随机种子
   - **我们的实现**：使用相同的 SfM 点云初始化，但可能在 densification 的随机采样上有差异
   - **推测问题**：如果两个模型初始化完全相同，disagreement signal 可能太弱

2. **训练循环顺序**
   - 论文未给出明确的训练伪代码
   - **问题**：是同时反向传播两个模型？还是交替更新？
   - **我们的实现**：同时计算两个模型的损失并反向传播
   - **可能缺失**：论文可能有特殊的梯度隔离机制

### 2.2 Disagreement Loss 计算细节

**论文公式（论文 Section 3.1）：**
```
L_total = L_color + λ_dis * R_disagreement
```

**关键未明确点：**
1. **λ_dis 的具体值**
   - 论文主文未给出 λ_dis 的具体数值
   - **我们的实现**：使用 λ_dis = 0.01
   - **问题**：这个值可能过小或过大

2. **R_disagreement 的具体定义**
   - 论文只说了用 "fitness" 和 "RMSE"
   - **未明确**：这两个指标如何组合成单一损失值？
   - **我们的实现**：仅用于监控，未实际作为损失函数的一部分
   - **严重问题**：我们可能根本没有将 disagreement 作为优化目标！

### 2.3 Co-pruning 触发时机

**论文描述：** "Execute co-pruning every 5th optimization/densification alternation"

**我们的实现问题：**
- **我们做了**：计算 disagreement metrics（fitness, RMSE）
- **我们可能缺失**：
  1. 实际的 pruning 操作（删除不匹配的 Gaussians）
  2. 每 5 个 densification 循环后的 pruning
  3. Pruning 后的参数重新初始化

**关键差异：** 论文的 Co-pruning 是**主动删除不一致的 Gaussians**，而我们只是**被动监控指标**！

---

## 3. Stage 3 实现关键点（可能被我们遗漏的）

### 3.1 Pseudo-view 生成策略

**论文描述（公式 4）：**
```
P' = (t + ε, q)
其中：
- t ∈ P (从训练视角选择一个位置)
- ε ~ N(0, σ²) (位置扰动)
- q 为两个最近训练相机的平均四元数
```

**关键未明确参数：**
1. **ε 的标准差 σ**
   - 论文主文和 supplementary 都未给出具体值
   - **我们的推测**：σ = 0.02（基于归一化场景范围）
   - **可能问题**：这个值可能不适合 CT 场景

2. **"最近"的定义**
   - 是欧式距离？还是视角距离？
   - 3-views 场景下，相邻视角间隔 120°，"最近"可能有歧义

3. **Pseudo-view 生成频率**
   - 论文只说 "online pseudo view"
   - **未明确**：每个 iteration 生成多少个？1 个还是多个？
   - **我们的实现**：每 iteration 生成 1 个
   - **可能问题**：论文可能生成更多虚拟视角

### 3.2 Co-regularization 损失加权

**论文公式（公式 7）：**
```
L = L_color + λ_p * R_pcolor
其中 λ_p = 1.0
```

**关键问题：**
1. **λ_p = 1.0 是针对哪个数据集？**
   - 论文未区分不同数据集的权重设置
   - 3-views 极稀疏场景可能需要不同的权重

2. **Warmup 策略**
   - 论文未提及是否有 warmup
   - **可能缺失**：前期应该用较小的 λ_p（如 0.1），逐步增加到 1.0
   - **我们的实现**：直接使用 1.0（可能过于激进）

### 3.3 不确定性估计的具体实现

**论文描述：** "Consider pixels that exhibit high rendering disagreement as inaccurate and suppress the disagreement"

**关键未明确点：**
1. **"Suppress" 的具体机制**
   - 是降低这些像素的损失权重？
   - 还是完全忽略这些像素？
   - **我们的实现**：基于不确定性降低权重
   - **可能问题**：降低策略可能不正确

2. **不确定性阈值**
   - 多高的 disagreement 算"高"？
   - 论文未给出具体阈值

---

## 4. 稀疏场景特殊设计

### 4.1 3-views 的特殊处理

**论文未明确提到的可能策略：**

1. **更激进的 Pseudo-view 采样**
   - 3-views 场景下，可用的真实视角很少
   - **推测**：论文可能生成更多虚拟视角（如 4-8 个/iteration）
   - **我们的实现**：只生成 1 个

2. **更大的 λ_p 权重**
   - 真实视角少，需要更依赖虚拟视角正则化
   - **推测**：3-views 可能需要 λ_p = 2.0 甚至更高
   - **我们的实现**：使用默认值 1.0

3. **延迟启用 Stage 3**
   - **可能策略**：前 30% iterations 只训练 Stage 1，后 70% 加入 Stage 3
   - **我们的实现**：从第 1000 iteration 开始启用（可能过早）

### 4.2 Loss Weight 推荐值（基于论文消融）

| 参数 | 论文默认值 | 3-views 推测值 | 我们的实现 | 差异 |
|------|-----------|--------------|-----------|------|
| λ_dis (Stage 1) | **未明确** | 0.1~0.5 | 0.01 | 可能过小 ❌ |
| λ_p (Stage 3) | 1.0 | 1.5~2.0 | 1.0 | 可能不足 ⚠️ |
| σ (pseudo noise) | **未明确** | 0.01~0.05 | 0.02 | 未知 ❓ |
| Pruning 频率 | 每 5 个 densif | 每 5 个 | **未实现** | 缺失 ❌ |

---

## 5. 我们可能的实现问题（推测）

### 问题 1：Disagreement Loss 未实际参与优化

**推测原因：**
- 我们的代码中计算了 fitness 和 RMSE，但**可能没有将它们作为损失函数的一部分反向传播**
- 只是用于 logging 和监控

**证据：**
- `corgs_metrics.py` 只计算指标，未返回可微分的损失
- `train.py` 中未找到 `loss += lambda_dis * disagreement_loss` 的代码

**修复建议：**
```python
# 需要在 train.py 中添加
if enable_disagreement_loss:
    disagreement_loss = compute_disagreement_loss(gaussians_coarse, gaussians_fine)
    LossDict['loss_gs0'] += args.lambda_dis * disagreement_loss
    LossDict['loss_gs1'] += args.lambda_dis * disagreement_loss
```

### 问题 2：Co-pruning 未实际执行

**推测原因：**
- 我们计算了哪些 Gaussians 是不匹配的，但**没有删除它们**
- 论文的核心是 "prunes them"（修剪掉）

**证据：**
- `corgs_metrics.py` 中没有实际删除 Gaussians 的代码
- 只是返回 fitness 和 RMSE 数值

**修复建议：**
```python
# 需要在 densify_and_prune() 中添加
if iteration % (5 * densification_interval) == 0:
    # 计算 disagreement
    unmatched_mask = compute_unmatched_gaussians(gaussians_coarse, gaussians_fine, threshold=5.0)
    # 删除不匹配的 Gaussians
    gaussians_coarse.prune_points(unmatched_mask)
    gaussians_fine.prune_points(unmatched_mask)
```

### 问题 3：双模型初始化可能相同

**推测原因：**
- 如果两个模型的初始化和随机种子完全相同，disagreement signal 会很弱
- 论文利用的是 "randomness of densification"，我们可能没有引入足够的随机性

**修复建议：**
```python
# 在初始化时使用不同的随机种子
torch.manual_seed(42)  # coarse model
gaussians_coarse = GaussianModel(...)

torch.manual_seed(123)  # fine model（不同的种子）
gaussians_fine = GaussianModel(...)
```

### 问题 4：Pseudo-view 数量不足

**推测原因：**
- 3-views 场景下，每 iteration 只生成 1 个 pseudo-view 可能不够
- 论文可能生成更多虚拟视角来弥补真实视角的稀缺

**修复建议：**
```python
# 对于 3-views，每 iteration 生成 4-8 个 pseudo-view
num_pseudo_views = 8 if len(train_cameras) <= 3 else 1
for _ in range(num_pseudo_views):
    pseudo_camera = generate_pseudo_view(...)
    loss_pseudo += compute_pseudo_coreg_loss(...)
loss_pseudo /= num_pseudo_views  # 平均
```

### 问题 5：Stage 3 启用时机可能过早

**推测原因：**
- 我们从 iteration 1000 开始启用 Stage 3
- 如果 Stage 1 的几何还不稳定，Stage 3 可能引入错误的正则化信号

**修复建议：**
```python
# 延迟启用 Stage 3 到 30% iterations
stage3_start_iter = int(0.3 * max_iterations)  # 例如 15k 中的 4500
if iteration >= stage3_start_iter:
    # 启用 pseudo-view co-regularization
    ...
```

---

## 6. 建议的修复方向

### 短期修复（1-2 天）- 高优先级

1. **实现真正的 Disagreement Loss**
   - 将 disagreement metrics 转换为可微分的损失函数
   - 添加到训练循环中，权重 λ_dis = 0.1（保守开始）
   - **预期提升**：+0.2~0.4 dB

2. **实现 Co-pruning 机制**
   - 每 5 个 densification 循环后删除不匹配的 Gaussians
   - 使用距离阈值 τ = 5.0（论文默认值，对应归一化场景）
   - **预期提升**：+0.1~0.3 dB

3. **增加 Pseudo-view 数量**
   - 对于 3-views，每 iteration 生成 4 个 pseudo-view
   - **预期提升**：+0.3~0.5 dB

### 中期修复（3-5 天）- 中优先级

4. **超参数网格搜索**
   - λ_dis: [0.01, 0.05, 0.1, 0.5]
   - λ_p: [0.5, 1.0, 1.5, 2.0]
   - σ: [0.01, 0.02, 0.05]
   - **预期找到最优组合**

5. **实现双模型不同随机种子初始化**
   - 确保 densification 的随机性差异
   - **预期提升**：+0.1~0.2 dB

6. **Warmup 策略**
   - Stage 1: 前 20% iterations λ_dis 从 0.01 → 0.1
   - Stage 3: 前 30% iterations λ_p 从 0.1 → 1.0
   - **预期提升**：+0.2~0.3 dB

### 长期优化（1-2 周）- 低优先级

7. **查阅 CoR-GS 官方代码**
   - GitHub: https://github.com/jiaw-z/CoR-GS
   - 提取关键实现细节（如精确的 loss 计算）
   - **预期：** 确认我们缺失的关键细节

8. **联系论文作者**
   - 询问 3-views 场景下的特殊超参数设置
   - 确认 disagreement loss 的具体公式

---

## 7. 预期性能提升路线图

### 保守估计（实现所有短期修复）

| 修复项 | 预期提升 | 累计 PSNR |
|--------|---------|----------|
| Baseline | - | 28.547 |
| 当前 CoR-GS (有问题) | -0.40 | 28.148 |
| + Disagreement Loss | +0.3 | 28.45 |
| + Co-pruning | +0.2 | 28.65 |
| + 更多 Pseudo-view | +0.4 | **29.05** |

**目标：** 达到 29.05 dB（+0.50 dB vs baseline）

### 乐观估计（实现所有修复+超参数调优）

| 修复项 | 预期提升 | 累计 PSNR |
|--------|---------|----------|
| 保守估计 | - | 29.05 |
| + 超参数调优 | +0.3 | 29.35 |
| + Warmup 策略 | +0.2 | 29.55 |
| + 双模型随机种子 | +0.1 | **29.65** |

**目标：** 达到 29.65 dB（+1.10 dB vs baseline）

---

## 8. 下一步行动计划

### 立即行动（今天）

1. ✅ **完成此分析报告**
2. ⏳ **提交给用户审核**
3. ⏳ **等待用户批准修复方向**

### 等待批准后（明天开始）

4. **交付给编程专家**
   - 提供详细的代码修改清单
   - 包括具体的代码片段和文件位置
5. **快速验证实验**（5k iterations 测试）
   - 验证 Disagreement Loss 是否有效
   - 验证 Co-pruning 是否提升性能
6. **完整实验**（15k iterations）
   - 使用修复后的代码重新训练
   - 对比 baseline 和修复后的 CoR-GS

---

## 附录：论文关键信息汇总

### A. 核心超参数

| 参数 | 论文值 | 说明 |
|------|--------|------|
| Co-pruning 频率 | 每 5 个 densification | 定期删除不匹配的 Gaussians |
| 距离阈值 τ | 5.0 | Gaussian 匹配距离（归一化场景） |
| 颜色平衡 λ (D-SSIM) | 0.2 | L1 vs SSIM 权重 |
| Pseudo-view 权重 λ_p | 1.0 | 默认值（3-views 可能需要更高） |
| **Disagreement 权重 λ_dis** | **未明确** | **我们的关键缺失** |
| **位置扰动 σ** | **未明确** | **我们的关键缺失** |

### B. 训练配置

| 配置项 | LLFF/DTU | MipNeRF360 |
|--------|----------|------------|
| 总 iterations | 10,000 | 30,000 |
| Densification 频率 | 每 100 iters | 每 100 iters |
| Co-pruning 触发 | 每 500 iters | 每 500 iters |
| Pseudo-view 启用 | 从 iteration 1 | 从 iteration 1 |

---

**报告生成时间：** 2025-11-18
**分析者：** 3DGS 研究专家
**文档长度：** ~1,950 字
**下一步：** 等待用户批准修复方向
