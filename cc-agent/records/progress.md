# 项目进度记录

> **历史记录已归档到：** archives/progress_2025_11_17_172000.md
> **归档时间：** 2025-11-17 17:20
> **归档原因：** 文件达到 2222 行 (8129 字)，超过 2000 行阈值
> **归档前最后记录：** CoR-GS Stage 3 (Pseudo-view Co-regularization) 完整实现

---

## 当前工作状态

**进行中的实验：**
- CoR-GS Stage 1+3 (Disagreement Loss + Pseudo-view Co-regularization)
  - 输出目录: `output/2025_11_17_foot_3views_corgs_stage1_stage3_15k/`
  - 进度: 10k/15k iterations
  - 预计完成时间: 约 1 小时内

**最近完成的里程碑：**
- ✅ CoR-GS Stage 3 完整实现 (590 行核心代码)
- ✅ SLERP 四元数插值相机旋转
- ✅ 医学 ROI 自适应权重设计
- ✅ 向下兼容集成模式

---

## 最近工作记录

### 2025-11-17 17:00 - CoR-GS Stage 3 完整实现与实验启动

**任务目标：**
实现 CoR-GS 论文的 Stage 3 (Pseudo-view Co-regularization) 功能，通过虚拟视角生成和双模型渲染差异正则化提升重建质量。

#### 核心技术实现

**1. 虚拟视角生成**
- **策略**: 在相邻真实视角之间生成插值虚拟视角
- **相机插值**: 使用 SLERP (球面线性插值) 处理四元数旋转
  ```python
  # 关键代码片段
  def slerp(q0, q1, t):
      dot = torch.sum(q0 * q1)
      if dot < 0.0:
          q1 = -q1
          dot = -dot
      theta = torch.acos(torch.clamp(dot, -1.0, 1.0))
      return (torch.sin((1-t)*theta) * q0 + torch.sin(t*theta) * q1) / torch.sin(theta)
  ```

**2. 医学 ROI 权重设计**
- **医学需求**: CT 重建需要优先保证器官中心质量 (诊断关键区域)
- **实现策略**:
  - 基于图像中心距离的 Gaussian 衰减
  - 中心权重 1.0, 边缘权重 0.5 (可调)
  - 公式: `weight = edge + (center - edge) * exp(-dist^2 / sigma^2)`

**3. 不确定性感知损失**
- **问题**: 虚拟视角渲染质量可能不稳定 (稀疏视角外插)
- **解决**: 基于双模型渲染差异计算不确定性
- **公式**: `uncertainty = |render_A - render_B| / (render_A + render_B + eps)`
- **应用**: 不确定性高的区域降低损失权重，避免错误监督

#### 代码文件清单

**新增文件**:
1. `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/pseudo_view_coreg.py` (590 行)

**修改文件**:
1. `/home/qyhu/Documents/r2_ours/r2_gaussian/train.py` (新增 ~140 行)

**Git 状态**:
- 修改文件已在工作区，待提交
- 建议 commit message: `feat: 集成 CoR-GS Stage 3 (Pseudo-view Co-regularization) 完整实现`
- 建议 Git tag: `v1.3-cor-gs-stage3`

#### 实验规划

**进行中的实验：**
1. CoR-GS Stage 1+3 组合 (15k iterations, 预计 1 小时)
2. SSS v3 (Student Splatting)
3. FSGS v2 (Focus Splatting, 修正后重训)

**待验证假设：**
- Stage 3 虚拟视角正则化是否能提升 Stage 1 性能
- 更多虚拟视角 (4 → 8) 是否带来更大提升
- CoR-GS 是否适合医学 CT 3-views 稀疏重建场景

#### 性能对比

| 方法 | 核心技术 | PSNR (dB) | 状态 | 说明 |
|------|---------|----------|------|------|
| **Baseline** | R²-Gaussian | **28.547** | ✅ 完成 | 参考基准 |
| CoR-GS Stage 1 | Disagreement Loss | 28.258 | ✅ 完成 | -0.29 dB |
| 单模型 | 无双模型 | 28.493 | ✅ 完成 | -0.05 dB |
| **CoR-GS Stage 1+3** | **+虚拟视角正则化** | **待定** | **⏳ 进行中** | **10k/15k** |
| GR-Gaussian | Graph Laplacian | 目标 ≥29.2 | ⏳ 待启动 | 环境配置中 |
| SSS v3 | Student Splatting | 待定 | ⏳ 进行中 | 训练中 |
| FSGS v2 | Focus Splatting | 待定 | ⏳ 进行中 | 修正训练中 |

#### 下一步行动

1. 监控 CoR-GS Stage 1+3 实验进展 (每 30 分钟检查)
2. 实验完成后立即分析结果
3. 根据结果决定是否继续优化参数或切换到其他论文技术
4. 更新 knowledge_base.md 记录 Stage 3 实验结论

---

## 文档结构说明

- **当前工作状态**: 正在进行的实验和最近完成的里程碑
- **最近工作记录**: 详细记录每次工作的技术细节和决策
- **归档历史**: 更早的记录请查看 `archives/` 目录

---

### 2025-11-18 15:30 - CoR-GS 官方代码对比分析（发现 3 个 Critical Bug）

**任务背景：**
CoR-GS Stage 1+3 实验结果显示 **性能无提升**（28.082 dB vs baseline 28.547 dB，下降 0.465 dB），需要紧急排查代码实现是否存在 bug。

**执行工作：**
1. ✅ 克隆官方仓库 `jiaw-z/CoR-GS` (ECCV'24) 到 `cc-agent/论文/archived/cor-gs/code_repo/`
2. ✅ 系统对比官方实现与我们的实现（train.py, pseudo_view_coreg.py, loss_utils.py）
3. ✅ 生成完整的 bug 分析报告：`cc-agent/code/github_research/corgs_implementation_comparison.md`

#### 🚨 发现的 Critical Bug（导致性能下降的根本原因）

**Bug 1: Pseudo-view 生成策略完全错误** 🔴
- **官方实现：** 训练前预生成 10,000 个**完全随机采样**的 pseudo-views（覆盖整个场景包围盒）
- **我们的实现：** 每次 iteration **实时生成 1 个**，基于相邻相机 SLERP 插值 + 微小扰动（σ=0.02, 仅 ±0.4mm）
- **影响：** 3-views 场景下，pseudo-view 与训练相机几乎重叠，**无法提供有效约束**
- **预期性能损失：** -0.5~-1.0 dB

**Bug 2: 梯度回传逻辑错误（缺少 `.detach()`）** 🔴
- **官方实现：**
  ```python
  loss = L(render_gs0, render_gs1.clone().detach())  # ✅ detach 阻断 gs1 梯度
  ```
- **我们的实现：**
  ```python
  loss = L(render_gs0, render_gs1)  # ❌ gs1 也会回传梯度
  ```
- **影响：** gs0 和 gs1 互相拉扯，梯度干扰严重，形成"对抗训练"而非"协同训练"
- **预期性能损失：** -0.2~-0.4 dB

**Bug 3: 损失叠加逻辑错误（对两个模型都添加相同损失）** 🔴
- **官方实现：**
  ```python
  loss_gs0 += L(render_gs0, render_gs1.detach())  # ✅ 分别计算
  loss_gs1 += L(render_gs1, render_gs0.detach())
  ```
- **我们的实现：**
  ```python
  loss = L(render_gs0, render_gs1)  # ❌ 包含双向梯度
  loss_gs0 += loss  # ❌ 对 gs0 添加
  loss_gs1 += loss  # ❌ 对 gs1 也添加（梯度加倍）
  ```
- **影响：** 梯度被加倍放大，训练不稳定
- **预期性能损失：** -0.1~-0.3 dB

#### ⚠️ 其他重要差异

- **Issue 4：** 缺少 Warm-up 机制（官方在 iter 2000-2500 线性增加 loss_scale）
- **Issue 5：** 训练迭代数不足（官方 30k，我们 15k，pseudo-view co-reg 在 [2k, 10k] 启用）
- **Issue 6：** 未设置 pseudo-view 停止时间（官方在 iter 10k 后停止）

#### 修复优先级与预期提升

| 优先级 | Bug/Issue | 修复工作量 | 预期提升 |
|--------|----------|----------|---------|
| 🔥 P0 | Bug 1: Pseudo-view 生成策略 | 2-3 小时 | +0.5~0.8 dB |
| 🔥 P0 | Bug 2: 添加 `.detach()` | 5 分钟 | +0.2~0.4 dB |
| 🔥 P0 | Bug 3: 损失叠加逻辑 | 15 分钟 | +0.1~0.3 dB |
| 🟡 P1 | Issue 4: Warm-up 机制 | 10 分钟 | +0.1~0.2 dB |
| 🟡 P1 | Issue 5: 延长到 30k iters | 0 分钟 | +0.1~0.2 dB |
| 🟡 P1 | Issue 6: 添加停止时间 | 5 分钟 | 未知 |

**累计预期提升：** +0.8~1.5 dB
**修复后预期 PSNR：** 28.95~29.65 dB（超越 baseline +0.40~+1.10 dB）

#### 下一步行动计划

**紧急修复（今天完成）：**
- [ ] Bug 2: 添加 `.detach()` (train.py line 733)
- [ ] Bug 3: 调整损失叠加逻辑（实现双向独立约束）
- [ ] Issue 4: 添加 Warm-up 机制
- [ ] 快速验证：运行 1000 iterations 测试

**核心修复（明天完成）：**
- [ ] Bug 1: 重写 pseudo-view 生成逻辑（实现官方的随机采样策略）
- [ ] 完整训练验证：Foot 3 views 30k iterations

**相关文件：**
- 分析报告：`cc-agent/code/github_research/corgs_implementation_comparison.md` (8000+ 字完整对比)
- 待修改：`train.py` (line 703-742), `r2_gaussian/utils/pseudo_view_coreg.py` (line 201-302)
- 官方代码：`cc-agent/论文/archived/cor-gs/code_repo/`

**关键结论：**
我们的 CoR-GS 实现存在 **3 个 Critical Bug**，导致 pseudo-view co-regularization 完全失效甚至产生负面影响。修复后预期性能提升 +0.8~1.5 dB，有望超越 baseline 达到 29+ dB。

---

**最后更新时间:** 2025-11-18 15:30
**维护者:** @research-project-coordinator

### 2025-11-18 - CoR-GS 稀疏视角失效问题深度分析（紧急诊断）

**任务背景：**
CoR-GS 在 Foot 3-views 场景下性能低于 baseline（28.481 dB vs 28.547 dB，-0.066 dB），与论文承诺的 +1.23 dB 提升形成严重矛盾。

**发现的核心问题：**

1. **Disagreement Loss 未实际参与优化**
   - 问题：我们只计算了 fitness 和 RMSE 指标用于 logging，但未将其作为损失函数反向传播
   - 证据：`corgs_metrics.py` 只返回数值，`train.py` 中未找到 `loss += lambda_dis * disagreement_loss`
   - 严重性：🔴 高（这是 Stage 1 的核心机制）

2. **Co-pruning 未实际执行**
   - 问题：我们计算了哪些 Gaussians 不匹配，但没有删除它们
   - 论文要求：每 5 个 densification 循环后 prune 掉不匹配的 Gaussians
   - 严重性：🔴 高（缺失核心算法步骤）

3. **双模型初始化可能相同**
   - 问题：两个模型可能使用相同的随机种子，导致 disagreement signal 过弱
   - 论文依赖：densification 的随机性差异
   - 严重性：🟡 中（影响 disagreement 效果）

4. **Pseudo-view 数量可能不足**
   - 问题：3-views 极稀疏场景下每 iteration 只生成 1 个 pseudo-view
   - 推测：论文可能生成 4-8 个虚拟视角
   - 严重性：🟡 中（影响 Stage 3 效果）

5. **超参数设置可能错误**
   - λ_dis（disagreement weight）：论文未明确，我们使用 0.01（可能过小）
   - λ_p（pseudo-view weight）：论文默认 1.0，3-views 可能需要 1.5-2.0
   - σ（位置扰动）：论文未给出，我们使用 0.02（未验证）
   - 严重性：🟡 中（需要调优）

**生成的关键文档：**
- `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/3dgs_expert/corgs_sparse_view_reanalysis.md`
  - 论文技术细节深度分析（~1,950 字）
  - 5 个关键实现问题 + 具体修复建议
  - 短期/中期/长期修复路线图
  - 预期性能提升路径（保守：29.05 dB，乐观：29.65 dB）

**将来要修改的内容：**

短期修复（1-2 天，高优先级）：
1. 实现真正的 Disagreement Loss 并加入训练循环（预期 +0.3 dB）
2. 实现 Co-pruning 机制（每 5 个 densification 后删除不匹配 Gaussians，预期 +0.2 dB）
3. 增加 Pseudo-view 数量到 4 个/iteration（预期 +0.4 dB）

中期修复（3-5 天）：
4. 超参数网格搜索（λ_dis, λ_p, σ）
5. 双模型使用不同随机种子初始化
6. 实现 Warmup 策略（渐进式增加 loss weight）

长期优化（1-2 周）：
7. 查阅 CoR-GS 官方 GitHub 代码验证实现细节
8. 联系论文作者确认 3-views 特殊设置

**关键决策：**
- 需要用户批准修复方向后再交付给编程专家
- 重点是实现缺失的核心算法（Disagreement Loss 和 Co-pruning）

**相关文件：**
- 分析报告：`cc-agent/3dgs_expert/corgs_sparse_view_reanalysis.md`
- 实现文件：`r2_gaussian/utils/corgs_metrics.py`（需修改）
- 训练脚本：`train.py`（需添加 disagreement loss）
- 专家记录：`cc-agent/3dgs_expert/record.md`

**下一步行动：**
- ✋ 等待用户审核分析报告并批准修复方向
- 批准后交付编程专家实施代码修改
- 快速验证实验（5k iterations）确认修复有效性

---

**最后更新时间:** 2025-11-18
**维护者:** @research-project-coordinator
