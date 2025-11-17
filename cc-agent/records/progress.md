# 项目进度记录

---

## 2025-11-18 SSS (Student Splatting and Scooping) v6 Bug修复与重新训练

### 任务概述

**任务名称：** SSS (Student Splatting and Scooping) v6 Bug修复与重新训练

**任务目标：**
- 诊断 SSS-v5 训练失败的原因（PSNR 20.16 dB，100%负值）
- 修复发现的 3 个致命 Bug
- 重新训练 SSS-v6 验证修复效果

**执行状态：** 进行中（代码修复完成，正在训练）

**时间戳：** 2025-11-18

---

### 关键发现：3 个致命 Bug

#### Bug 1: Densification 负值传播
**位置：** `r2_gaussian/gaussian/gaussian_model.py` (行 702, 751)

**问题描述：**
- 新增高斯点直接复制父点的负 opacity
- 导致负值像病毒一样传播，无法自我修正

**修复方案：**
- 基于 density 重新初始化新高斯点的 opacity
- 确保倾向于正值（0.5-0.8 范围）
- 在 `densification_postfix()` 和 `densify_and_split()` 中实现

#### Bug 2: Balance Loss 梯度失效
**位置：** `train.py` (行 818)

**问题描述：**
- 使用 `torch.abs(pos_count - target)` 导致当 pos_count→0 时梯度→0
- 完全失去纠正能力，无法引导优化器回到正值

**修复方案：**
- 改用直接惩罚负值的损失函数
- 新损失：`negative_penalty * 0.5 + positive_encouragement * 0.2`
- 直接惩罚负值数量，鼓励正值数量

#### Bug 3: Opacity 激活范围过大
**位置：** `r2_gaussian/gaussian/gaussian_model.py` (行 73)

**问题描述：**
- `tanh` 允许 [-1, 1] 完整范围
- 容易让优化器滑向负值，无自然边界保护

**修复方案：**
- 将激活函数从 `tanh` 改为偏移 sigmoid
- 新范围：[-0.2, 1.0]
- 保留轻微负值灵活性，同时大幅偏向正值

---

### 实验结果对比

| 版本 | PSNR (dB) | SSIM | Positive Ratio | 状态 |
|------|-----------|------|----------------|------|
| SSS-v5 (Bug版本) | 20.16 | 0.778 | 0% | 失败 ❌ |
| SSS-v6 (修复版本) | 训练中 | 训练中 | 初始化 100% | 进行中 ⏳ |
| 目标 | ≥ 28.0 | ≥ 0.85 | ≥ 90% | 待验证 |

**SSS-v6 初始化验证：**
- Opacity 范围：[0.01, 0.47]
- Positive Ratio：100% ✅
- 初始化质量：良好

---

### 修改的文件清单

1. **`r2_gaussian/gaussian/gaussian_model.py`**
   - 修复 Bug 1：`densification_postfix()` 和 `densify_and_split()` 重新初始化 opacity
   - 修复 Bug 3：将 opacity 激活函数从 `tanh` 改为偏移 sigmoid

2. **`train.py`**
   - 修复 Bug 2：重写 SSS balance loss 逻辑

3. **`scripts/train_foot3_sss_v6.sh`**
   - 新增训练脚本，用于 SSS-v6 实验

---

### 当前训练状态

**训练任务 ID：** 98aa8e

**训练配置：**
- 输出目录：`output/2025_11_18_foot_3views_sss_v6`
- 迭代数：30000
- 数据集：Foot (3 views)
- 预计训练时间：8-10 小时

**监控命令：**
```bash
watch -n 30 'tail -n 50 output/2025_11_18_foot_3views_sss_v6/train_log.txt | grep -E "(Iteration|PSNR|SSIM|Balance|Positive|opacity)"'
```

---

### 下一步行动计划

**待完成任务：**
1. 等待 SSS-v6 训练完成（预计 8-10 小时）
2. 分析训练结果指标：
   - PSNR（目标 ≥ 28 dB）
   - SSIM（目标 ≥ 0.85）
   - Positive Ratio（目标 ≥ 90%）
3. 对比性能：v5 vs v6 vs Baseline
4. 可视化分析：渲染结果质量对比

**决策点：**
- 如果成功（PSNR ≥ 28 dB）：
  - 记录到 `knowledge_base.md` 作为成功案例
  - 考虑进一步优化超参数
- 如果失败（PSNR < 28 dB）：
  - 深入分析失败原因
  - 评估是否需要调整损失函数权重
  - 考虑放弃 SSS 方法或寻找替代方案

---

### 技术总结

**核心教训：**
1. Densification 过程中必须重新初始化敏感参数（如 opacity）
2. 损失函数设计需考虑梯度有效性，避免梯度消失
3. 激活函数选择应有合理的输出范围偏向性
4. 负值传播具有"病毒式"扩散特性，需要在源头阻断

**适用场景：**
- 任何涉及 opacity 优化的 3DGS 变体
- 需要控制参数正负性的深度学习任务
- Densification 过程中的参数初始化策略

**风险提示：**
- SSS 方法本质上挑战了 3DGS 的核心设计假设（opacity ∈ [0, 1]）
- 即使修复 Bug，仍可能面临收敛困难
- 需要密切监控训练过程中的 Positive Ratio

---

*记录者：@research-project-coordinator*
*记录时间：2025-11-18*
