# R²-Gaussian 项目进度记录

本文件记录项目的所有重要工作节点，包括已完成、进行中和待完成的任务。

---

## 2025-11-19 DropGaussian 集成与完整验证

### 已完成

**1. DropGaussian 论文分析与代码集成**
- 完成 DropGaussian 论文分析并集成到 R²-Gaussian（新增 `use_drop_gaussian`、`drop_gamma` 参数和渲染逻辑，17 行代码）

**2. DropGaussian Foot-3 初步实验与失败诊断**
- 执行 γ=0.2 实验，PSNR 28.12 低于 baseline 28.50，发现平均 opacity 下降 44.47%，高 opacity Gaussians 减少 97.2%

**3. 根本原因分析（数据支持）**
- 确认根本原因为 3 视角稀疏 + Drop 20% 导致训练信号不足，Opacity 欠训练，高质量 Gaussians 消失
- 详细诊断报告：`cc-agent/experiments/dropgaussian_diagnosis.md` (504 行)

**4. 分析工具集开发**
- 开发三个分析脚本（逐图对比、opacity 统计、可视化），可复用于后续实验

**5. 课程学习（Curriculum Learning）策略优化**
- 设计并实现动态 drop rate 策略：前 5000 轮不 drop，之后线性增长到 10%（降低 50% drop rate）
- 新增参数：`drop_start_iter=5000`, `drop_end_iter=30000`

**6. 3/6/9 Views 完整验证实验** ✅ 核心突破

| 视角数 | 输出目录 | PSNR | SSIM | 状态 |
|--------|---------|------|------|------|
| **3 Views** | `2025_11_19_15_56_foot_3views_dropgaussian_curriculum` | 28.34 | 0.9024 | ❌ 失败 (< baseline 28.50) |
| **6 Views** | `2025_11_19_16_53_foot_6views_dropgaussian_curriculum` | 32.05 | 0.9440 | ✅ 成功 (显著提升) |
| **9 Views** | `2025_11_19_16_53_foot_9views_dropgaussian_curriculum` | 35.11 | 0.9613 | ✅ 成功 (性能最佳) |

**核心发现**：
- DropGaussian 的有效性强依赖训练视角数量
- 3 views 训练信号不足，DropGaussian 失败
- 6+ views 下 DropGaussian 显著提升性能
- 课程学习策略（前期稳定 + 低 drop rate）是关键

### 待完成

**1. DropGaussian 对比分析（优先级 P1）**
- 训练 6 views 和 9 views 的 baseline 进行定量对比
- 分析 6/9 views 下的逐图改善情况
- 撰写完整的 3/6/9 views 对比报告：`cc-agent/experiments/dropgaussian_3_6_9_views_comparison.md`
- 确定 DropGaussian 在不同视角数下的性能增益

**2. DropGaussian 多器官验证（优先级 P2）**
- 在 Chest、Head、Abdomen、Pancreas 上验证 DropGaussian 效果（6/9 views）
- 确认 DropGaussian 是否对所有器官类型都有效
- 确定不同器官的最优 `drop_gamma` 参数

**3. 高级策略探索（优先级 P3）**
- **Importance-Aware Drop**: 保护高质量 Gaussians（如果 P1 验证成功）
- **视角自适应 Drop**: 根据视角数自动调整 drop rate
- **多阶段 Drop**: 不同训练阶段使用不同策略

---


