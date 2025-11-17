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

**最后更新时间:** 2025-11-17 17:20
**维护者:** @research-project-coordinator
