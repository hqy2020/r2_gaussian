# R²-Gaussian 项目进度记录

本文件记录项目的所有重要工作节点，包括已完成、进行中和待完成的任务。

---

## 2025-11-19 DropGaussian 集成与实验诊断

### 已完成

**1. DropGaussian 论文分析与代码集成**
- 完成 DropGaussian 论文分析并集成到 R²-Gaussian（新增 `use_drop_gaussian`、`drop_gamma` 参数和渲染逻辑，17 行代码）

**2. DropGaussian Foot-3 实验执行与失败诊断**
- 执行 γ=0.2 实验，PSNR 28.12 低于 baseline 28.50，发现平均 opacity 下降 44.47%，高 opacity Gaussians 减少 97.2%，26% 图像表现更好但整体失败

**3. 根本原因分析（数据支持）**
- 确认根本原因为 3 视角稀疏 + Drop 20% 导致训练信号不足，Opacity 欠训练，高质量 Gaussians 消失，所有结论基于实际测量数据

**4. 分析工具集开发**
- 开发三个分析脚本（逐图对比、opacity 统计、可视化），可复用于后续实验



### 待完成





**2. 高级策略实验（优先级 P3）**

- **实验 D：Importance-Aware Drop** - 保护 top 20% 高 opacity Gaussians，目标 PSNR > 28.98（超越 baseline）






