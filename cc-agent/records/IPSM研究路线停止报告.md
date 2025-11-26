# IPSM 研究路线停止报告

> **项目**: IPSM (Inline Prior Guided Score Matching) 集成到 R²-Gaussian
> **日期**: 2025-11-26
> **结论**: ❌ 研究路线失败，正式停止

---

## 1. 背景概述

### 1.1 IPSM 论文来源

IPSM (Inline Prior Guided Score Matching) 来源于 ICLR 2025 论文，原设计用于稀疏视角新视角合成任务（LLFF 数据集）。其核心思想是：

- 利用预训练的 Stable Diffusion Inpainting 模型作为图像先验
- 通过 Score Distillation 引导 3DGS 优化
- 使用深度估计和几何一致性约束 unseen 区域

### 1.2 迁移动机

鉴于 R²-Gaussian 面临的极端稀疏视角问题（仅 3 视角），我们尝试将 IPSM 的扩散先验方法迁移到 CT 重建任务中，期望利用预训练扩散模型的图像生成能力弥补稀疏视角的信息不足。

---

## 2. 实现工作量

### 2.1 代码实现（共 ~806 行）

| 文件 | 行数 | 功能 |
|------|------|------|
| `r2_gaussian/utils/depth_estimator.py` | ~172 | DPT 单目深度估计器 |
| `r2_gaussian/utils/diffusion_utils.py` | ~238 | SD Inpainting 封装 |
| `r2_gaussian/utils/ipsm_utils.py` | ~267 | X-ray Warping 模块 |
| `r2_gaussian/utils/loss_utils.py` | +116 | Pearson 深度 + 几何一致性损失 |
| `r2_gaussian/arguments/__init__.py` | +31 | IPSMParams 参数类 |
| `train.py` | +84 | IPSM 训练循环集成 |

### 2.2 研究投入

- **开发时间**: 约 6 天（2025-11-20 至 2025-11-26）
- **实验次数**: 超过 15 次调试和验证运行
- **文档编写**: 3 份详细文档（实现指南、完成报告、工作记录）

---

## 3. 实验结果

### 3.1 实验配置

```yaml
数据集: Foot-3 views
迭代次数: 30,000
IPSM 启用区间: iter 1,000 - 25,000
损失权重:
  λ_geo: 0.1 (几何一致性)
  λ_tv: 0.01 (全变分)
```

### 3.2 量化结果对比

| 指标 | Baseline (R²-Gaussian) | IPSM | 变化 |
|------|------------------------|------|------|
| **PSNR 3D** | 28.4873 | 22.35 | **-6.14 (-21.5%)** |
| **SSIM 3D** | 0.9005 | 0.719 | **-0.181 (-20.1%)** |
| PSNR 2D | ~28.49 | 28.43 | -0.06 |
| SSIM 2D | ~0.900 | 0.899 | -0.001 |

### 3.3 结论

**IPSM 显著恶化了 3D 重建质量**：

- 3D PSNR 下降 6.14 dB（约 21.5%）
- 3D SSIM 下降 0.181（约 20.1%）
- 仅 2D 投影指标持平

---

## 4. 失败原因分析

### 4.1 根本问题：Domain Gap

IPSM 的核心依赖是 **Stable Diffusion Inpainting** 预训练模型，该模型：

- 训练于 LAION-5B 等自然图像数据集
- 对 RGB 自然场景有强先验
- **从未见过 CT/X-ray 医学图像**

当应用于 CT 图像时：

```
CT 灰度图 → 伪 RGB (重复通道) → SD Inpainting → 不相关的自然图像先验
```

SD 模型会"幻觉"出自然图像的纹理、边缘和颜色，这与 CT 图像的物理特性（衰减系数分布）完全不匹配。

### 4.2 几何先验失效

IPSM 使用 DPT 深度估计器提供几何约束，但：

1. DPT 同样是在自然图像上预训练的
2. CT 投影图像的"深度"概念与自然场景完全不同
3. X-ray 是透视投影（穿透式），不是透视深度

### 4.3 Warping 假设不成立

IPSM 的 inverse warping 基于假设：

> 相邻视角的渲染结果应该在像素级别一致

但在 CT 重建中：

- 稀疏 3 视角间隔 60°，远超"相邻"范围
- X-ray 投影是积分成像，不是表面反射成像
- 几何一致性约束可能误导优化方向

### 4.4 数值稳定性问题

在实验过程中遇到的问题：

- UNet FP16 dtype mismatch（已修复）
- 深度估计返回无效值
- Score Distillation 梯度不稳定

---

## 5. 经验教训

### 5.1 自然图像先验不可直接迁移

预训练扩散模型（SD、DALL-E 等）的知识高度特化于自然图像分布。医学成像有其独特的物理模型和数据分布，简单迁移无法奏效。

### 5.2 需要领域特定的生成模型

如果要在 CT 重建中使用扩散先验，需要：

- 在 CT/X-ray 数据上预训练的扩散模型
- 理解 CT 物理成像模型的架构设计
- 大规模医学影像数据集（隐私和法规挑战）

### 5.3 简化版 IPSM 仍然失败

即使移除 Score Distillation，仅保留几何一致性约束，结果仍然明显劣于 baseline，说明问题不在于 SD 模型本身，而在于整个方法论框架不适用于 CT 成像。

---

## 6. 正式决定

**基于上述分析，正式宣布：**

> ❌ **IPSM 研究路线停止**
>
> 该技术路线已证明不适用于 R²-Gaussian 项目的 CT 稀疏重建任务。
> 相关代码将保留在 `ipsm` 分支以供参考，但不会合并到主分支。

---

## 7. 后续建议

### 7.1 不推荐的方向

- ❌ 继续调整 IPSM 超参数
- ❌ 尝试其他自然图像预训练的扩散模型
- ❌ 在当前框架下增加更多正则化

### 7.2 可能有效的替代方向

1. **CT 专用先验**
   - 使用 CT 数据预训练的生成模型
   - 医学图像特定的正则化方法

2. **物理约束增强**
   - 强化 X-ray 物理成像模型
   - 利用 CT 重建的解析解约束

3. **数据增强**
   - 更多训练视角的数据增强策略
   - 利用 CT 体数据的物理对称性

4. **其他 3DGS 改进**
   - 参考 CVPR/ECCV 2024 的 3DGS 稀疏重建方法
   - 几何结构约束（如 CoR-GS 的表面约束思想）

---

## 8. 参考资料

- IPSM 原论文：*Inline Prior Guided Score Matching for Sparse View Synthesis*
- R²-Gaussian 论文：NeurIPS 2024
- 相关文档：
  - `cc-agent/ipsm/ipsm.md` - 论文分析
  - `cc-agent/ipsm/IPSM集成实现指南.md` - 实施文档
  - `cc-agent/records/ipsm_integration_record.md` - 工作记录

---

**报告结束**

*本报告由 R²-Gaussian 研究团队撰写，记录 IPSM 技术路线的完整探索过程，为后续研究提供参考。*
