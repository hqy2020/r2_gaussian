# 实验记录

## [2025-11-19 14:05] 任务：DropGaussian Foot-3 实验失败诊断

**状态**: 已完成

**版本**: d1e759c (drop-gaussian branch)

**任务描述**:
对 DropGaussian 在 Foot-3 视角的训练实验进行深度诊断，识别失败原因并提供修复建议。

**诊断方法**:
1. 数据收集：读取训练输出、���估结果、Gaussian 统计
2. Good/Fail Cases 识别：分析训练曲线关键时间点
3. 数据差异对比：DropGaussian vs Baseline 全面对比
4. Root Cause Analysis：证据链推理，找出根本原因

**关键发现**:
- **主要问题**：Drop Rate 过高（80%）与稀疏视角（3 views）不兼容
- **直接证据**：
  - 平均 opacity 下降 44.5%（0.046 → 0.025）
  - 高 opacity Gaussians 减少 97.2%（106 → 3）
  - 训练后期 PSNR 下降 0.13 dB（28.25 → 28.12）
  - Gaussian 数量增加 10.5% 但质量极低

**根本原因**:
1. Drop 机制导致有效训练次数减少 80%（30000 → 6000）
2. Opacity 参数严重欠训练，无法收敛
3. Densification 过度补偿形成恶性循环
4. 训练后期随机性破坏收敛稳定性

**修复建议**（优先级排序）:
1. **P1**: 降低 Drop Rate（γ 从 0.2 → 0.5/0.7/0.9）
2. **P1**: 后期禁用 Drop（iteration > 15000）
3. **P2**: 调整 Densification 策略
4. **P3**: Importance-Aware Drop 机制

**下一步实验计划**:
- 实验 1：γ 参数扫描（0.5, 0.7, 0.9）
- 实验 2：后期禁用 Drop
- 实验 3：完全禁用 Drop（对照实验）
- 实验 4：Importance-Aware Drop（如果前 3 个成功）

**交付物**:
- 诊断报告：`cc-agent/experiments/dropgaussian_diagnosis.md` (504 行)
- 包含定量对比、训练曲线、Gaussian 统计、修复建议、实验计划

**时间成本**: ~1.5 小时（数据收集 + 分析 + 报告撰写）
