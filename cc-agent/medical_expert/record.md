# 医学 CT 影像专家工作记录

---

## 当前任务

**任务名称:** CoR-GS Stage 3 (Pseudo-view Co-regularization) 医学可行性评估
**开始时间:** 2025-11-17 晚上
**完成时间:** (进行中)
**版本号:** v1.0
**状态:** ⏳ 进行中

### 任务目标
评估 CoR-GS Stage 3 Pseudo-view Co-regularization 机制在 Foot 3 视角极稀疏 CT 重建中的医学适用性，重点关注虚拟视角生成的医学合理性、临床价值和风险。

### 评估背景
- **当前性能**: Stage 1 PSNR 28.148 dB (-0.40 dB vs baseline 28.547 dB)
- **几何质量**: Fitness=1.0, RMSE=0.012 mm (完美几何对齐)
- **核心问题**: 几何-渲染脱钩 (几何对齐但密度/不透明度未协同)
- **论文预期**: Stage 3 在 3-views 场景贡献 +0.70~1.04 dB
- **临床基准**: PSNR ≥28.5 dB 为诊断可用阈值

### Stage 3 核心机制
1. **Pseudo-view 生成**: 从相邻训练视角通过四元数 SLERP 插值生成虚拟相机，位置添加 ε~N(0, σ²=0.02²) 扰动
2. **Co-regularization 损失**: R_pcolor = 0.8×L1(I'¹, I'²) + 0.2×D-SSIM(I'¹, I'²)
3. **总损失**: L = L_color + λ_p×R_pcolor (λ_p=1.0)

---

## 历史任务归档

### 任务 1: CoR-GS Stage 1 双模型剪枝方法医学可行性评估
**任务状态:** ✅ 已完成
**时间:** 2025-11-16
**核心结论:** 中等至高可行性 (⭐⭐⭐⭐), 需投影域适配
**交付物:** `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/medical_expert/corgs_medical_feasibility_report.md`

### 任务 2: CoR-GS Stage 2 (Co-pruning) 医学可行性评估
**任务状态:** ✅ 已完成
**时间:** 2025-11-17
**版本号:** v1.0
**核心结论:** ⭐⭐⭐☆☆ (中等可行性)
**最终建议:** ❌ 不建议立即实施
**交付物:** `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/medical_expert/medical_feasibility_corgs_stage2.md`

**关键理由:**
- 基线过低问题: Stage 1 (28.15 dB) + Co-pruning 预期 (+0.40 dB) = 28.55 dB，仅勉强达标
- 几何已饱和: Fitness=1.0 说明双模型几何对齐已完美，Co-pruning 改善空间有限
- 风险-收益不平衡: 3 views 极度稀疏场景下，剪除 30%+ Gaussian 点可能导致解剖细节丢失

### 任务 3: GR-Gaussian 技术医学可行性评估
**任务状态:** ✅ 已完成
**时间:** 2025-11-17
**版本号:** v1.0
**临床评估:** ⭐⭐⭐⭐½ (高临床可行性)
**交付物:** `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/medical_expert/medical_feasibility_gr_gaussian.md`

**三项创新点医学价值:**
1. **Denoised Point Cloud Initialization:** ⭐⭐⭐⭐½ (直击 3 视角 FDK 初始化噪声)
2. **Pixel-Graph-Aware Gradient:** ⭐⭐⭐⭐⭐ (解决针状伪影，最大临床痛点)
3. **Graph Laplacian Regularization:** ⭐⭐⭐⭐ (符合组织密度连续性假设)

**预期提升:** PSNR +0.65 dB → 29.22 dB (跨越临床可用阈值)

---

**版本:** v1.0
**最后更新:** 2025-11-17
