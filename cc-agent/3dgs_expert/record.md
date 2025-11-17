# 3DGS Expert Work Record

## Current Task
**Task:** CoR-GS Stage 3 - Pseudo-view Co-regularization 深度分析与实现方案设计
**Paper:** CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
**Status:** ✅ 完成 - 等待用户批准
**Start Time:** 2025-11-17
**Version:** v4.0

## Context
- **User Decision**: 跳过 Stage 2 (Co-pruning)，直接实施 Stage 3
  - 原因：Stage 2 基线过低（28.148 dB），预期收益仅 +0.40 dB（持平 baseline）
  - Stage 3 优势：论文 Table 6 显示单独效果 +1.04 dB（更显著）
- **Stage 1 Current Status**: PSNR 28.148 dB, SSIM 0.9003（-0.40 dB vs baseline 28.547）
  - Geometry Fitness: 1.0（粗/精模型完美对齐）
  - RMSE: 0.012 mm（几何一致性优异）
  - **核心问题**：几何-渲染脱钩（几何完美但渲染不足）
- **Goal**: 通过 Pseudo-view Co-regularization 突破稀疏视角约束，目标 +1.04 dB → 29.188 dB

## Deliverables
1. ✅ `innovation_analysis_corgs_stage3.md` (核心技术原理 + 算法流程, ~1,980 字)
2. ✅ `implementation_plan_corgs_stage3.md` (完整代码方案 + 时间表, ~2,450 字)

## Key Analysis Findings
- **Pseudo-view 生成策略**: 从相邻训练视角插值旋转（SLERP）+ 位置扰动（ε~N(0, σ²)）
- **Co-regularization 损失**: R_pcolor = 0.8 * L1(I'¹, I'²) + 0.2 * D-SSIM(I'¹, I'²)
- **总损失**: L = L_color + λ_p * R_pcolor（λ_p = 1.0）
- **训练流程**: 每 iteration 生成 1 个 pseudo-view，渲染两个模型，计算一致性损失

## Expected Performance
- **Stage 3 单独**: +1.04 dB → 29.188 dB（论文数据）
- **Stage 1+3 组合**: +1.20 dB（协同效应 +0.16 dB）
- **保守预期**: Foot 3 views 从 28.148 → 28.85~29.19 dB
- **vs Baseline 28.547**: +0.30~0.64 dB 提升

## Implementation Highlights
- **核心文件**: `r2_gaussian/utils/pseudo_view_coreg.py` (~150 行)
- **修改文件**: `train.py` (~130 行新增/修改)
- **新增参数**: `--enable_pseudo_coreg`, `--lambda_pseudo`, `--pseudo_noise_std`
- **向下兼容**: 不启用时训练流程保持不变
- **实施周期**: 7-10 天（算法实现 → 集成 → 测试 → 实验）

## Technical Challenges
1. 四元数 SLERP 插值正确性（需单元测试验证）
2. Pseudo-view 相机参数完整性（FoV, intrinsics 复制）
3. 3 views 极度稀疏场景下 pseudo-view 质量
4. 计算开销增加 ~40%（额外渲染 pseudo-views）

## Next Steps (待用户批准)
1. ✋ 用户批准实施方案
2. → 交付给编程专家执行代码实现
3. → 单元测试验证算法正确性
4. → 快速实验验证（5k iterations）
5. → 完整训练实验（15k iterations）
6. → 超参数调优（如需要）

## Previous Task History
2. Pixel-Graph-Aware Gradient Strategy (PGA) - Graph 构建与梯度增强
3. Graph Laplacian Regularization - 损失函数集成

## User Approved Decisions
- ✅ 实施方案: 完整实施 (De-Init + PGA + Graph Laplacian)
- ✅ 依赖库: 使用 PyTorch Geometric (提供 fallback)
- ✅ 超参数: 论文默认值 (k=6, λ_g=1e-4, λ_lap=8e-4, σ_d=3)
- ✅ 优先级: 核心功能快速验证

## Current Deliverables
- ✅ `innovation_analysis_gr_gaussian.md` (2025-11-17, 1998 字)
- ✅ `implementation_plan_gr_gaussian.md` (2025-11-17, 2487 字)
  - 完整的架构设计图和数据流向
  - 文件级代码修改方案（含伪代码）
  - PyTorch Geometric 安装指南
  - 7-10 天实施时间表
  - 单元测试和集成测试计划

## Performance Context
- Baseline (foot 3 views): PSNR 28.547 dB, SSIM 0.9008
- 预期提升: +0.5~0.8 dB → 29.05~29.35 dB

---

## Previous Task (Archived)
**Task:** CoR-GS 实现方案设计 (已完成)
**Paper:** CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
**Status:** ✅ Completed
**Deliverables:**
- ✅ `corgs_innovation_analysis.md` (2025-11-16, 1987 字)
- ✅ `corgs_medical_feasibility_report.md` (医学专家, 2025-11-16, 1998 字)
- ✅ `implementation_plans/corgs_implementation_plan.md` (预计交付)
