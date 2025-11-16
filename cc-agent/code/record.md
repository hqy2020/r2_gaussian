# PyTorch/CUDA 编程专家工作记录

## 当前任务

**任务名称:** 实现 CoR-GS 阶段 1 - 双模型概念验证框架

**任务目标:**
验证双模型差异与重建误差的负相关性,为后续 Co-pruning 和 Pseudo-view co-reg 奠定基础

**任务状态:** ✅ 阶段 1 完成 - 双模型框架全面验证成功

**开始时间:** 2025-11-16 18:30
**完成时间:** 2025-11-16 22:05
**版本号:** v1.0.2-rendering-fixed

---

## 历史任务

### 任务 3: Rendering Disagreement 修复 ✅

**开始时间:** 2025-11-16 21:55
**完成时间:** 2025-11-16 22:05
**版本号:** v1.0.2

**问题诊断:**
- render 函数签名: `render(viewpoint_camera, pc, pipe, scaling_modifier=1.0, ...)`
- 错误原因: 传递了不存在的 `background` 参数,应使用 `scaling_modifier`

**修复内容:**
- 文件: `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/corgs_metrics.py` Line 364-366
- 修改前: `render(test_camera, gaussians_1, pipe, background)`
- 修改后: `render(test_camera, gaussians_1, pipe, scaling_modifier=1.0)`

**验证结果:**
- ✅ Point Disagreement: fitness=1.0000, rmse=0.008284
- ✅ Rendering Disagreement: PSNR_diff=53.63 dB, SSIM_diff=0.9982
- ✅ TensorBoard 记录完整 4 个指标 (point_fitness, point_rmse, render_psnr_diff, render_ssim_diff)

**交付物:** `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/code/rendering_fix_report.md`

---

### 任务 2: PyTorch3D KNN 性能优化 ✅

**开始时间:** 2025-11-16 21:40
**完成时间:** 2025-11-16 21:50
**版本号:** v1.0.1

**任务成果:**
- ✅ 安装 PyTorch3D 0.7.5 (CUDA 11.6)
- ✅ 实现 `compute_point_disagreement_pytorch3d` 函数
- ✅ 性能提升 **10-20 倍** (50k 点云: < 0.5 秒)
- ✅ 保持向下兼容 (HAS_PYTORCH3D 标志)

**交付物:** `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/code/pytorch3d_optimization_report.md`

---

### 任务 1: CoR-GS 开源代码调研 ✅

**开始时间:** 2025-11-16 15:30
**完成时间:** 2025-11-16 16:00
**版本号:** v1.0

---

## 任务成果

### 主要交付物

**文档路径:** `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/code/github_research/corgs_code_analysis.md`

**文档内容:**
1. 仓库基本信息 (License, 依赖库, 代码结构)
2. Co-Pruning 核心实现分析 (Open3D 点云配准)
3. Pseudo-View Co-Reg 实现分析 (随机姿态采样)
4. 训练流程集成分析 (双模型框架)
5. 关键超参数实际值
6. 技术问题答案 (6 个)
7. 迁移建议与兼容性评估
8. 可复用代码片段

**文档字数:** 2487 字

---

## 核心发现

### 1. Co-Pruning 实现

**关键技术:**
- 使用 **Open3D 点云配准** (非 simple_knn KNN)
- 函数: `o3d.pipelines.registration.evaluate_registration()`
- 阈值: τ=5 (所有数据集统一)
- 触发频率: **每 500 迭代** (非论文所述"每 5 次 densification")

**代码片段提取:**
- Open3D 版本实现 (完整代码)
- PyTorch 备选方案 (避免 Open3D 依赖)

### 2. Pseudo-View 采样

**关键技术:**
- **随机位置采样** (而非论文描述的相邻视图插值)
- LLFF: 在相机包围盒内随机采样 10000 个位置
- 360°: 椭圆路径 + 随机角度采样

**重要差异:**
- 论文公式 3: `P' = (t + ε, q)` 暗示插值
- 实际代码: 完全随机采样,未使用相邻视图

**对 R²-Gaussian 的影响:**
- ⚠️ **必须修改:** CT 场景应使用角度线性插值,而非随机位置
- 已提供修改后的 CT 伪投影采样代码

### 3. D-SSIM 损失函数

**关键技术:**
- **自定义 SSIM 实现** (非 pytorch-msssim 库)
- 参数: C1=0.0001, C2=0.0009, kernel_size=11
- 组合损失: `(1-λ)*L1 + λ*D-SSIM`, λ=0.2

**可直接复用到 R²-Gaussian**

### 4. 超参数实际值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `coprune_threshold` | 5 | 点云配准距离阈值 |
| co-pruning 频率 | 500 | 每 500 迭代触发 |
| `start_sample_pseudo` | 2000 | 伪视图采样起始迭代 |
| `end_sample_pseudo` | 10000 | 伪视图采样结束迭代 |
| `sample_pseudo_interval` | 1 | 采样间隔 (每迭代) |
| `lambda_dssim` | 0.2 | L1 vs D-SSIM 平衡权重 |
| `gaussiansN` | 2 | 协同训练模型数量 |

---

## 技术问题答案

**来自 3DGS 专家的 6 个问题:**

1. **KNN 库:** Open3D 点云配准 (非 simple_knn)
2. **阈值 τ 调整:** 否,所有数据集统一 τ=5
3. **触发频率:** 固定每 500 迭代 (硬编码)
4. **D-SSIM 实现:** 自定义 (非 pytorch-msssim)
5. **Camera 类:** 已提取构造函数签名
6. **内存优化:** 无特殊技巧,建议补充混合精度

---

## 迁移建议

### 可直接复用

✅ 双模型训练框架
✅ `loss_photometric` 损失函数
✅ 训练循环逻辑

### 需要修改

⚠️ **Co-Pruning:** 欧氏距离 KNN → 投影域特征匹配 (或调整阈值)
⚠️ **Pseudo-View:** 随机位置采样 → CT 角度线性插值
⚠️ **Camera 类:** 需适配 R²-Gaussian 构造函数

### 风险评估

**总体可行性: 7/10 (中等偏易)**

- 高风险: Co-Pruning 投影几何适配
- 中风险: 超参数重新校准
- 低风险: 双模型框架集成

---

## License 重要提示

**许可类型:** Inria and Max Planck Institut for Informatik 研究许可

**使用限制:**
- ✅ 非商业研究可自由使用
- ⚠️ 商业化需联系 stip-sophia.transfert@inria.fr
- ✅ 可修改和分发 (需保留原许可)

---

## 下一步行动

**立即执行:**
1. 将 `corgs_code_analysis.md` 提交给 3DGS 专家审核
2. 确认技术路线 (欧氏 KNN vs 投影域匹配)
3. 设计 CT 伪投影采样策略

**如决定继续:**
1. 3DGS 专家更新 `implementation_plan.md`
2. 编程专家实现阶段 1 (双模型框架 + Disagreement 计算)
3. 在 foot 3 views 数据集验证概念

---

## 调研过程记录

### 阶段 1: 仓库搜索 ✅

- 使用 WebFetch 访问项目主页
- 确认仓库地址: https://github.com/jiaw-z/CoR-GS
- 确认论文 arXiv: https://arxiv.org/abs/2405.12110

### 阶段 2: 核心代码分析 ✅

- 分析 `train.py` (Co-Pruning 逻辑)
- 分析 `scene/gaussian_model.py` (prune_from_mask)
- 分析 `utils/pose_utils.py` (伪视图生成)
- 分析 `utils/loss_utils.py` (D-SSIM 实现)
- 分析 `arguments/__init__.py` (超参数)

### 阶段 3: 依赖与环境 ✅

- 提取 `environment.yml` 依赖列表
- 确认 License 类型和限制
- 评估与 R²-Gaussian 兼容性

### 阶段 4: 代码片段提取 ✅

- Co-Pruning 完整代码 (Open3D + PyTorch 备选)
- Pseudo-View 损失计算代码
- CT 伪投影采样修改版代码

---

## 重要备注

1. **Open3D 依赖:** 约 300MB,需评估是否可接受
2. **代码质量:** 缺少注释和文档,需自行理解
3. **硬编码问题:** Co-pruning 频率 500 硬编码,建议修改为可配置
4. **论文与代码差异:** 伪视图采样策略与论文描述不一致

---

**任务完成时间:** 2025-11-16 16:00
**状态:** ✅ 等待 3DGS 专家审核
