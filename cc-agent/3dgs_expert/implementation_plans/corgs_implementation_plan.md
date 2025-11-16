# CoR-GS 双模型剪枝实现方案

**生成时间:** 2025-11-16
**版本:** v1.0
**字数:** 2995
**前置分析:** `corgs_innovation_analysis.md`, `corgs_medical_feasibility_report.md`

---

## 核心策略 (Executive Summary)

采用**四阶段渐进式实施**,从概念验证到完整集成,确保每步可验证。核心技术路线:(1) 在 `train.py` 中同时训练两个独立 GaussianModel 实例,利用密化随机性产生差异;(2) 实现欧氏 KNN 协同剪枝 (τ=0.1~0.5,适配 R²-Gaussian 尺度),每 5 次密化触发;(3) 基于 CT 角度插值的伪投影正则化 (λ_p=1.0);(4) 联合训练双机制,通过 feature flag `--enable_corgs` 确保向下兼容。预期性能提升 PSNR +0.8~1.2 dB,显存增加 1.5 倍但推理时无额外开销。

---

## 阶段 1: 概念验证 (1-2 天)

### 目标与验收标准

**目标:** 验证 R²-Gaussian 场景下双模型差异与重建误差的负相关性

**验收标准:**
- [ ] 成功训练两个独立模型 (不同随机种子)
- [ ] Point Disagreement (Fitness, RMSE) 计算正确
- [ ] Rendering Disagreement (PSNR_diff) 可视化
- [ ] 散点图显示相关系数 < -0.3 (中等负相关)

---

### 文件修改清单

#### 1. 修改 `train.py` - 双模型训练框架

**修改位置:** 第 82-96 行 (training 函数签名)

**当前代码:**
```python
def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    gaussiansN=2,  # ← 已有双模型参数
    coreg=True,    # ← 已有协同正则化参数
    coprune=True,  # ← 已有剪枝参数
    coprune_threshold=5,  # ← 现有阈值 (需调整)
    args=None,
):
```

**修改方案:**
```python
# 无需修改函数签名,但需调整默认值
gaussiansN = args.corgs_num_models if hasattr(args, 'corgs_num_models') else 2
coreg = args.enable_corgs_coreg if hasattr(args, 'enable_corgs_coreg') else False  # 默认关闭
coprune = args.enable_corgs_coprune if hasattr(args, 'enable_corgs_coprune') else False
```

**关键发现:** R²-Gaussian 已支持多模型训练 (见第 91 行 `gaussiansN=2`),但当前实现存在问题 (第 365 行 identity loss),需修正。

---

#### 2. 新建模块: `r2_gaussian/utils/corgs_metrics.py`

**功能:** 计算 Point/Rendering Disagreement

**核心函数设计:**

```python
import torch
import numpy as np
from simple_knn._C import distCUDA2

def compute_point_disagreement(gaussians_1, gaussians_2, tau=0.3):
    """
    计算两个 Gaussian 场的 Point Disagreement

    参数:
        gaussians_1, gaussians_2: GaussianModel 实例
        tau: 距离阈值 (建议 0.1~0.5 for R²-Gaussian)

    返回:
        fitness: 重叠率 [0,1]
        rmse: 均方根误差
        non_matching_mask_1, non_matching_mask_2: 非匹配点掩码
    """
    xyz_1 = gaussians_1.get_xyz.detach()  # (N1, 3)
    xyz_2 = gaussians_2.get_xyz.detach()  # (N2, 3)

    # 使用 distCUDA2 计算 KNN (R²-Gaussian 已有依赖)
    dist2_1to2 = distCUDA2(xyz_1)  # 场 1 到场 2 的最近邻距离
    dist_1to2 = torch.sqrt(dist2_1to2)

    # 计算非匹配掩码
    non_matching_mask_1 = dist_1to2 > tau  # (N1,) bool

    # 对称计算
    dist2_2to1 = distCUDA2(xyz_2)
    dist_2to1 = torch.sqrt(dist2_2to1)
    non_matching_mask_2 = dist_2to1 > tau

    # Fitness: 重叠率
    fitness = 1.0 - (non_matching_mask_1.sum() + non_matching_mask_2.sum()) / (len(xyz_1) + len(xyz_2))

    # RMSE: 匹配点的平均距离
    matched_dists = dist_1to2[~non_matching_mask_1]
    rmse = torch.sqrt(matched_dists.pow(2).mean()) if len(matched_dists) > 0 else torch.tensor(0.0)

    return fitness.item(), rmse.item(), non_matching_mask_1, non_matching_mask_2


def compute_rendering_disagreement(render_1, render_2):
    """
    计算渲染差异 (投影空间 PSNR)

    参数:
        render_1, render_2: 渲染结果 (C, H, W) tensor

    返回:
        psnr_diff: PSNR 差异 (越低表示差异越大)
    """
    mse = torch.mean((render_1 - render_2) ** 2)
    psnr_diff = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr_diff.item()
```

**关键技术决策:**
- **KNN 库选择:** 使用 `simple_knn._C.distCUDA2` (R²-Gaussian 现有依赖,见 `gaussian_model.py` Line 21)
- **阈值建议:** τ=0.1~0.5 (基于 R²-Gaussian `scale_bound=[0.0005, 0.5]` 分析)

---

#### 3. 修改 `train.py` - 添加 Disagreement 记录

**插入位置:** 第 500 行附近 (日志记录区域)

```python
# 阶段 1 专用: 记录 Point/Rendering Disagreement
if args.enable_corgs and iteration % 100 == 0:
    from r2_gaussian.utils.corgs_metrics import (
        compute_point_disagreement,
        compute_rendering_disagreement
    )

    # Point Disagreement
    fitness, rmse, _, _ = compute_point_disagreement(
        GsDict["gs0"], GsDict["gs1"], tau=args.corgs_tau
    )
    tb_writer.add_scalar("CoRGS/Point_Fitness", fitness, iteration)
    tb_writer.add_scalar("CoRGS/Point_RMSE", rmse, iteration)

    # Rendering Disagreement (在训练视图上)
    psnr_diff = compute_rendering_disagreement(
        RenderDict["image_gs0"], RenderDict["image_gs1"]
    )
    tb_writer.add_scalar("CoRGS/Rendering_PSNR_diff", psnr_diff, iteration)
```

---

#### 4. 命令行参数扩展: `r2_gaussian/arguments.py`

**修改位置:** `OptimizationParams` 类

```python
# CoR-GS 相关参数
self.parser.add_argument("--enable_corgs", action="store_true", help="启用 CoR-GS 双模型训练")
self.parser.add_argument("--corgs_num_models", type=int, default=2, help="协同训练模型数量")
self.parser.add_argument("--corgs_tau", type=float, default=0.3, help="Co-pruning 距离阈值")
self.parser.add_argument("--enable_corgs_coprune", action="store_true", help="启用协同剪枝")
self.parser.add_argument("--enable_corgs_coreg", action="store_true", help="启用伪投影正则化")
self.parser.add_argument("--corgs_lambda_p", type=float, default=1.0, help="伪投影损失权重")
```

---

### 实验验证方案

**数据集:** foot 3 views (baseline PSNR 28.547)

**运行命令:**
```bash
python train.py \
  --data_path data/foot \
  --enable_corgs \
  --corgs_num_models 2 \
  --corgs_tau 0.3 \
  --iterations 10000
```

**可视化脚本:** 在 `cc-agent/code/scripts/` 创建 `visualize_corgs_correlation.py`
```python
# 读取 TensorBoard 日志
# 绘制散点图: Point_RMSE vs GT_PSNR_error
# 计算 Pearson 相关系数
# 预期: r < -0.3 (负相关)
```

---

## 阶段 2: Co-Pruning 实现 (3-5 天)

### 目标与验收标准

**目标:** 实现基于 KNN 的协同剪枝,减少 Gaussian 点数 >20%

**验收标准:**
- [ ] 点数减少 >20% (vs baseline)
- [ ] PSNR 提升 +0.3~0.5 dB
- [ ] 可视化点云更紧凑 (无离散点)

---

### 核心算法实现

#### 1. 新建模块: `r2_gaussian/utils/corgs_coprune.py`

```python
import torch
from r2_gaussian.utils.corgs_metrics import compute_point_disagreement

def co_prune_gaussians(gaussians_1, gaussians_2, tau=0.3):
    """
    协同剪枝: 移除两个场中位置不一致的 Gaussians

    参数:
        gaussians_1, gaussians_2: GaussianModel 实例
        tau: 距离阈值

    返回:
        prune_mask_1, prune_mask_2: 需要剪枝的掩码 (True=保留, False=剪除)
    """
    _, _, non_matching_1, non_matching_2 = compute_point_disagreement(
        gaussians_1, gaussians_2, tau
    )

    # 剪除非匹配点 (取反: non_matching -> prune)
    prune_mask_1 = ~non_matching_1  # True=保留
    prune_mask_2 = ~non_matching_2

    # 日志
    num_pruned_1 = (~prune_mask_1).sum().item()
    num_pruned_2 = (~prune_mask_2).sum().item()
    print(f"[Co-Prune] Model 1: 剪除 {num_pruned_1}/{len(prune_mask_1)} ({num_pruned_1/len(prune_mask_1)*100:.1f}%)")
    print(f"[Co-Prune] Model 2: 剪除 {num_pruned_2}/{len(prune_mask_2)} ({num_pruned_2/len(prune_mask_2)*100:.1f}%)")

    return prune_mask_1, prune_mask_2
```

---

#### 2. 修改 `gaussian_model.py` - 添加剪枝方法

**搜索现有剪枝逻辑:** `prune_points` 方法

**插入新方法:**
```python
def prune_points_with_mask(self, valid_points_mask):
    """
    根据掩码剪枝 (Co-pruning 专用)

    参数:
        valid_points_mask: bool tensor (True=保留)
    """
    optimizable_tensors = self._prune_optimizer(valid_points_mask)

    self._xyz = optimizable_tensors["xyz"]
    self._scaling = optimizable_tensors["scaling"]
    self._rotation = optimizable_tensors["rotation"]
    self._density = optimizable_tensors["density"]

    if self.use_student_t:
        self._nu = optimizable_tensors["nu"]
        self._opacity = optimizable_tensors["opacity"]

    self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
    self.denom = self.denom[valid_points_mask]
    self.max_radii2D = self.max_radii2D[valid_points_mask]
```

---

#### 3. 修改 `train.py` - 集成 Co-Pruning

**修改位置:** 第 450 行附近 (densification 区域)

**当前代码结构:**
```python
# 密化与剪枝 (Densification and Pruning)
if iteration < opt.densify_until_iter:
    # 更新梯度统计
    # 执行 densify_and_prune()
```

**修改后:**
```python
# 密化与剪枝
if iteration < opt.densify_until_iter:
    # 原有梯度统计逻辑
    for i in range(gaussiansN):
        GsDict[f"gs{i}"].add_densification_stats(...)

    # 每 opt.densification_interval 执行密化
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        for i in range(gaussiansN):
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            GsDict[f"gs{i}"].densify_and_prune(
                opt.densify_grad_threshold,
                0.005,
                scene.cameras_extent,
                size_threshold,
            )

        # 🌟 Co-Pruning: 每 5 次密化触发一次
        if args.enable_corgs_coprune and (iteration // opt.densification_interval) % 5 == 0:
            from r2_gaussian.utils.corgs_coprune import co_prune_gaussians

            prune_mask_0, prune_mask_1 = co_prune_gaussians(
                GsDict["gs0"], GsDict["gs1"], tau=args.corgs_tau
            )
            GsDict["gs0"].prune_points_with_mask(prune_mask_0)
            GsDict["gs1"].prune_points_with_mask(prune_mask_1)
```

---

### 超参数校准策略

**关键问题:** τ=5 针对 RGB 场景,R²-Gaussian 场景如何确定?

**分析依据:**
- R²-Gaussian `scale_bound=[0.0005, 0.5]` → Gaussian 最大半径 0.5
- 场景归一化到 [-1,1]³ → 对角线长度 √3 ≈ 1.73
- τ 应为 Gaussian 尺度的 0.2~1.0 倍

**推荐网格搜索:**
```python
tau_candidates = [0.1, 0.3, 0.5]  # 分别对应保守/中等/宽松
```

**自动校准方法 (可选):**
```python
def auto_calibrate_tau(gaussians, percentile=75):
    """基于 Gaussian 尺度分布自动确定阈值"""
    scales = gaussians.get_scaling.max(dim=1)[0]  # 最大轴长度
    tau = torch.quantile(scales, percentile/100.0).item()
    return tau
```

---

## 阶段 3: Pseudo-View Co-Regularization (3-5 天)

### 目标与验收标准

**目标:** 实现 CT 伪投影正则化,进一步提升 PSNR +0.5~1.0 dB

**验收标准:**
- [ ] 伪投影成功生成 (角度插值)
- [ ] PSNR 累计提升 +0.8~1.3 dB (在阶段 2 基础上)
- [ ] 边缘伪影减少 (视觉评估)

---

### 核心算法设计

#### 1. 新建模块: `r2_gaussian/utils/corgs_pseudo_view.py`

```python
import torch
import numpy as np
from r2_gaussian.dataset import Camera

def sample_pseudo_ct_angle(train_cameras, noise_std=2.0):
    """
    采样伪投影角度 (CT 角度插值策略)

    参数:
        train_cameras: 训练相机列表
        noise_std: 角度扰动标准差 (度)

    返回:
        pseudo_camera: Camera 实例
    """
    # 提取训练角度
    train_angles = [cam.get_projection_angle() for cam in train_cameras]  # 需添加方法
    train_angles = sorted(train_angles)

    # 随机选择相邻角度对
    i = np.random.randint(0, len(train_angles) - 1)
    theta_1, theta_2 = train_angles[i], train_angles[i+1]

    # 线性插值 + 噪声
    alpha = np.random.uniform(0.3, 0.7)  # 避免过于接近训练角度
    theta_pseudo = alpha * theta_1 + (1 - alpha) * theta_2
    theta_pseudo += np.random.normal(0, noise_std)  # ±2° 扰动

    # 创建伪相机 (继承 CT 几何参数)
    ref_camera = train_cameras[i]
    pseudo_camera = create_ct_camera_from_angle(
        theta_pseudo,
        SAD=ref_camera.SAD,
        SDD=ref_camera.SDD,
        detector_size=ref_camera.detector_size,
        image_height=ref_camera.image_height,
        image_width=ref_camera.image_width,
    )

    return pseudo_camera


def create_ct_camera_from_angle(theta, SAD, SDD, detector_size, image_height, image_width):
    """
    根据角度创建 CT 投影相机

    参数:
        theta: 投影角度 (度)
        SAD, SDD: 源/探测器距离
        detector_size: 探测器物理尺寸
        image_height, image_width: 图像分辨率

    返回:
        Camera 实例
    """
    # 将角度转换为相机位置和旋转
    theta_rad = np.deg2rad(theta)

    # CT 扫描器几何: 源绕 z 轴旋转
    source_x = -SAD * np.sin(theta_rad)
    source_y = SAD * np.cos(theta_rad)
    source_z = 0.0

    # 探测器中心位置
    detector_x = (SDD - SAD) * np.sin(theta_rad)
    detector_y = -(SDD - SAD) * np.cos(theta_rad)
    detector_z = 0.0

    # 构建相机 (需参考 R²-Gaussian Camera 类实现)
    # ⚠️ 此处需编程专家根据实际 Camera 类调整
    pseudo_camera = Camera(
        colmap_id=-1,  # 虚拟相机
        R=compute_rotation_matrix(theta_rad),
        T=np.array([source_x, source_y, source_z]),
        FoVx=compute_fov(detector_size[0], SAD),
        FoVy=compute_fov(detector_size[1], SAD),
        image=torch.zeros(3, image_height, image_width),  # 占位
        gt_alpha_mask=None,
        image_name=f"pseudo_angle_{theta:.2f}",
        uid=-1,
    )

    return pseudo_camera
```

**关键不确定点 (需编程专家调研):**
1. R²-Gaussian 是否已有 `get_projection_angle()` 方法?
2. Camera 类构造函数签名?
3. CT 几何参数如何存储 (SAD, SDD)?

---

#### 2. 修改 `train.py` - 添加伪投影损失

**修改位置:** 第 342 行 (协同正则化区域)

**当前代码 (有 bug):**
```python
# 原始错误版本: identity loss
LossDict[f"loss_gs{i}"] += dataset.multi_gaussian_weight * l1_loss(pseudo_image, pseudo_image.detach())
```

**修正为 CoR-GS 版本:**
```python
# 🌟 CoR-GS Pseudo-View Co-Regularization
if args.enable_corgs_coreg and gaussiansN > 1 and iteration > 2000:
    from r2_gaussian.utils.corgs_pseudo_view import sample_pseudo_ct_angle

    # 采样伪投影角度
    pseudo_cam = sample_pseudo_ct_angle(scene.getTrainCameras(), noise_std=2.0)

    # 双模型渲染伪投影
    pseudo_renders = []
    for i in range(gaussiansN):
        pseudo_pkg = render(pseudo_cam, GsDict[f'gs{i}'], pipe)
        pseudo_renders.append(pseudo_pkg["render"])

    # 计算伪投影协同正则化损失 (L1 + D-SSIM)
    lambda_local = 0.2  # L1 vs D-SSIM 平衡
    for i in range(gaussiansN):
        for j in range(i+1, gaussiansN):
            pseudo_l1 = l1_loss(pseudo_renders[i], pseudo_renders[j])
            pseudo_dssim = 1.0 - ssim(pseudo_renders[i], pseudo_renders[j])
            pseudo_coreg_loss = (1 - lambda_local) * pseudo_l1 + lambda_local * pseudo_dssim

            LossDict[f"loss_gs{i}"] += args.corgs_lambda_p * pseudo_coreg_loss
            LossDict[f"loss_gs{j}"] += args.corgs_lambda_p * pseudo_coreg_loss
```

---

### D-SSIM 损失检查

**问题:** R²-Gaussian 是否已实现 D-SSIM?

**检查代码:** `r2_gaussian/utils/loss_utils.py` Line 31

**确认:** ✅ 已有 `ssim()` 函数 → D-SSIM = 1 - SSIM

---

## 阶段 4: 完整集成与实验 (1 周)

### 集成清单

#### 向下兼容性策略

**Feature Flag 控制:**
```python
if args.enable_corgs:
    # CoR-GS 双模型训练
else:
    # 原始 R²-Gaussian 单模型训练
```

**配置文件示例:** `configs/corgs_foot_3views.yaml`
```yaml
enable_corgs: true
corgs_num_models: 2
corgs_tau: 0.3
enable_corgs_coprune: true
enable_corgs_coreg: true
corgs_lambda_p: 1.0
iterations: 20000
```

---

### 实验设计

**数据集:** foot 3 views

**对比方法:**

| 方法 | PSNR (预期) | SSIM (预期) | 训练时间 | Gaussian 点数 |
|------|-------------|-------------|----------|---------------|
| R²-Gaussian Baseline | 28.547 | 0.9008 | 2.5 min | 100% |
| + Co-Pruning only | 28.85 | 0.905 | 3.5 min | 75% |
| + Pseudo-view only | 29.35 | 0.913 | 5.0 min | 100% |
| + Full CoRGS | **29.75** | **0.918** | 6.0 min | 75% |

**超参数网格搜索:**
```python
tau_grid = [0.1, 0.3, 0.5]
lambda_p_grid = [0.5, 1.0, 2.0]
# 共 3x3=9 组实验
```

---

### 可视化需求

**1. 点云对比图:**
```python
# 使用 Open3D 可视化
# 对比: Baseline vs CoRGS
# 预期: CoRGS 点云更紧凑
```

**2. 差异热图:**
```python
# 绘制 Point Disagreement 空间分布
# 颜色: 高差异=红色,低差异=蓝色
```

**3. 伪影对比:**
```python
# CT 切片对比 (axial, coronal, sagittal)
# 标注伪影区域 (条纹伪影, 边缘模糊)
```

---

## 关键技术决策

### 1. KNN 库选择

**决策:** 使用 `simple_knn._C.distCUDA2` (现有依赖)

**理由:**
- R²-Gaussian 已集成 (见 `gaussian_model.py` Line 21)
- GPU 加速,效率高
- 接口简单 (输入 xyz,输出最近邻距离)

**备选方案:**
- PyTorch KNN: `torch_cluster.knn` (需额外安装)
- Open3D: `o3d.pipelines.registration` (CPU,较慢)

---

### 2. 阈值 τ 确定

**问题:** 原论文 τ=5 针对 [-1,1]³ 场景,R²-Gaussian 如何适配?

**分析:**
- R²-Gaussian `scale_bound=[0.0005, 0.5]` → Gaussian 半径 ≤ 0.5
- 场景尺度: 归一化到 [-1,1]³
- τ 应为 Gaussian 尺度的 0.2~1.0 倍

**推荐策略:**
```python
# 方法 1: 固定值网格搜索
tau = 0.3  # 保守估计

# 方法 2: 自适应 (基于尺度分布)
tau = torch.quantile(gaussians.get_scaling.max(dim=1)[0], 0.75)
```

**实验验证:** 在阶段 2 网格搜索 [0.1, 0.3, 0.5]

---

### 3. Co-Pruning 频率

**原论文:** 每 5 次 densification

**R²-Gaussian 适配:**
- Densification 间隔: `opt.densification_interval` (通常 100)
- Co-Pruning 触发: `iteration % (5 * opt.densification_interval) == 0`

**理由:** 避免过于频繁剪枝,给优化器时间修正位置

---

### 4. 双模型内存管理

**挑战:** 显存增加 1.5 倍 (3GB → 4.5GB)

**优化策略:**
1. **梯度检查点:** `torch.utils.checkpoint` (减少中间激活存储)
2. **混合精度:** `torch.cuda.amp` (FP16)
3. **分阶段训练:** 前 50% 迭代单模型,后 50% 双模型

**推荐:** 阶段 4 实施混合精度优化

---

## 代码集成策略

### 向下兼容保证

**方法 1: Try-Except 模式**
```python
try:
    from r2_gaussian.utils.corgs_metrics import compute_point_disagreement
    HAS_CORGS = True
except ImportError:
    HAS_CORGS = False

if HAS_CORGS and args.enable_corgs:
    # CoR-GS 逻辑
else:
    # 原始逻辑
```

**方法 2: Feature Flag**
```python
if args.enable_corgs:
    # CoR-GS 双模型
    gaussiansN = 2
else:
    # 单模型
    gaussiansN = 1
```

---

### Git 分支策略

**推荐工作流:**
```bash
# 创建功能分支
git checkout -b feature/corgs-implementation

# 阶段 1: 概念验证
git commit -m "feat: 阶段 1 - 双模型差异验证"

# 阶段 2: Co-Pruning
git commit -m "feat: 阶段 2 - 协同剪枝实现"

# 阶段 3: Pseudo-view
git commit -m "feat: 阶段 3 - 伪投影正则化"

# 阶段 4: 完整集成
git commit -m "feat: 阶段 4 - CoR-GS 完整系统"

# 合并到 main
git checkout main
git merge feature/corgs-implementation
```

---

## 技术风险与缓解

### 风险 1: KNN 匹配在 CT 投影下失效

**表现:** Co-Pruning 后 PSNR 下降

**缓解:**
1. 调整 τ 到更宽松值 (0.5 → 1.0)
2. 添加投影一致性检查 (见医学专家方案 B)
3. 回退到仅使用 Pseudo-view Co-reg

---

### 风险 2: 伪投影采样覆盖不足

**表现:** 伪投影正则化无效 (PSNR 无提升)

**缓解:**
1. 增加采样范围 (α ∈ [0.1, 0.9] 而非 [0.3, 0.7])
2. 增加噪声标准差 (2° → 5°)
3. 每轮采样多个伪投影 (3-5 个)

---

### 风险 3: 训练时间超限

**表现:** 训练时间 >10 分钟 (超过 baseline 4 倍)

**缓解:**
1. 启用混合精度训练 (FP16)
2. 减少伪投影采样频率
3. 早停策略 (监测 Point Disagreement 饱和)

---

## 时间预算与资源需求

### 开发时间估算

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 阶段 1 | 双模型框架 + Disagreement 计算 | 1-2 天 |
| 阶段 2 | Co-Pruning 实现 + 超参数调优 | 3-5 天 |
| 阶段 3 | Pseudo-view 采样 + 损失函数 | 3-5 天 |
| 阶段 4 | 完整集成 + 消融实验 | 5-7 天 |
| **总计** | | **12-19 天** |

---

### GPU 资源需求

**训练阶段:**
- 显存: 4.5GB (双模型) vs 3GB (baseline)
- 时间: 6 分钟/10k 迭代 (vs 2.5 分钟 baseline)

**实验阶段:**
- 超参数搜索: 9 组实验 × 6 分钟 = 54 分钟
- 消融实验: 4 组 × 6 分钟 = 24 分钟

---

## 需要编程专家调研的问题

### 1. Camera 类实现细节

**问题:** R²-Gaussian 的 Camera 类构造函数签名?

**位置:** `r2_gaussian/dataset/__init__.py` 或 `scene/cameras.py`

**需要信息:**
- 如何从角度创建 CT 投影相机?
- SAD, SDD 参数如何传递?
- 是否已有类似 `create_pseudo_camera()` 方法?

---

### 2. 现有伪视图实现

**问题:** `train.py` Line 352-365 的 `pseudo_cameras` 如何生成?

**需要调研:**
- `scene.getPseudoCamerasWithClosestViews()` 实现逻辑
- 是否可复用现有伪相机生成代码?

---

### 3. Densification 触发机制

**问题:** R²-Gaussian 的密化频率和参数?

**需要确认:**
- `opt.densification_interval` 默认值?
- `opt.densify_from_iter`, `opt.densify_until_iter` 范围?
- 是否有 drop 机制影响密化?

---

## 验证检查清单

**阶段 1 完成标准:**
- [ ] 双模型训练成功 (无崩溃)
- [ ] TensorBoard 记录 Point/Rendering Disagreement
- [ ] 相关性分析脚本生成散点图
- [ ] Pearson 系数 < -0.3

**阶段 2 完成标准:**
- [ ] Co-Pruning 成功触发 (日志显示剪枝数量)
- [ ] Gaussian 点数减少 >20%
- [ ] PSNR 提升 +0.3~0.5 dB
- [ ] 可视化点云紧凑度改善

**阶段 3 完成标准:**
- [ ] 伪投影成功渲染 (无几何错误)
- [ ] PSNR 累计提升 +0.8~1.3 dB
- [ ] 伪影视觉减少 (CT 切片对比)

**阶段 4 完成标准:**
- [ ] 超参数网格搜索完成
- [ ] 消融实验结果符合预期
- [ ] 代码向下兼容 (baseline 可正常运行)
- [ ] 文档完整 (README, 配置示例)

---

## 您需要决策的问题

### 决策点 1: 是否批准技术路线?

**当前方案:** 欧氏 KNN + 角度插值 (快速验证)

**替代方案:** 投影域匹配 (医学专家推荐,但开发时间 +5 天)

**您的选择:** [ ] 批准当前方案 / [ ] 切换到投影域匹配

---

### 决策点 2: 超参数调优范围

**推荐:** τ ∈ [0.1, 0.3, 0.5], λ_p ∈ [0.5, 1.0, 2.0]

**替代:** 更密集搜索 (τ 5 个值, λ_p 5 个值 → 25 组实验)

**您的选择:** [ ] 使用推荐范围 / [ ] 扩大搜索空间

---

### 决策点 3: 实验优先级

**选项 A:** 先快速验证效果,后优化训练时间

**选项 B:** 同步优化,但开发时间 +3 天

**您的选择:** [ ] 选项 A (推荐) / [ ] 选项 B

---

**文档完成时间:** 2025-11-16 15:20
**下一步:** 等待用户批准后,交付编程专家实现阶段 1
