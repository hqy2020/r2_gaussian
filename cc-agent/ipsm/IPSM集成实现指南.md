# IPSM集成到R²-Gaussian实现指南

> **状态**: 核心模块已完成 ✓
> **待完成**: render()深度输出 → 参数配置 → train.py集成 → 验证

---

## 📋 已完成模块

### ✅ 1. 深度估计器 (`r2_gaussian/utils/depth_estimator.py`)
- **功能**: DPT单目深度估计
- **特性**:
  - 支持CT灰度图像→RGB转换
  - 全局单例模式避免重复加载
  - 占位符模式（如果DPT加载失败）
- **使用**:
  ```python
  from r2_gaussian.utils.depth_estimator import get_depth_estimator
  depth_est = get_depth_estimator()
  depth_map = depth_est.estimate(ct_image)  # (H, W)
  ```

### ✅ 2. 扩散模型封装 (`r2_gaussian/utils/diffusion_utils.py`)
- **功能**: SD Inpainting延迟加载
- **特性**:
  - 动态加载/卸载（节省显存）
  - FP16推理
  - IPSM双阶段score matching
- **使用**:
  ```python
  from r2_gaussian.utils.diffusion_utils import DiffusionGuidance, ct_to_rgb

  diffusion = DiffusionGuidance()
  diffusion.load_model()  # iter 2K时调用

  loss_ipsm = diffusion.compute_ipsm_loss(
      x_0=ct_to_rgb(rendered_img),
      I_warped=ct_to_rgb(warped_img),
      mask=consistency_mask,
      eta_r=0.1
  )

  diffusion.unload_model()  # iter 9.5K时调用
  ```

### ✅ 3. 损失函数 (`r2_gaussian/utils/loss_utils.py`)
- **新增函数**:
  - `pearson_correlation_loss()`: Pearson深度正则化
  - `geometry_consistency_loss()`: Masked L1 loss
  - `ipsm_depth_regularization()`: 组合seen/unseen深度loss

### ✅ 4. X-ray Warping (`r2_gaussian/utils/ipsm_utils.py`)
- **功能**: 体素反投影warping
- **类**: `XRayIPSMWarping`
- **核心方法**:
  - `warp_via_voxel_projection()`: 主warping函数
  - `sample_nearby_viewpoint()`: 采样伪视角
- **使用**:
  ```python
  from r2_gaussian.utils.ipsm_utils import XRayIPSMWarping, sample_nearby_viewpoint

  ipsm_warp = XRayIPSMWarping(scanner_cfg, pipe)
  pseudo_cam = sample_nearby_viewpoint(base_cam, angle_range=15.0)

  warped_img, mask = ipsm_warp.warp_via_voxel_projection(
      source_image=gt_image,
      source_cam=base_cam,
      target_cam=pseudo_cam,
      target_depth=pseudo_depth,
      tau=0.3
  )
  ```

---

## 🔧 待实现步骤

### 步骤1: 修改render()输出深度

**文件**: `r2_gaussian/gaussian/render_query.py`

**目标**: 在render()返回字典中添加`"depth"`键

**实现方案**:

```python
def render(...):
    # ... 现有代码 ...

    # 在rasterizer调用后添加深度渲染
    rendered_image, radii = rasterizer(...)

    # === 新增: 渲染深度图 ===
    # 方法A: 使用Z-buffer深度
    depth_map, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=means3D[:, 2:3],  # 使用Z坐标作为"颜色"
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # 方法B: 如果rasterizer支持直接深度输出
    # 需要检查submodules/xray-gaussian-rasterization-voxelization
    # 是否有depth output选项

    return {
        "render": rendered_image,
        "depth": depth_map,  # 新增
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
```

**验证**:
```python
render_pkg = render(viewpoint_cam, gaussians, pipe)
assert "depth" in render_pkg
print(f"Depth shape: {render_pkg['depth'].shape}")  # (H, W)
```

**注意**:
- 需要检查CUDA rasterizer是否支持深度输出
- 如果不支持，可以先用占位符（zeros）快速验证框架

---

### 步骤2: 添加命令行参数

**文件**: `r2_gaussian/arguments/__init__.py`

**新增类** `IPSMParams`:

```python
class IPSMParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.enable_ipsm = False
        self.ipsm_start_iter = 2000
        self.ipsm_end_iter = 9500
        self.lambda_ipsm = 1.0  # 降低（考虑domain gap）
        self.lambda_ipsm_depth = 0.5
        self.lambda_ipsm_geo = 4.0  # 提高（增强几何约束）
        self.ipsm_eta_r = 0.1
        self.ipsm_eta_d = 0.1
        self.ipsm_mask_tau = 0.3
        self.ipsm_mask_tau_geo = 0.1
        self.ipsm_cfg_scale = 7.5
        self.ipsm_pseudo_angle_range = 15.0
        self.sd_model_path = "stabilityai/stable-diffusion-2-inpainting"
        super().__init__(parser, "IPSM Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        return g
```

**在`__init__.py`中注册**:
```python
def get_combined_args(parser: ArgumentParser):
    ...
    ipsm_params = IPSMParams(parser)  # 新增
    args = parser.parse_args(sys.argv[1:])
    ...
    return (
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        ipsm_params.extract(args)  # 新增
    )
```

**使用示例**:
```bash
python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/foot_3_ipsm \
    --enable_ipsm \
    --lambda_ipsm 1.0 \
    --lambda_ipsm_geo 4.0 \
    --iterations 30000
```

---

### 步骤3: 集成到train.py

**文件**: `train.py`

**主要修改点**:

#### 3.1 初始化IPSM组件

```python
def training(dataset, opt, pipe, ipsm, tb_writer, ...):  # 新增ipsm参数
    ...

    # === IPSM初始化 ===
    ipsm_warp = None
    diffusion_guide = None
    depth_estimator = None

    if ipsm.enable_ipsm:
        from r2_gaussian.utils.ipsm_utils import XRayIPSMWarping
        from r2_gaussian.utils.diffusion_utils import DiffusionGuidance
        from r2_gaussian.utils.depth_estimator import get_depth_estimator
        from r2_gaussian.utils.loss_utils import (
            ipsm_depth_regularization,
            geometry_consistency_loss
        )

        ipsm_warp = XRayIPSMWarping(scanner_cfg, pipe)
        diffusion_guide = DiffusionGuidance(ipsm.sd_model_path)
        depth_estimator = get_depth_estimator()

        print(f"✓ IPSM enabled: iter {ipsm.ipsm_start_iter}-{ipsm.ipsm_end_iter}")
        print(f"  λ_IPSM={ipsm.lambda_ipsm}, λ_depth={ipsm.lambda_ipsm_depth}, λ_geo={ipsm.lambda_ipsm_geo}")
```

#### 3.2 训练循环中集成

```python
for iteration in range(first_iter, opt.iterations + 1):
    # === 动态加载扩散模型 ===
    if ipsm.enable_ipsm and iteration == ipsm.ipsm_start_iter:
        print(f"[ITER {iteration}] Loading diffusion model...")
        diffusion_guide.load_model()

    # ... 原有渲染 ...
    render_pkg = render(viewpoint_cam, gaussians, pipe)
    image = render_pkg["render"]
    depth_seen = render_pkg["depth"]  # 新增

    # === IPSM guidance ===
    if ipsm.enable_ipsm and ipsm.ipsm_start_iter <= iteration < ipsm.ipsm_end_iter:
        # 1. 采样伪视角
        from r2_gaussian.utils.ipsm_utils import sample_nearby_viewpoint
        pseudo_cam = sample_nearby_viewpoint(
            viewpoint_cam,
            angle_range=ipsm.ipsm_pseudo_angle_range
        )

        # 2. 渲染伪视角
        pseudo_pkg = render(pseudo_cam, gaussians, pipe)
        x_0_j = pseudo_pkg["render"]
        depth_unseen = pseudo_pkg["depth"]

        # 3. Inverse warping
        I_warped, mask_warp = ipsm_warp.warp_via_voxel_projection(
            gt_image,
            viewpoint_cam,
            pseudo_cam,
            depth_unseen,
            tau=ipsm.ipsm_mask_tau
        )

        # 4. 深度正则化
        depth_mono_seen = depth_estimator.estimate(gt_image)
        depth_mono_unseen = depth_estimator.estimate(x_0_j)

        loss_ipsm_depth = ipsm_depth_regularization(
            depth_seen, depth_mono_seen,
            depth_unseen, depth_mono_unseen,
            eta_d=ipsm.ipsm_eta_d
        )
        loss["ipsm_depth"] = loss_ipsm_depth
        loss["total"] += ipsm.lambda_ipsm_depth * loss_ipsm_depth

        # 5. 几何一致性（更严格mask）
        _, mask_geo = ipsm_warp.warp_via_voxel_projection(
            gt_image, viewpoint_cam, pseudo_cam, depth_unseen,
            tau=ipsm.ipsm_mask_tau_geo
        )
        loss_geo = geometry_consistency_loss(x_0_j, I_warped, mask_geo)
        loss["ipsm_geo"] = loss_geo
        loss["total"] += ipsm.lambda_ipsm_geo * loss_geo

        # 6. Score distillation
        from r2_gaussian.utils.diffusion_utils import ct_to_rgb
        loss_ipsm_sd = diffusion_guide.compute_ipsm_loss(
            ct_to_rgb(x_0_j),
            ct_to_rgb(I_warped),
            mask_warp,
            eta_r=ipsm.ipsm_eta_r,
            cfg_scale=ipsm.ipsm_cfg_scale
        )
        loss["ipsm_sd"] = loss_ipsm_sd
        loss["total"] += ipsm.lambda_ipsm * loss_ipsm_sd

    # === 卸载扩散模型 ===
    if ipsm.enable_ipsm and iteration == ipsm.ipsm_end_iter:
        print(f"[ITER {iteration}] Unloading diffusion model...")
        diffusion_guide.unload_model()

    # ... 原有backward和优化 ...
```

#### 3.3 主函数调用修改

```python
# train.py 底部
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    ipsm_p = IPSMParams(parser)  # 新增

    args = parser.parse_args(sys.argv[1:])

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        ipsm_p.extract(args),  # 新增
        tb_writer,
        ...
    )
```

---

## 🧪 验证流程

### 阶段0: 代码验证 (500迭代)

**目的**: 确认IPSM代码可运行，无crash

**命令**:
```bash
python train.py \
    -s /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle \
    -m output/ipsm_test_500 \
    --enable_ipsm \
    --iterations 500 \
    --ipsm_start_iter 100 \
    --ipsm_end_iter 400 \
    --lambda_ipsm 0.1  # 降低避免影响过大
```

**检查点**:
- [ ] 程序正常启动
- [ ] iter 100成功加载扩散模型
- [ ] IPSM loss正常计算（不是NaN/Inf）
- [ ] iter 400成功卸载扩散模型
- [ ] Total loss正常下降
- [ ] 无CUDA OOM错误

**预期输出**:
```
[ITER 100] Loading diffusion model...
✓ 扩散模型加载成功
[ITER 101] loss: 1.5e-01, pts: 1.2e+05, ipsm_depth: 0.45, ipsm_geo: 0.12, ipsm_sd: 0.08
...
[ITER 400] Unloading diffusion model...
✓ 扩散模型已卸载，显存已释放
```

---

### 阶段1: 完整训练 (30K迭代)

**目的**: 与baseline对比，验证IPSM效果

**命令**:
```bash
python train.py \
    -s /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle \
    -m output/$(date +%Y_%m_%d_%H_%M)_foot_3views_ipsm \
    --gaussiansN 1 \
    --enable_ipsm \
    --lambda_ipsm 1.0 \
    --lambda_ipsm_depth 0.5 \
    --lambda_ipsm_geo 4.0 \
    --ipsm_eta_r 0.1 \
    --ipsm_eta_d 0.1 \
    --ipsm_mask_tau 0.3 \
    --ipsm_mask_tau_geo 0.1 \
    --ipsm_cfg_scale 7.5 \
    --ipsm_start_iter 2000 \
    --ipsm_end_iter 9500 \
    --iterations 30000
```

**评估**:
```bash
python test.py -m output/YYYY_MM_DD_HH_MM_foot_3views_ipsm
```

**成功标准**:
- PSNR > 28.4873 (baseline)
- SSIM > 0.9005 (baseline)

---

## 📊 预期结果

### Baseline (R²-Gaussian, Foot-3)
- PSNR: 28.4873
- SSIM: 0.9005

### 目标 (R²-Gaussian + IPSM)
- PSNR: > 28.5 (+0.5%)
- SSIM: > 0.901 (+0.05%)

**保守估计**: 由于CT domain gap，提升可能小于IPSM在LLFF上的表现（+7.2% SSIM）

---

## ⚠️ 已知风险和缓解措施

| 风险 | 缓解措施 | 状态 |
|------|---------|------|
| **render()不支持深度** | 先用占位符验证框架，后修改rasterizer | 🔍 待检查 |
| **SD对CT效果差** | 已降低λ_IPSM (2.0→1.0)，提高λ_geo (2.0→4.0) | ✅ 已处理 |
| **X-ray warping不准** | 体素反投影方案，物理准确 | ✅ 已实现 |
| **显存不足** | FP16推理 + 动态加载/卸载 + batch=1 | ✅ 已处理 |
| **DPT对CT深度不准** | 框架优先，后续可替换Depth Anything | ⏸️ 待验证 |

---

## 🚀 下一步行动

### 立即执行:
1. **检查render()深度支持**
   ```bash
   # 检查rasterizer源码
   cat r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/README.md
   # 或测试占位符方案
   ```

2. **添加命令行参数**
   - 修改`r2_gaussian/arguments/__init__.py`
   - 添加`IPSMParams`类

3. **集成到train.py**
   - 遵循上述步骤3的代码模板

### 验证顺序:
```
Step 1: 500迭代快速验证 (5-10分钟)
   ↓
Step 2: 检查loss曲线，确认无异常
   ↓
Step 3: 完整30K训练 (约1-2小时)
   ↓
Step 4: test.py评估，对比baseline
```

---

## 📝 关键代码片段速查

### 快速启用IPSM
```python
if ipsm.enable_ipsm and ipsm_start <= iteration < ipsm_end:
    # 伪视角采样
    pseudo_cam = sample_nearby_viewpoint(base_cam)

    # 渲染+warping
    pseudo_pkg = render(pseudo_cam, gaussians, pipe)
    warped, mask = ipsm_warp.warp_via_voxel_projection(...)

    # 损失计算
    loss_depth = ipsm_depth_regularization(...)
    loss_geo = geometry_consistency_loss(...)
    loss_sd = diffusion_guide.compute_ipsm_loss(...)

    # 累加到total loss
    loss["total"] += λ_depth * loss_depth + λ_geo * loss_geo + λ_sd * loss_sd
```

### TensorBoard监控
```python
# 在logging部分添加
if ipsm.enable_ipsm:
    tb_writer.add_scalar('ipsm/depth_loss', loss["ipsm_depth"], iteration)
    tb_writer.add_scalar('ipsm/geo_loss', loss["ipsm_geo"], iteration)
    tb_writer.add_scalar('ipsm/sd_loss', loss["ipsm_sd"], iteration)
```

---

## ✅ 检查清单

实施前确认:
- [ ] 已安装依赖: `diffusers`, `transformers`, `torch.hub (MiDaS)`
- [ ] 已阅读`innovation_migration_guide.md`
- [ ] 已理解IPSM核心原理
- [ ] 已备份原始`train.py`

实施后确认:
- [ ] `render()`返回包含`"depth"`键
- [ ] 命令行`--enable_ipsm`被识别
- [ ] 500迭代测试通过
- [ ] 扩散模型正确加载/卸载
- [ ] TensorBoard显示IPSM相关loss曲线

---

**文档版本**: v1.0
**创建时间**: 2025-11-20
**作者**: Claude (R²-Gaussian科研助手系统)
