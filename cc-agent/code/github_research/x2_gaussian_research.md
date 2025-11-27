# X²-Gaussian GitHub 仓库调研报告

**调研日期**: 2025-11-18
**仓库地址**: https://github.com/yuyouxixi/x2-gaussian
**论文**: [ICCV 2025] X²-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction
**调研者**: PyTorch/CUDA 编程专家

---

## 【核心结论】

X²-Gaussian 实现了连续时间 CT 重建，核心技术包括 K-Planes 空间分解（6 个平面的多分辨率特征网格）、多头解码器（解耦位置/旋转/缩放优化）、TV 正则化（5 种类型）以及两阶段渐进式训练策略。代码库包含 55.9% Python + 34.3% CUDA，依赖 TIGRE 和两个自定义 CUDA 子模块。**关键可移植模块**：`hexplane.py`（K-Planes 实现）、`deformation.py`（多头解码器）、`regulation.py`（正则化）。**主要改造点**：需将时间维度改为视角嵌入，移除动态场景相关代码，CUDA 子模块需评估兼容性。迁移建议采用"渐进式集成策略"，先复用纯 Python 模块，再逐步适配 CUDA 加速。

---

## 【详细分析】

### 1. 仓库结构分析

#### 1.1 主要目录结构

```
x2-gaussian/
├── x2_gaussian/              # 核心代码包
│   ├── arguments/            # 参数管理
│   ├── dataset/              # 数据集加载
│   ├── gaussian/             # 高斯模型核心 ⭐
│   │   ├── gaussian_model.py    # 主模型类
│   │   ├── hexplane.py          # K-Planes 实现
│   │   ├── deformation.py       # 多头解码器
│   │   ├── regulation.py        # 正则化
│   │   ├── render_query.py      # 渲染管线
│   │   ├── initialize.py        # 初始化
│   │   └── grid.py              # 网格工具
│   ├── submodules/           # CUDA 子模块
│   │   ├── simple-knn/          # GPU 加速 KNN
│   │   └── xray-gaussian-rasterization-voxelization/  # X-ray 光栅化
│   └── utils/                # 工具函数
├── train.py                  # 训练主程序 ⭐
├── initialize_pcd.py         # 点云初始化
└── requirements.txt          # 依赖清单
```

**⭐ 标记为关键文件**

#### 1.2 代码组成

- **Python**: 55.9%（主要逻辑）
- **CUDA**: 34.3%（性能关键部分）
- **C++**: 8.9%（接口层）

#### 1.3 CUDA 扩展

**子模块 1**: `simple-knn`
- 功能：GPU 加速的 K 近邻搜索
- 用途：密度自适应控制（查找邻近高斯）
- 兼容性：可能与 R²-Gaussian 现有 KNN 模块冲突

**子模块 2**: `xray-gaussian-rasterization-voxelization`
- 功能：X-ray 光线投射 + 体素化渲染
- 用途：核心渲染管线（替代标准 3DGS 光栅化）
- 兼容性：R²-Gaussian 使用同名子模块，需检查版本差异

---

### 2. 关键实现细节

#### 2.1 K-Planes 空间分解（`hexplane.py`）

**文件路径**: `x2_gaussian/gaussian/hexplane.py`

**核心类**: `HexPlaneField`

**初始化逻辑**（约第 50-80 行）:
```python
# 生成平面索引组合（4D 输入 → 6 个平面）
self.plane_coef = itertools.combinations(range(4), 2)
# 结果: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
# 对应: [xy, xz, xt, yz, yt, zt]（空间+时间）
```

**多分辨率设计**（约第 100-130 行）:
```python
for i, res in enumerate(multiscale_res_multipliers):
    # 空间维度按倍数缩放
    resolution = [int(base_res * res), int(base_res * res), int(base_res * res), time_res]
    # 时间维度保持不变
    self.grids.append(init_grid_param(out_dim, resolution))
```

**特征编码**（约第 200-250 行）:
```python
def interpolate_ms_features(pts, time):
    # 对每个平面执行 grid_sample
    plane_features = grid_sample_wrapper(grid, normalized_coords)
    # 多平面特征融合（逐元素乘法）
    fused = plane1_feat * plane2_feat * ...
    # 多分辨率特征拼接
    return torch.cat([level1_feat, level2_feat, ...], dim=-1)
```

**关键参数**:
- `kplanes_config['resolution']`: 基础分辨率（例如 [64, 64, 64, 25]）
- `multiscale_res_multipliers`: 多尺度倍数（例如 [1, 2, 4]）
- `concat_features`: True（拼接）或 False（求和）

**可移植性评估**:
- ✅ **纯 Python 实现**，无 CUDA 依赖
- ✅ **模块化设计**，可独立使用
- ⚠️ **需改造点**: 时间维度 → 视角嵌入（维度 3 → 视角索引）
- ⚠️ **数据格式**: 需确保输入归一化到 [-1, 1]³ 空间

---

#### 2.2 多头解码器（`deformation.py`）

**文件路径**: `x2_gaussian/gaussian/deformation.py`

**核心类**: `deform_network`

**网络架构**（约第 80-150 行）:
```python
# 共享骨干网络
self.feature_out = nn.Sequential(
    nn.Linear(input_dim, W), nn.ReLU(),
    nn.Linear(W, W), nn.ReLU(),
    ...,  # 深度 D 层
)

# 多头分支
self.pos_deform = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
self.scales_deform = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
self.rotations_deform = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
```

**时间嵌入**（约第 50-70 行）:
```python
# 位置编码（周期性正弦-余弦变换）
def poc_fre(input_data, poc_buf):
    # input_data: [N, 1] 时间戳
    # 返回: [N, 2*poc_buf] 高维特征
    return torch.cat([torch.sin(2**i * π * input_data) for i in range(poc_buf)] +
                     [torch.cos(2**i * π * input_data) for i in range(poc_buf)])

# 时间网络
self.timenet = nn.Sequential(
    nn.Linear(time_embed_dim, W//2), nn.ReLU(),
    nn.Linear(W//2, W), nn.ReLU()
)
```

**前向传播**（约第 200-250 行）:
```python
def forward(rays_pts_emb, time):
    # 1. 获取 K-Planes 特征
    grid_feature = self.grid(pts, time)

    # 2. 时间嵌入
    time_emb = poc_fre(time, self.timenet.in_features)
    time_feature = self.timenet(time_emb)

    # 3. 拼接特征
    hidden = torch.cat([grid_feature, time_feature], dim=-1)

    # 4. 共享编码
    hidden = self.feature_out(hidden)

    # 5. 多头解码
    dx = self.pos_deform(hidden) if not self.args.no_dx else 0
    ds = self.scales_deform(hidden) if not self.args.no_ds else 0
    dr = self.rotations_deform(hidden) if not self.args.no_dr else 0

    return pts + dx, scales + ds, rotations * dr  # 简化表示
```

**解耦优化机制**:
- `no_dx/no_ds/no_dr` 标志可独立禁用某个分支
- 掩膜加权合并（`mask`）控制变形幅度
- `apply_rotation` 决定旋转使用加法或四元数乘法

**可移植性评估**:
- ✅ **标准 PyTorch MLP**，易于迁移
- ⚠️ **时间嵌入需改为视角嵌入**: 修改 `poc_fre` 输入从时间戳 → 视角索引
- ⚠️ **动态变形可选**: R²-Gaussian 场景为静态，可简化或保留作为扩展能力

---

#### 2.3 TV 正则化（`regulation.py`）

**文件路径**: `x2_gaussian/gaussian/regulation.py`

**核心实现**（约第 20-50 行）:
```python
class PlaneTV(nn.Module):
    def compute_plane_tv(self, t):
        # t: [C, H, W] 平面参数
        h, w = t.shape[-2:]
        count_h = (h - 1) * w
        count_w = h * (w - 1)

        # 水平方向 TV
        h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
        # 垂直方向 TV
        w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()

        return 2 * (h_tv / count_h + w_tv / count_w)

    def forward(self, grid_params):
        total = 0
        for resolution_grids in grid_params:
            for plane_idx, plane in enumerate(resolution_grids):
                # 仅对空间平面计算（跳过时间平面）
                if plane_idx in [0, 1, 2]:  # xy, xz, yz 平面
                    total += self.compute_plane_tv(plane)
        return self.weight * total
```

**5 种正则化类型**（约第 100-300 行）:

| 类名 | 作用目标 | 损失公式 | 权重参数 |
|------|---------|---------|---------|
| `PlaneTV` | K-Planes 空间平滑 | ∑(∇H² + ∇W²) | `plane_tv_weight` |
| `TimeSmoothness` | 时间维度平滑 | ∑(∂²t/∂t²) | `time_smoothness_weight` |
| `L1TimePlanes` | 时间平面稀疏化 | ∑\|time_plane\| | `l1_time_planes` |
| `L1ProposalNetwork` | 提议网络稀疏 | ∑\|proposal_weights\| | - |
| `DepthTV` | 深度图平滑 | ∑(∇depth²) | - |

**可移植性评估**:
- ✅ **纯 PyTorch 实现**，直接可用
- ✅ **PlaneTV 最关键**，建议优先迁移
- ⚠️ **TimeSmoothness 需改造**: 时间维度 → 视角维度（或移除）
- ⚠️ **权重需调优**: 医学 CT 场景可能需要不同权重平衡

---

#### 2.4 训练策略（`train.py`）

**文件路径**: `train.py`

**两阶段训练**（约第 150-200 行）:
```python
# Stage 1: Coarse
coarse_iter = args.coarse_iter  # 默认 5000
for iteration in range(1, coarse_iter + 1):
    stage = 'coarse'
    # 基础渲染损失 + SSIM
    loss = l1_loss + lambda_dssim * ssim_loss

# Stage 2: Fine
for iteration in range(coarse_iter + 1, args.iterations + 1):
    stage = 'fine'
    # 完整损失函数
    loss = l1_loss + lambda_dssim * ssim_loss
    if iteration > 7000:
        loss += lambda_prior * prior_loss  # 引入先验
    loss += lambda_tv * tv_loss  # 3D TV
    loss += time_smoothness_weight * time_tv_loss  # 4D TV
```

**学习率调度**（约第 100-120 行）:
```python
# 指数衰减
def get_expon_lr_func(lr_init, lr_final, max_steps):
    return lambda step: lr_init * (lr_final / lr_init) ** (step / max_steps)

# 每次迭代更新
gaussians.update_learning_rate(iteration)
```

**密度自适应控制**（约第 250-300 行）:
```python
if iteration >= densify_from_iter and iteration <= densify_until_iter:
    # 每 densification_interval 迭代执行一次
    if iteration % densification_interval == 0:
        gaussians.densify_and_prune(
            max_grad=0.0002,         # 梯度阈值
            min_density=0.01,        # 密度阈值
            extent=cameras_extent,
            max_screen_size=20       # 屏幕尺寸阈值
        )

    # 每 density_reset_interval 迭代重置密度
    if iteration % density_reset_interval == 0:
        gaussians.reset_density()
```

**关键超参数**（默认值）:
- `coarse_iter`: 5000
- `iterations`: 30000
- `densify_from_iter`: 500
- `densify_until_iter`: 15000
- `densification_interval`: 100
- `density_reset_interval`: 3000
- `lambda_dssim`: 0.2
- `lambda_prior`: 0.01（仅 fine 阶段后期）
- `lambda_tv`: 0.001
- `time_smoothness_weight`: 0.01

**可移植性评估**:
- ✅ **训练流程清晰**，易于理解和复现
- ⚠️ **两阶段策略需评估**: R²-Gaussian 是否需要 coarse/fine 分阶段？
- ⚠️ **先验损失需替换**: X2-GS 的先验是基于动态场景，静态 CT 需重新设计
- ✅ **密度控制可直接复用**: 逻辑与 R²-Gaussian baseline 类似

---

### 3. GaussianModel 核心类分析

**文件路径**: `x2_gaussian/gaussian/gaussian_model.py`

**主要属性**（约第 50-100 行）:
```python
class GaussianModel:
    def __init__(self, args):
        # 标准 3DGS 参数
        self._xyz = None              # 位置 [N, 3]
        self._scaling = None          # 缩放 [N, 3]
        self._rotation = None         # 旋转 [N, 4]（四元数）
        self._density = None          # 密度 [N, 1]

        # X2-GS 特有
        self._deformation = deform_network(args)  # 变形网络
        self.period = nn.Parameter(torch.log(torch.tensor(2.8)))  # 周期参数
        self.t_seq = torch.linspace(0, args.kplanes_config['resolution'][3]-1, ...)
        self._deformation_table = None  # 变形激活表
        self._deformation_accum = None  # 变形累积量
```

**与 R²-Gaussian 的主要差异**:

| 功能模块 | R²-Gaussian | X²-Gaussian | 差异说明 |
|---------|------------|------------|---------|
| 核心参数 | xyz, color, opacity, scale, rotation | xyz, density, scale, rotation | X2 用 density 替代 opacity/color |
| 变形网络 | ❌ 无 | ✅ K-Planes + MLP | X2 支持动态变形 |
| 周期参数 | ❌ 无 | ✅ period (可学习) | X2 用于周期性运动 |
| 缩放激活 | `exp()` | `sigmoid()` (可选) | X2 支持缩放边界约束 |
| 密度重置 | ❌ 无 | ✅ reset_density() | X2 定期重置避免退化 |
| 变形表 | ❌ 无 | ✅ deformation_table | X2 动态决定哪些高斯需变形 |

**学习率调度**（约第 200-250 行）:
```python
def update_learning_rate(self, iteration):
    # 为 7 个参数组分别更新学习率
    for param_group in self.optimizer.param_groups:
        if param_group['name'] == 'xyz':
            lr = self.xyz_scheduler(iteration)
        elif param_group['name'] == 'grid':
            lr = self.grid_scheduler(iteration)
        # ... 其他参数组
        param_group['lr'] = lr
```

**可移植性评估**:
- ⚠️ **需大幅改造**: R²-Gaussian 使用 color/opacity，X2 使用 density
- ✅ **变形网络可选**: 可作为扩展模块保留（即使静态场景也能用于视角自适应）
- ⚠️ **周期参数可移除**: CT 场景无周期性需求
- ✅ **密度重置逻辑值得借鉴**: 可能提升训练稳定性

---

### 4. 依赖库清单

| 库名 | 版本 | 用途 | R²-Gaussian 兼容性 |
|------|------|------|------------------|
| torch | 2.1.2+cu118 | 深度学习框架 | ⚠️ 版本需统一 |
| numpy | 1.24.4 | 数值计算 | ✅ 兼容 |
| TIGRE | 2.3 | CT 数据生成与初始化 | ⚠️ R²-Gaussian 可能未使用 |
| matplotlib | 3.7.5 | 可视化 | ✅ 兼容 |
| tensorboardX | 2.6.2.2 | 训练日志 | ✅ 兼容 |
| plyfile | 1.0.3 | PLY 文件读写 | ✅ 兼容 |
| open3d | 0.18.0 | 点云处理 | ✅ 兼容 |
| SimpleITK | 2.4.0 | 医学图像处理 | ✅ 兼容 |
| pydicom | - | DICOM 格式支持 | ✅ 兼容 |
| scikit-image | 0.21.0 | 图像处理 | ✅ 兼容 |

**CUDA 子模块**:
- `simple-knn`: GPU 加速 KNN（需检查是否与 R²-Gaussian 冲突）
- `xray-gaussian-rasterization-voxelization`: 核心渲染器（**需重点评估版本兼容性**）

---

### 5. 可移植性分析与实现建议

#### 5.1 可直接复用的模块

| 模块 | 文件 | 复用方式 | 优先级 |
|------|------|---------|-------|
| K-Planes | `hexplane.py` | 复制到 `r2_gaussian/utils/` | 🔥 高 |
| TV 正则化 | `regulation.py` | 复制 `PlaneTV` 类 | 🔥 高 |
| 多头解码器 | `deformation.py` | 改造后复制（时间→视角） | 🟡 中 |
| 位置编码 | `deformation.py` | 复制 `poc_fre` 函数 | 🟡 中 |

#### 5.2 需改造的模块

**A. K-Planes 改造 (hexplane.py)**

**原始实现**（4D: xyz + time）:
```python
# 6 个平面: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
# 对应: [xy, xz, xt, yz, yt, zt]
```

**改造为 3 视角场景**:
```python
# 修改 1: 时间维度 → 视角维度
resolution = [64, 64, 64, 3]  # 3 个视角

# 修改 2: 视角嵌入（离散化）
view_embedding = nn.Embedding(3, embed_dim)  # 3 个视角各一个嵌入向量
view_feature = view_embedding(view_id)  # view_id ∈ {0, 1, 2}

# 修改 3: 仍保留 6 平面结构（xy, xz, x-view, yz, y-view, z-view）
# 或简化为 3 空间平面 + 视角特征拼接
```

**B. 多头解码器改造 (deformation.py)**

**原始时间嵌入**:
```python
time_emb = poc_fre(time_stamp, poc_buf)  # 连续时间
```

**改为视角嵌入**:
```python
# 方案 1: 离散嵌入（推荐）
view_emb = nn.Embedding(num_views, embed_dim)
view_feature = view_emb(view_id)

# 方案 2: 角度编码（更通用）
angle = view_id * (2 * π / num_views)  # 假设视角均匀分布
view_feature = torch.cat([torch.sin(angle), torch.cos(angle)])
```

**C. GaussianModel 改造 (gaussian_model.py)**

**关键差异处理**:
```python
# 1. 保留 color 和 opacity（R²-Gaussian 需要）
self._features_dc = None  # 颜色 SH 系数
self._opacity = None      # 不透明度

# 2. 添加 density（X2-Gaussian 风格，可选）
self._density = None

# 3. 条件选择是否启用变形网络
if args.use_deformation:
    self._deformation = deform_network(args)
else:
    self._deformation = None
```

#### 5.3 渐进式集成策略

**阶段 1: 基础验证（1-2 天）**
1. 复制 `hexplane.py` 到 `r2_gaussian/utils/kplanes.py`
2. 修改时间维度 → 视角维度（3 视角）
3. 编写单元测试验证特征插值正确性
4. 不修改训练循环，仅测试模块功能

**阶段 2: TV 正则化集成（1 天）**
1. 复制 `PlaneTV` 类到 `r2_gaussian/utils/regularization.py`
2. 在 `train.py` 中添加 TV loss
3. 实验权重设置（建议从 0.0001 开始）

**阶段 3: 多头解码器集成（2-3 天）**
1. 复制 `deformation.py` 并改造时间嵌入
2. 在 `gaussian_model.py` 中添加 `_deformation` 属性（默认 None）
3. 修改 `render_query.py` 调用变形网络（如果启用）
4. 使用 try-except 确保向下兼容：
   ```python
   try:
       if self._deformation is not None:
           xyz, scale, rot = self._deformation(xyz, scale, rot, view_id)
   except AttributeError:
       pass  # 回退到 baseline 行为
   ```

**阶段 4: 训练策略优化（1-2 天）**
1. 评估是否需要两阶段训练（coarse/fine）
2. 调整学习率调度器参数
3. 实验密度重置策略（`reset_density()`）

**阶段 5: 完整实验（3-5 天）**
1. 在 Chest 3-view 数据上对比 baseline vs. +K-Planes vs. +K-Planes+TV vs. 完整 X2
2. 消融实验验证各模块贡献
3. 超参数调优

---

### 6. 技术风险评估

| 风险类型 | 描述 | 严重性 | 缓解方案 |
|---------|------|-------|---------|
| CUDA 子模块冲突 | `xray-gaussian-rasterization-voxelization` 版本差异 | 🔴 高 | 先用 R²-GS 的渲染器，后续评估是否需要升级 |
| 时间维度适配 | 从连续时间 → 离散视角的改造 | 🟡 中 | 使用 Embedding 层代替连续编码 |
| 超参数不匹配 | X2 的参数可能不适合静态 CT | 🟡 中 | 从小权重开始（TV loss × 0.1），逐步调优 |
| 内存开销 | K-Planes 多分辨率网格占用显存 | 🟢 低 | 减少分辨率或降低多尺度级数 |
| 训练不稳定 | 密度重置可能导致崩溃 | 🟢 低 | 监控训练曲线,必要时禁用 reset_density |

---

### 7. 与 R²-Gaussian Baseline 的对比总结

**渲染管线差异**:
- **R²-Gaussian**: 标准 X-ray 高斯投影（透视投影 + 累积密度）
- **X²-Gaussian**: 类似，但支持动态变形（可选）

**模型参数差异**:
- **R²-Gaussian**: 静态高斯（xyz, color, opacity, scale, rotation）
- **X²-Gaussian**: 动态高斯（xyz + deformation, density, scale, rotation）

**优化目标差异**:
- **R²-Gaussian**: L1 + SSIM
- **X²-Gaussian**: L1 + SSIM + TV + TimeSmoothness + Prior

**可借鉴的创新点**:
1. ✅ **K-Planes 空间分解**: 提升特征表达能力
2. ✅ **TV 正则化**: 促进空间平滑性
3. ✅ **多头解码器**: 解耦优化不同属性
4. ⚠️ **密度重置**: 需实验验证稳定性
5. ❌ **时间平滑**: 不适用于静态场景

---

## 【需要您的决策】

### 决策点 1: 迁移策略选择

**选项 A: 保守策略（推荐）**
- 仅迁移 K-Planes + PlaneTV 正则化
- 不改动 GaussianModel 核心架构
- 实验验证后决定是否继续
- **优点**: 风险低，易于回退
- **缺点**: 可能收益有限

**选项 B: 激进策略**
- 完整迁移 K-Planes + 多头解码器 + TV + 密度重置
- 改造 GaussianModel 支持 deformation 模块
- 采用两阶段训练
- **优点**: 最大化技术潜力
- **缺点**: 实现周期长（7-10 天），风险高

**选项 C: 混合策略**
- 第一阶段：K-Planes + TV（3 天）
- 第二阶段：多头解码器（5 天）
- 根据实验结果决定是否继续
- **优点**: 平衡收益与风险
- **缺点**: 需要两轮实验

**我的推荐**: **选项 C（混合策略）**
理由: 稀疏视角 CT 重建最需要空间先验（K-Planes）和平滑约束（TV），多头解码器可视第一阶段结果决定。

---

### 决策点 2: CUDA 子模块处理

**问题**: X2-Gaussian 使用的 `xray-gaussian-rasterization-voxelization` 可能与 R²-Gaussian 版本不同。

**选项 A: 保持现状**
- 继续使用 R²-Gaussian 的渲染器
- 不引入 X2 的 CUDA 代码
- **风险**: 可能无法复现 X2 的性能

**选项 B: 升级渲染器**
- 克隆 X2 的 CUDA 子模块并编译
- 替换 R²-Gaussian 的渲染器
- **风险**: 可能破坏现有功能

**选项 C: 双版本共存**
- 通过配置文件选择使用哪个渲染器
- 需要适配层处理接口差异
- **风险**: 代码复杂度增加

**我的推荐**: **选项 A（保持现状）**
理由: 先验证算法层面的改进,CUDA 优化可后续进行。如果性能确实提升明显,再考虑升级渲染器。

---

### 决策点 3: 时间维度改造方案

**问题**: X2 的时间维度如何映射到 R²-Gaussian 的视角？

**选项 A: 离散嵌入（推荐）**
```python
view_emb = nn.Embedding(num_views, 64)
view_feat = view_emb(view_id)  # view_id ∈ {0, 1, 2}
```
- **优点**: 简单高效,每个视角独立优化
- **缺点**: 泛化能力弱（固定视角数量）

**选项 B: 角度编码**
```python
angle = camera.azimuth  # 从相机参数提取角度
view_feat = [sin(angle), cos(angle), sin(2*angle), cos(2*angle), ...]
```
- **优点**: 泛化到任意视角
- **缺点**: 需要相机姿态信息

**选项 C: 混合方案**
- 离散嵌入 + 角度编码拼接
- **优点**: 兼顾表达能力和泛化性
- **缺点**: 参数量增加

**我的推荐**: **选项 A（离散嵌入）**
理由: R²-Gaussian 的稀疏视角场景（3/6/9 views）是固定的,离散嵌入足够且高效。

---

### 下一步行动

**如果选择混合策略（推荐）**:

1. **立即执行（我来做）**:
   - 创建 `code_review.md` 详细列出需修改的文件
   - 准备代码迁移清单和测试计划

2. **等待您批准后执行**:
   - 复制 `hexplane.py` 并改造时间维度
   - 集成 `PlaneTV` 到训练循环
   - 在 Chest 3-view 上运行对比实验

3. **第一阶段完成后汇报**:
   - 提供实验结果（PSNR/SSIM 对比）
   - 根据结果决定是否继续迁移多头解码器

**请您确认**:
- ✅ 是否同意混合策略？
- ✅ 是否需要我立即创建 `code_review.md`？
- ✅ 实验优先级: Chest > Foot > Head?

---

**文档版本**: v1.0
**字数统计**: 约 2480 字
**下次更新**: 用户审核后开始代码实现阶段
