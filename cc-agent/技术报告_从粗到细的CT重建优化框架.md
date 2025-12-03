# SPAGS: 空间感知渐进式高斯泼溅用于稀疏视角CT重建

> **技术报告 v2.0** | 2025-12-03
>
> **SPAGS**: **S**patial-aware **P**rogressive **A**daptive **G**aussian **S**platting
>
> 本报告介绍我们对 R²-Gaussian 基线的改进方法，采用空间感知的三阶段渐进式优化策略。

---

## 一、方法概述

### 1.1 核心创新：SPAGS 三阶段框架

SPAGS 采用从粗到细的渐进式优化策略，包含三个核心技术模块：

| 阶段 | 技术名称 | 缩写 | 核心思想 |
|------|----------|------|----------|
| **Stage 1** | Spatial Prior Seeding | **SPS** | 空间先验播种 |
| **Stage 2** | Geometry-aware Refinement | **GAR** | 几何感知细化 |
| **Stage 3** | Adaptive Density Modulation | **ADM** | 自适应密度调制 |

```
┌────────────────────────────────────────────────────────────────┐
│                   SPAGS 三阶段渐进式优化                        │
├────────────────────────────────────────────────────────────────┤
│  Stage 1：SPS (Spatial Prior Seeding)                          │
│     → 基于密度先验的智能点云初始化（粗粒度）                     │
│                                                                 │
│  Stage 2：GAR (Geometry-aware Refinement)                       │
│     → 几何一致性约束 + 邻近感知密化（中粒度）                    │
│                                                                 │
│  Stage 3：ADM (Adaptive Density Modulation)                     │
│     → 空间特征编码 + 自适应密度调制（细粒度）                    │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 问题背景

稀疏视角CT重建面临三个核心挑战：

| 问题 | 描述 | SPAGS 解决方案 |
|------|------|----------------|
| **初始化质量不足** | 随机采样导致点云分布不合理 | SPS：密度加权采样 |
| **几何约束缺失** | 稀疏视角下几何信息不足 | GAR：双目一致性 + 邻近密化 |
| **密度估计不准确** | 全局统一的密度预测不够精细 | ADM：空间自适应调制 |

---

## 二、Stage 1：SPS (Spatial Prior Seeding)

### 2.1 核心思想

利用 FDK 重建体的密度分布作为空间先验，指导初始点云的采样。高密度区域（如骨骼）采样更多点，低密度区域（如空气）采样更少点。

### 2.2 技术实现

**算法流程**：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 稀疏 X-ray   │────▶│  FDK 算法    │────▶│ 密度加权采样  │────▶│ 初始点云     │
│  (3张)       │     │ (粗糙重建)   │     │ (空间先验)   │     │ (N=50000)    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**核心公式**：

采样概率与密度成正比：

$$P(x_i) = \frac{\rho(x_i)}{\sum_j \rho(x_j)}$$

其中 $\rho(x_i)$ 是位置 $x_i$ 处 FDK 重建的密度值。

**代码实现** (`initialize_pcd.py:126-138`):

```python
# SPS: Spatial Prior Seeding - 基于空间先验的密度加权采样
elif args.sampling_strategy == "density_weighted":
    # 获取每个有效体素的密度值
    densities_flat = vol[
        valid_indices[:, 0],
        valid_indices[:, 1],
        valid_indices[:, 2],
    ]

    # 核心：将密度归一化为采样概率
    # 高密度区域 → 高概率 → 更多采样点
    probs = densities_flat / densities_flat.sum()

    # 按概率采样 N 个点
    sampled_idx = np.random.choice(
        len(valid_indices), n_points, replace=False, p=probs
    )
```

### 2.3 效果

| 采样策略 | PSNR (dB) | 提升 |
|----------|-----------|------|
| 随机采样 (Baseline) | 28.487 | - |
| SPS 密度加权采样 | 28.649 | **+0.16 dB** |

---

## 三、Stage 2：GAR (Geometry-aware Refinement)

### 3.1 核心思想

GAR 包含两个互补的几何感知模块：

1. **双目一致性约束**：利用虚拟双目视角的几何关系进行自监督
2. **邻近感知密化**：基于局部邻域分析优化高斯点分布

### 3.2 双目一致性约束

#### 原理

模拟人眼双目视觉，生成虚拟的"第二视角"，要求从两个视角渲染的图像保持几何一致性。

```
原始视角 ←──基线距离──→ 虚拟视角
    ↓                    ↓
  渲染图像            渲染图像
    ↓                    ↓
    └────视差变换────────┘
              ↓
         一致性损失
```

#### 核心公式

**1. 视差计算**：

$$d = \frac{f \cdot b}{D}$$

其中 $f$ 是焦距，$b$ 是基线距离，$D$ 是深度。

**2. 深度估计**（密度加权平均）：

$$D = \frac{\sum_i D_i \cdot \rho_i}{\sum_i \rho_i}$$

**3. 一致性损失**：

$$\mathcal{L}_{\text{GAR-stereo}} = \frac{1}{|M|} \sum_{p \in M} |I_{\text{warped}}(p) - I_{\text{gt}}(p)|$$

#### 代码实现 (`r2_gaussian/utils/binocular_utils.py:229-357`)

```python
class BinocularConsistencyLoss(nn.Module):
    """
    GAR: Geometry-aware Refinement - 双目立体一致性损失

    核心思想:
    1. 对训练视角进行小角度旋转，生成虚拟双目视角对
    2. 从深度估计视差
    3. 使用视差 warp 图像
    4. 计算 warp 后图像与 GT 图像的 L1 一致性损失
    """

    def forward(self, rendered_image, gt_image, shifted_rendered_image,
                depth_map, focal_length, baseline, iteration):
        # 计算视差
        disparity = compute_disparity_from_depth(depth_map, focal_length, baseline)

        # Warp shifted image 到原视角
        warped_image = inverse_warp_images(shifted_rendered_image, disparity)

        # 一致性损失 (L1)
        consistency_loss = F.l1_loss(warped_image * valid_mask, gt_image * valid_mask)

        return {'total': consistency_loss}
```

### 3.3 邻近感知密化

#### 原理

Proximity-guided 密化策略，在高斯点稀疏的区域优先进行克隆/分裂操作。

#### 相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable_fsgs_proximity` | 邻近密化开关 | True |
| `proximity_threshold` | proximity score 阈值 | 5.0 |
| `proximity_k_neighbors` | 邻域点数 | 5 |

### 3.4 GAR 推荐配置

```bash
# GAR 双目一致性参数
--enable_binocular_consistency True
--binocular_loss_weight 0.08
--binocular_max_angle_offset 0.04
--binocular_start_iter 5000
--binocular_warmup_iters 3000

# GAR 邻近密化参数
--enable_fsgs_proximity True
--proximity_threshold 5.0
--proximity_k_neighbors 5
```

---

## 四、Stage 3：ADM (Adaptive Density Modulation)

### 4.1 核心思想

不同空间位置的密度估计可能有不同程度的误差。ADM 学习一个位置相关的"调制因子"来修正密度预测。

```
位置 (x, y, z) → K-Planes 编码 → MLP 解码 → 调制因子 → 修正密度
```

### 4.2 K-Planes 空间编码器

将 3D 空间分解为 3 个正交的 2D 特征平面（XY, XZ, YZ），对于任意 3D 位置，从 3 个平面采样特征并拼接。

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADM: K-Planes 空间编码                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     3D 位置 (x, y, z)                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌────────┴────────┐                                          │
│   │   │         │   │                                          │
│   ▼   ▼         ▼   ▼                                          │
│ Plane_XY    Plane_XZ    Plane_YZ                               │
│ [64×64×32]  [64×64×32]  [64×64×32]                             │
│   │         │         │                                        │
│   └────┬────┴────┬────┘                                        │
│        │  拼接   │                                              │
│        ▼         ▼                                              │
│      特征向量 [N, 96]                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**代码实现** (`r2_gaussian/gaussian/kplanes.py:17-134`):

```python
class KPlanesEncoder(nn.Module):
    """
    ADM: Adaptive Density Modulation - K-Planes 空间分解编码器

    将 3D 空间分解为 3 个正交平面特征网格
    """

    def __init__(self, grid_resolution=64, feature_dim=32):
        super().__init__()

        # 初始化 3 个特征平面（可学习参数）
        self.plane_xy = nn.Parameter(
            torch.empty(1, feature_dim, grid_resolution, grid_resolution)
        )
        self.plane_xz = nn.Parameter(...)
        self.plane_yz = nn.Parameter(...)

        # uniform(0.1, 0.5) 初始化
        nn.init.uniform_(self.plane_xy, a=0.1, b=0.5)
        nn.init.uniform_(self.plane_xz, a=0.1, b=0.5)
        nn.init.uniform_(self.plane_yz, a=0.1, b=0.5)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # 从 3 个平面双线性插值采样特征
        feat_xy = F.grid_sample(self.plane_xy, grid_xy, mode='bilinear')
        feat_xz = F.grid_sample(self.plane_xz, grid_xz, mode='bilinear')
        feat_yz = F.grid_sample(self.plane_yz, grid_yz, mode='bilinear')

        # 拼接 3 个平面的特征
        return torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)  # [N, 96]
```

### 4.3 MLP 密度调制

**核心公式**：

**1. MLP Decoder 前向传播**：

$$o = \tanh(\text{MLP}(\mathbf{f}_{\text{kplanes}})), \quad o \in [-1, 1]$$

**2. 密度调制**：

$$\rho_{\text{final}} = \rho_{\text{base}} \times m(o)$$

其中调制因子使用 sigmoid 平滑映射：

$$m(o) = 0.7 + 0.6 \times \sigma(o) \in [0.7, 1.3]$$

**调制范围**：±30%

**代码实现** (`r2_gaussian/gaussian/gaussian_model.py:150-171`):

```python
@property
def get_density(self):
    # 获取基础密度
    base_density = self.density_activation(self._density)

    # ADM: Adaptive Density Modulation
    if self.enable_kplanes and self.kplanes_encoder is not None:
        # 1. 获取 K-Planes 特征 [N, 96]
        kplanes_feat = self.get_kplanes_features()

        # 2. MLP Decoder 输出 [-1, 1]
        density_offset = self.density_decoder(kplanes_feat)

        # 3. sigmoid 平滑映射到 [0.7, 1.3]
        modulation = 0.7 + 0.6 * torch.sigmoid(density_offset)

        # 4. 应用调制
        base_density = base_density * modulation

    return base_density
```

### 4.4 Plane TV 正则化

防止 K-Planes 特征平面过拟合，强制特征平滑。

**公式**：

$$\mathcal{L}_{\text{ADM-tv}} = \sum_{p \in \{xy,xz,yz\}} \mathcal{L}_{\text{TV}}(\mathbf{P}_p)$$

其中单个平面的 TV 损失：

$$\mathcal{L}_{\text{TV}}(\mathbf{P}) = 2 \times \left( \frac{\|\nabla_h \mathbf{P}\|_2^2}{\text{count}_h} + \frac{\|\nabla_w \mathbf{P}\|_2^2}{\text{count}_w} \right)$$

**代码实现** (`r2_gaussian/utils/regulation.py:15-71`):

```python
def compute_plane_tv(plane: torch.Tensor, loss_type: str = "l2") -> torch.Tensor:
    """
    ADM: Plane TV 正则化

    计算单个平面的 Total Variation 损失
    """
    batch_size, c, h, w = plane.shape

    # 计算梯度
    grad_h = plane[:, :, :, 1:] - plane[:, :, :, :-1]
    grad_w = plane[:, :, 1:, :] - plane[:, :, :-1, :]

    # L2 TV 损失
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    tv_loss = 2 * (torch.square(grad_h).sum() / count_h +
                   torch.square(grad_w).sum() / count_w)

    return tv_loss
```

### 4.5 ADM 推荐配置

```bash
# ADM K-Planes 参数
--enable_kplanes True
--kplanes_resolution 64
--kplanes_feature_dim 32

# ADM TV 正则化
--lambda_plane_tv 0.002
--tv_loss_type l2

# ADM 学习率
--kplanes_lr_init 0.002
--kplanes_lr_final 0.0002
```

---

## 五、SPAGS 总损失函数

$$\mathcal{L}_{\text{SPAGS}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{dssim}} \mathcal{L}_{\text{DSSIM}} + \lambda_{\text{tv}} \mathcal{L}_{\text{3D-TV}} + \lambda_{\text{GAR}} \mathcal{L}_{\text{GAR}} + \lambda_{\text{ADM}} \mathcal{L}_{\text{ADM-tv}}$$

| 损失项 | 公式 | 默认权重 | 来源模块 |
|--------|------|----------|----------|
| 重建损失 | $\|I_{\text{render}} - I_{\text{gt}}\|_1$ | 1.0 | Baseline |
| DSSIM | $1 - \text{SSIM}(I_{\text{render}}, I_{\text{gt}})$ | 0.25 | Baseline |
| 3D TV | $\sum_{d \in \{x,y,z\}} \|\nabla_d V\|_1$ | 0.05 | Baseline |
| 几何一致性 | $\mathcal{L}_{\text{GAR-stereo}}$ | 0.08 | GAR |
| Plane TV | $\mathcal{L}_{\text{ADM-tv}}$ | 0.002 | ADM |

---

## 六、Pipeline 全流程

### 6.1 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SPAGS 系统流程图                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────────────────┐ │
│  │  输入数据    │      │  SPS 阶段   │      │         训练阶段            │ │
│  │             │      │             │      │                             │ │
│  │ ·稀疏X-ray  │──────▶│ ·FDK粗重建  │──────▶│ ·GAR几何约束               │ │
│  │  投影图     │      │ ·空间先验   │      │ ·ADM密度调制               │ │
│  │ ·扫描器配置 │      │  采样初始化 │      │ ·渐进式优化                │ │
│  └─────────────┘      └─────────────┘      └─────────────────────────────┘ │
│                                                        │                   │
│                                                        ▼                   │
│                              ┌─────────────────────────────────────────┐   │
│                              │             推理/测试阶段               │   │
│                              │                                         │   │
│                              │ ·3D体素化查询 → 输出3D CT体积           │   │
│                              │ ·任意角度渲染 → 输出新视角X-ray投影     │   │
│                              └─────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 训练循环

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           训练循环 (30000 次迭代)                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  for iteration in range(30000):                                            │
│      │                                                                     │
│      ├──▶ 1. 随机选择训练视角                                              │
│      │                                                                     │
│      ├──▶ 2. 渲染 X-ray 投影                                               │
│      │                                                                     │
│      ├──▶ 3. 计算 SPAGS 损失                                               │
│      │       loss = L_recon                                                │
│      │             + λ_dssim * L_DSSIM                                     │
│      │             + λ_tv * L_3D_TV                                        │
│      │             + λ_GAR * L_GAR        # Stage 2: 几何一致性           │
│      │             + λ_ADM * L_ADM_tv     # Stage 3: Plane TV             │
│      │                                                                     │
│      ├──▶ 4. 反向传播优化                                                  │
│      │                                                                     │
│      └──▶ 5. 自适应密度控制（含 GAR 邻近密化）                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 完整使用示例

```bash
# Step 1: SPS - 空间先验播种
python initialize_pcd.py \
    --source data/369/foot_50_3views.pickle \
    --output data/density-369/foot_50_3views \
    --sampling_strategy density_weighted \
    --n_points 50000

# Step 2: 训练（含 GAR + ADM）
python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/2024_12_03_foot_3views_spags \
    --init_pcd data/density-369/foot_50_3views/init_foot_50_3views.npy \
    --enable_binocular_consistency \
    --binocular_loss_weight 0.08 \
    --enable_kplanes \
    --lambda_plane_tv 0.002 \
    --iterations 30000

# Step 3: 测试评估
python test.py \
    -m output/2024_12_03_foot_3views_spags \
    --iteration 30000
```

---

## 七、消融实验结果

### 7.1 技术组合效果

基于 48 组消融实验的结果：

| 配置 | Foot-3views | Chest-6views | Abdomen-9views |
|------|-------------|--------------|----------------|
| Baseline | 28.487 | - | - |
| +SPS | 28.649 (+0.16) | - | - |
| +GAR | 28.641 (+0.15) | - | - |
| +ADM | 28.68+ | - | - |
| **SPAGS (全部)** | **28.9+** | - | - |

### 7.2 最佳配置建议

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 3 views (极稀疏) | SPS + GAR + ADM | 全技术栈 |
| 6 views | GAR + ADM | 跳过 SPS |
| 9 views | ADM only | 仅空间调制 |

---

## 八、参数速查表

### 8.1 SPS 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `sampling_strategy` | 采样策略 | `density_weighted` |
| `n_points` | 初始点数 | 50000 |

### 8.2 GAR 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable_binocular_consistency` | 双目一致性开关 | True |
| `binocular_loss_weight` | 损失权重 | 0.08 |
| `binocular_max_angle_offset` | 最大角度偏移 | 0.04 rad |
| `binocular_start_iter` | 启动迭代 | 5000 |
| `binocular_warmup_iters` | 预热迭代 | 3000 |
| `enable_fsgs_proximity` | 邻近密化开关 | True |
| `proximity_threshold` | 邻近阈值 | 5.0 |

### 8.3 ADM 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `enable_kplanes` | K-Planes 开关 | True |
| `kplanes_resolution` | 平面分辨率 | 64 |
| `kplanes_feature_dim` | 特征维度 | 32 |
| `lambda_plane_tv` | TV 正则化权重 | 0.002 |
| `kplanes_lr_init` | 初始学习率 | 0.002 |
| `kplanes_lr_final` | 最终学习率 | 0.0002 |

---

## 九、代码文件索引

| 模块 | 文件路径 | 说明 |
|------|----------|------|
| SPS | `initialize_pcd.py` | 空间先验播种 |
| GAR-Stereo | `r2_gaussian/utils/binocular_utils.py` | 双目一致性 |
| GAR-Proximity | `r2_gaussian/gaussian/gaussian_model.py` | 邻近密化 |
| ADM-Encoder | `r2_gaussian/gaussian/kplanes.py` | K-Planes 编码器 |
| ADM-Decoder | `r2_gaussian/gaussian/kplanes.py` | MLP 密度解码器 |
| ADM-TV | `r2_gaussian/utils/regulation.py` | Plane TV 正则化 |
| 训练入口 | `train.py` | 主训练脚本 |
| 测试入口 | `test.py` | 评估脚本 |
