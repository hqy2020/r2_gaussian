# 从粗到细的稀疏视角CT重建优化框架

> **技术报告 v1.0** | 2025-12-03
>
> 本报告介绍我们对R²-Gaussian基线的改进方法，采用从粗到细的三阶段渐进式优化策略。

---

## 一、为什么需要改进？（动机）

### 1.1 问题背景：稀疏视角CT重建的困境

想象你只能从3个角度拍摄一个物体的照片，然后要重建出它的完整3D模型——这就是稀疏视角CT重建面临的挑战。

```
传统CT扫描：360°连续拍摄 → 信息充足 → 重建清晰
稀疏视角CT：仅3/6/9个角度 → 信息严重不足 → 重建困难
```

**为什么要用稀疏视角？**
- 减少辐射剂量（保护患者）
- 加快扫描速度（提高效率）
- 降低设备成本

### 1.2 现有方法（R²-Gaussian）的三个问题

| 问题 | 通俗解释 | 后果 |
|------|----------|------|
| **初始化质量不足** | 一开始放置3D点的位置不够合理 | 训练起点差，收敛慢 |
| **几何约束缺失** | 模型只看到3个角度的图片，容易"过拟合" | 换个角度看就不对了 |
| **密度估计不准确** | 判断"这里有多少组织"不够精确 | 重建的CT图像模糊或有伪影 |

### 1.3 我们的解决思路

**核心想法**：像画画一样，先画轮廓（粗），再画细节（细）。

```
┌────────────────────────────────────────────────────────────────┐
│                   从粗到细的三阶段优化                           │
├────────────────────────────────────────────────────────────────┤
│  第一阶段：Init-PCD                                             │
│     → 智能放置初始3D点（粗粒度）                                 │
│                                                                 │
│  第二阶段：Bino + FSGS                                          │
│     → 利用几何约束精炼表面（中粒度）                              │
│                                                                 │
│  第三阶段：X²-Gaussian                                          │
│     → 根据空间位置微调密度（细粒度）                              │
└────────────────────────────────────────────────────────────────┘
```

---

## 二、方法详解

### 2.1 第一阶段：Init-PCD（智能初始化）

#### 通俗解释

就像下棋时的开局布局：

- **原来的方法**：在棋盘上随机撒棋子 → 有些地方太密，有些地方太稀
- **我们的方法**：在"重要"的位置多放棋子 → 骨骼等高密度区域放更多点

**核心思想**：利用FDK重建体（虽然模糊但包含有用信息）的密度分布来指导采样。

#### 技术实现

**文字描述**：
1. 首先用传统FDK算法从稀疏视角重建一个粗糙的3D体积
2. 计算每个体素位置的密度值
3. 密度越高的区域，被采样的概率越大
4. 最终生成50000个初始点，形成高质量的初始点云

**代码实现** (`initialize_pcd.py:126-138`):

```python
# 密度加权采样：高密度区域更可能被采样
elif args.sampling_strategy == "density_weighted":
    print(f"Using density-weighted sampling strategy.")

    # 获取每个有效体素的密度值
    densities_flat = vol[
        valid_indices[:, 0],
        valid_indices[:, 1],
        valid_indices[:, 2],
    ]

    # 🎯 核心：将密度归一化为采样概率
    # 密度高的区域 → 概率大 → 更容易被采样
    probs = densities_flat / densities_flat.sum()

    # 按概率采样50000个点
    sampled_idx = np.random.choice(
        len(valid_indices), n_points, replace=False, p=probs
    )
```

**效果对比**：

| 采样策略 | PSNR (dB) | 说明 |
|----------|-----------|------|
| 随机采样 (原版) | 28.487 | 基线 |
| 密度加权采样 | 28.649 | **+0.16 dB** |

---

### 2.2 第二阶段：Bino + FSGS（几何精细化）

这一阶段包含两个互补的技术。

#### 2.2.1 Bino：双目一致性约束

##### 通俗解释

人有两只眼睛，可以通过"视差"感知深度。我们模拟这个原理：

```
左眼看到的 ←→ 右眼看到的
     ↓              ↓
  比较差异 → 推断物体的深度位置
```

在CT场景中，我们生成一个"虚拟的第二视角"，要求从这两个角度渲染的图像保持一致。

##### 技术实现

**文字描述**：
1. 对于每个训练视角，生成一个小角度偏移的"虚拟双目视角"
2. 分别从两个视角渲染图像
3. 计算深度图，然后用视差公式将一个视角的图像"变形"到另一个视角
4. 要求变形后的图像与真实渲染尽量一致

**代码实现** (`r2_gaussian/utils/binocular_utils.py:31-120`):

```python
class SmoothLoss(nn.Module):
    """
    边缘感知的视差平滑损失

    原理: 在图像边缘处允许更大的视差变化，在平坦区域强制平滑
    公式: L_smooth = mean(|∇d| * exp(-|∇I|))
    """

    def forward(self, disparity: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # 计算图像梯度并生成边缘权重
        # 系数 -0.33 控制边缘敏感度
        edge_x_im = torch.exp(self.edge_conv_x_3(image).abs() * -0.33)
        edge_y_im = torch.exp(self.edge_conv_y_3(image).abs() * -0.33)

        # 计算视差梯度
        edge_x_d = self.edge_conv_x_1(disparity)
        edge_y_d = self.edge_conv_y_1(disparity)

        # 🎯 核心：边缘感知平滑损失
        # 边缘处允许大视差变化，平坦处强制平滑
        loss = (edge_x_im * edge_x_d.abs()).mean() + \
               (edge_y_im * edge_y_d.abs()).mean()
        return loss
```

**训练流程中的调用** (`train.py:259-293`):

```python
# 双目立体一致性损失
if use_binocular and iteration >= opt.binocular_start_iter:
    # 1. 生成随机角度偏移（约2-3度）
    angle_offset = get_random_angle_offset(opt.binocular_max_angle_offset)

    # 2. 创建虚拟双目相机
    shifted_cam, baseline = create_shifted_camera(viewpoint_cam, angle_offset)

    # 3. 渲染虚拟视角
    shifted_render_pkg = render(shifted_cam, gaussians, pipe)
    shifted_image = shifted_render_pkg["render"]

    # 4. 估计深度图
    depth_map = estimate_depth_for_ct(gaussians, viewpoint_cam, ...)

    # 5. 🎯 计算双目一致性损失
    bino_losses = binocular_loss_module(
        rendered_image=image,
        gt_image=gt_image,
        shifted_rendered_image=shifted_image,
        depth_map=depth_map,
        ...
    )

    # 6. 加入总损失
    loss["total"] += opt.binocular_loss_weight * bino_losses["total"]
```

#### 2.2.2 FSGS：伪视角深度监督

##### 通俗解释

我们借助一个预训练的"深度估计模型"（MiDaS）来提供额外的监督信号：

```
渲染的图像 → MiDaS模型 → 估计的深度图
                              ↓
              要求渲染深度与估计深度一致
```

这就像请了一个"深度专家"来指导我们的重建。

##### 技术实现

**文字描述**：
1. 在训练视角之外采样"伪视角"
2. 从伪视角渲染图像
3. 用预训练的MiDaS模型估计深度
4. 要求我们的渲染深度与MiDaS估计的深度保持一致

**代码实现** (`r2_gaussian/utils/depth_estimator.py:22-99`):

```python
class MiDaSDepthEstimator(nn.Module):
    """
    MiDaS 深度估计器
    使用预训练的 DPT (Dense Prediction Transformer) 模型估计相对深度
    """

    def __init__(self, model_type="dpt_hybrid", device="cuda"):
        super().__init__()

        # 🎯 加载预训练模型
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            "DPT_Hybrid",  # 平衡速度和质量
            pretrained=True
        )
        self.model = self.model.to(device)
        self.model.eval()

        # 冻结参数（不训练深度估计器本身）
        for param in self.parameters():
            param.requires_grad = False
```

**训练流程中的调用** (`train.py:315-346`):

```python
# FSGS 伪视角深度监督
if use_depth_supervision and iteration >= opt.start_sample_pseudo:
    # 1. 生成伪视角
    pseudo_cam = pseudo_view_sampler.sample_pseudo_view(viewpoint_cam)

    # 2. 渲染伪视角
    with torch.no_grad():
        pseudo_render_pkg = render(pseudo_cam, gaussians, pipe)
        pseudo_image = pseudo_render_pkg["render"]

        # 3. 🎯 MiDaS估计深度（不需要梯度）
        midas_depth = depth_estimator(pseudo_image.unsqueeze(0)).squeeze(0)

    # 4. 获取渲染深度（需要梯度，用于反向传播）
    rendered_depth = pseudo_render_pkg.get("depth", pseudo_image)

    # 5. 计算深度损失（使用Pearson相关性）
    loss_depth = compute_depth_loss(rendered_depth, midas_depth, loss_type="pearson")

    # 6. 加入总损失
    loss["total"] += opt.depth_pseudo_weight * loss_depth
```

---

### 2.3 第三阶段：X²-Gaussian（空间自适应密度调制）

#### 通俗解释

想象一个"智能调节器"，根据3D空间中的位置自动调整密度预测：

```
位置 (x, y, z) → 查询3个特征平面 → 拼接特征 → MLP网络 → 密度调制因子
                     ↓
            XY平面 + XZ平面 + YZ平面
```

**核心思想**：不同空间位置的密度估计可能有不同程度的误差，我们学习一个"补偿因子"来修正。

#### 技术实现

##### 2.3.1 K-Planes 空间编码器

**文字描述**：
1. 将3D空间分解为3个正交的2D特征平面（XY, XZ, YZ）
2. 每个平面是一个64×64的特征网格，每个网格点存储32维特征
3. 对于任意3D位置，从3个平面采样特征并拼接

**代码实现** (`r2_gaussian/gaussian/kplanes.py:17-134`):

```python
class KPlanesEncoder(nn.Module):
    """
    K-Planes 空间分解编码器

    将 3D 空间 (x,y,z) 分解为 3 个正交平面特征网格：
    - plane_xy: [1, 32, 64, 64] 特征平面
    - plane_xz: [1, 32, 64, 64] 特征平面
    - plane_yz: [1, 32, 64, 64] 特征平面
    """

    def __init__(self, grid_resolution=64, feature_dim=32):
        super().__init__()

        # 🎯 初始化3个特征平面（可学习参数）
        self.plane_xy = nn.Parameter(
            torch.empty(1, feature_dim, grid_resolution, grid_resolution)
        )
        self.plane_xz = nn.Parameter(
            torch.empty(1, feature_dim, grid_resolution, grid_resolution)
        )
        self.plane_yz = nn.Parameter(
            torch.empty(1, feature_dim, grid_resolution, grid_resolution)
        )

        # 使用 uniform(0.1, 0.5) 初始化（对齐X²-Gaussian原版）
        nn.init.uniform_(self.plane_xy, a=0.1, b=0.5)
        nn.init.uniform_(self.plane_xz, a=0.1, b=0.5)
        nn.init.uniform_(self.plane_yz, a=0.1, b=0.5)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        提取指定位置的 K-Planes 特征
        输入: xyz [N, 3] - N个3D点
        输出: features [N, 96] - 拼接后的特征 (32 × 3)
        """
        # 归一化坐标到 [-1, 1]
        xyz_normalized = self._normalize_coords(xyz)
        x, y, z = xyz_normalized[:, 0], xyz_normalized[:, 1], xyz_normalized[:, 2]

        # 🎯 从3个平面双线性插值采样特征
        # Plane XY：使用 (x, y) 坐标
        grid_xy = torch.stack([x, y], dim=-1).view(1, N, 1, 2)
        feat_xy = F.grid_sample(self.plane_xy, grid_xy, mode='bilinear')

        # Plane XZ：使用 (x, z) 坐标
        grid_xz = torch.stack([x, z], dim=-1).view(1, N, 1, 2)
        feat_xz = F.grid_sample(self.plane_xz, grid_xz, mode='bilinear')

        # Plane YZ：使用 (y, z) 坐标
        grid_yz = torch.stack([y, z], dim=-1).view(1, N, 1, 2)
        feat_yz = F.grid_sample(self.plane_yz, grid_yz, mode='bilinear')

        # 拼接3个平面的特征
        return torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)  # [N, 96]
```

##### 2.3.2 MLP Decoder 密度调制

**文字描述**：
1. 将96维K-Planes特征输入3层MLP网络
2. 输出一个标量值（通过Tanh限制在[-1, 1]）
3. 将输出映射到调制因子[0.7, 1.3]（即±30%调制）
4. 用调制因子乘以基础密度

**代码实现** (`r2_gaussian/gaussian/kplanes.py:172-231`):

```python
class DensityMLPDecoder(nn.Module):
    """
    MLP Decoder：将 K-Planes 特征映射到 density 调制因子
    """

    def __init__(self, input_dim=96, hidden_dim=128, num_layers=3):
        super().__init__()

        layers = []
        # 第一层：96 → 128
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # 中间层：128 → 128
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # 输出层：128 → 1
        layers.append(nn.Linear(hidden_dim, 1))
        # 🎯 Tanh约束输出到 [-1, 1] 防止极端调制
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
```

**密度调制逻辑** (`r2_gaussian/gaussian/gaussian_model.py:150-171`):

```python
@property
def get_density(self):
    # 获取基础密度
    base_density = self.density_activation(self._density)

    # 🎯 K-Planes 特征调制
    if self.enable_kplanes and self.kplanes_encoder is not None:
        # 1. 获取K-Planes特征 [N, 96]
        kplanes_feat = self.get_kplanes_features()

        # 2. MLP Decoder输出 [-1, 1]
        density_offset = self.density_decoder(kplanes_feat)

        # 3. 🎯 关键：使用sigmoid平滑映射到 [0.7, 1.3]
        # 这意味着最多调整±30%的密度
        modulation = 0.7 + 0.6 * torch.sigmoid(density_offset)

        # 4. 应用调制
        base_density = base_density * modulation

    return base_density
```

##### 2.3.3 TV正则化防止过拟合

**代码实现** (`train.py:295-304`):

```python
# K-Planes TV 正则化损失
if opt.lambda_plane_tv > 0 and gaussians.enable_kplanes:
    planes = gaussians.kplanes_encoder.get_plane_params()

    # 🎯 计算总变差损失，强制特征平面平滑
    tv_loss_planes = compute_plane_tv_loss(
        planes=planes,
        weights=opt.plane_tv_weight_proposal,
        loss_type="l2",  # 使用L2范数
    )

    loss["plane_tv"] = tv_loss_planes
    loss["total"] += opt.lambda_plane_tv * tv_loss_planes  # lambda=0.002
```

---

## 三、技术组合策略

### 3.1 消融实验发现

通过48组消融实验，我们发现不同场景需要不同的技术组合：

| 场景 | 最佳组合 | 提升 | 原因 |
|------|----------|------|------|
| **3 views** | Init-PCD + X² + Bino | +0.45 dB | 极稀疏场景需要最多约束 |
| **6 views** | X² + FSGS | +0.09 dB | 中等场景需要深度监督 |
| **9 views** | 纯X² | +0.17 dB | 信息充足，特征增强足够 |

### 3.2 关键发现

1. **X²-Gaussian是核心技术**：在所有最佳组合中都存在
2. **全技术组合(IXFB)不是最优**：存在技术冲突
3. **Init-PCD对极稀疏场景关键**：但对密集场景可能有害

---

## 四、推荐配置

### 4.1 3 views极稀疏场景

```bash
python train.py \
    -s data/density-369/foot_50_3views \  # 使用密度加权点云
    --enable_kplanes \                     # 启用K-Planes
    --kplanes_resolution 64 \
    --kplanes_dim 32 \
    --lambda_plane_tv 0.002 \              # TV正则化
    --enable_binocular_consistency \       # 启用双目一致性
    --binocular_loss_weight 0.08 \
    --binocular_max_angle_offset 0.04
```

### 4.2 6 views中等场景

```bash
python train.py \
    -s data/369/chest_50_6views \          # 可用标准点云
    --enable_kplanes \
    --depth_pseudo_weight 0.08 \           # 启用深度监督
    --start_sample_pseudo 5000
```

### 4.3 9 views相对密集场景

```bash
python train.py \
    -s data/369/abdomen_50_9views \        # 标准点云
    --enable_kplanes \
    --lambda_plane_tv 0.001                # 较弱的TV正则化
```

---

## 五、总结

我们的方法通过**从粗到细的三阶段优化**解决稀疏视角CT重建的挑战：

```
Stage 1: Init-PCD    → 智能初始化（在正确的地方放点）
Stage 2: Bino + FSGS → 几何精细化（利用双目和深度约束表面）
Stage 3: X²-Gaussian → 密度微调（空间自适应补偿）
```

**核心贡献**：
1. 场景自适应的技术组合策略
2. 密度加权初始化充分利用先验信息
3. 多约束联合优化提升泛化能力

---

## 附录：代码文件索引

| 技术 | 核心文件 | 关键行号 |
|------|----------|----------|
| Init-PCD | `initialize_pcd.py` | 126-138 (密度加权采样) |
| K-Planes | `r2_gaussian/gaussian/kplanes.py` | 17-134 (编码器), 172-231 (解码器) |
| 密度调制 | `r2_gaussian/gaussian/gaussian_model.py` | 150-171 |
| Bino | `r2_gaussian/utils/binocular_utils.py` | 31-120 (平滑损失) |
| FSGS | `r2_gaussian/utils/depth_estimator.py` | 22-99 |
| 训练集成 | `train.py` | 259-293 (Bino), 315-346 (FSGS), 295-304 (TV) |
