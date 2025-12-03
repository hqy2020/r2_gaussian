# 从粗到细的稀疏视角CT重建优化框架

> **技术报告 v1.2** | 2025-12-03
>
> 本报告介绍我们对R²-Gaussian基线的改进方法，采用从粗到细的三阶段渐进式优化策略。
>
> **v1.2 更新**：
> - 移除 FSGS 深度监督（MiDaS 预训练模型）
> - 移除 Bino 边缘感知平滑损失（SmoothLoss）
> - 统一为纯图像一致性约束

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

## 1.5 Pipeline 全流程概览

在深入各个技术之前，我们先从宏观角度理解整个系统是如何工作的。

### 1.5.1 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          R²-Gaussian 系统流程图                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────────────────┐ │
│  │  输入数据    │      │  预处理阶段  │      │         训练阶段            │ │
│  │             │      │             │      │                             │ │
│  │ ·稀疏X-ray  │──────▶│ ·FDK粗重建  │──────▶│ ·渲染X-ray投影              │ │
│  │  投影图     │      │ ·点云初始化  │      │ ·计算损失                   │ │
│  │ ·扫描器配置 │      │ ·高斯模型创建│      │ ·反向传播优化               │ │
│  │ ·角度信息   │      │             │      │ ·自适应密度控制             │ │
│  └─────────────┘      └─────────────┘      └─────────────────────────────┘ │
│                                                        │                   │
│                                                        ▼                   │
│                              ┌─────────────────────────────────────────┐   │
│                              │             推理/测试阶段               │   │
│                              │                                         │   │
│                              │ ·3D体素化查询 → 输出3D CT体积           │   │
│                              │ ·任意角度渲染 → 输出新视角X-ray投影     │   │
│                              │ ·评估指标计算 (PSNR, SSIM)             │   │
│                              └─────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.5.2 输入数据格式

系统接受 `.pickle` 格式的 CT 数据文件，包含以下关键信息：

```python
# 数据文件结构 (如 foot_50_3views.pickle)
data = {
    # 扫描器几何配置
    "DSD": 1500.0,        # 射线源到探测器距离 (mm)
    "DSO": 1000.0,        # 射线源到旋转中心距离 (mm)
    "nVoxel": [256, 256, 256],   # 体素网格尺寸
    "dVoxel": [0.5, 0.5, 0.5],   # 体素物理大小 (mm)
    "nDetector": [256, 256],     # 探测器像素数
    "dDetector": [0.8, 0.8],     # 探测器像素大小 (mm)

    # 训练数据
    "train": {
        "angles": [0.0, 2.094, 4.189],   # 3个扫描角度 (弧度)
        "projections": np.array(...),    # X-ray投影图 [3, 256, 256]
    },

    # 测试数据 (用于评估)
    "val": {
        "angles": [...],          # 更多角度用于验证
        "projections": [...],
    },

    # Ground Truth 3D体积 (用于评估)
    "image": np.array(...),       # CT体积 [256, 256, 256]
}
```

**代码位置** (`r2_gaussian/dataset/dataset_readers.py:193-306`):

```python
def readNAFInfo(path, eval):
    """读取 NAF 格式的 CT 数据"""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # 场景归一化：将体积缩放到 [-1, 1]³ 立方体
    scene_scale = 2 / max(scanner_cfg["sVoxel"])

    # 为每个扫描角度创建相机
    for i_split in range(n_split):
        frame_angle = angles[i_split]
        c2w = angle2pose(scanner_cfg["DSO"], frame_angle)  # 角度 → 相机位姿
        # ...创建 CameraInfo 对象
```

### 1.5.3 预处理阶段：点云初始化

在训练开始前，需要创建初始的 3D 高斯点云。

**步骤**：

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 稀疏X-ray图  │────▶│  FDK 算法    │────▶│ 密度加权采样  │────▶│ 初始点云.npy │
│  (3张)       │     │ (粗糙重建)   │     │ (50000个点)  │     │ (xyz+density)│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**代码位置** (`initialize_pcd.py`):

```python
# 1. 用 FDK 算法重建粗糙的 3D 体积
vol = tigre.algorithms.fdk(projs, geo, angles)

# 2. 密度加权采样：高密度区域采样更多点
probs = densities_flat / densities_flat.sum()  # 归一化为概率
sampled_idx = np.random.choice(len(valid_indices), n_points, p=probs)

# 3. 保存初始点云 [N, 4] = [x, y, z, density]
np.save(output_path, point_cloud)  # init_foot_50_3views.npy
```

### 1.5.4 训练阶段：核心循环

训练的核心是"渲染-比较-优化"的迭代过程。

**代码位置** (`train.py:50-433` - `training` 函数):

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           训练循环 (30000 次迭代)                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  for iteration in range(30000):                                            │
│      │                                                                     │
│      ├──▶ 1. 随机选择一个训练视角                                          │
│      │       viewpoint_cam = viewpoint_stack.pop(random_index)             │
│      │                                                                     │
│      ├──▶ 2. 渲染 X-ray 投影                                               │
│      │       render_pkg = render(viewpoint_cam, gaussians, pipe)           │
│      │       image = render_pkg["render"]  # 渲染结果                      │
│      │                                                                     │
│      ├──▶ 3. 计算损失                                                      │
│      │       loss = L1(image, gt_image)                                    │
│      │             + λ_dssim * (1 - SSIM)                                  │
│      │             + λ_tv * TV_3D                   # 3D总变差正则化       │
│      │             + λ_bino * BinocularLoss         # 双目一致性 (可选)    │
│      │             + λ_depth * DepthLoss            # 深度监督 (可选)      │
│      │             + λ_plane_tv * PlaneTVLoss       # K-Planes正则化 (可选)│
│      │                                                                     │
│      ├──▶ 4. 反向传播                                                      │
│      │       loss.backward()                                               │
│      │       optimizer.step()                                              │
│      │                                                                     │
│      └──▶ 5. 自适应密度控制 (每100次迭代)                                   │
│              - 分裂：梯度大且尺度大的高斯 → 分裂成2个                       │
│              - 克隆：梯度大但尺度小的高斯 → 复制一份                        │
│              - 剪枝：密度过低或超出边界的高斯 → 删除                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**核心渲染函数** (`r2_gaussian/gaussian/render_query.py:79-159`):

```python
def render(viewpoint_camera, pc, pipe, scaling_modifier=1.0):
    """
    渲染 X-ray 投影

    原理：将 3D 高斯点沿射线方向投影到 2D 探测器平面，
          累积沿路径的密度贡献
    """
    # 配置光栅化器
    raster_settings = GaussianRasterizationSettings(
        image_height=viewpoint_camera.image_height,
        image_width=viewpoint_camera.image_width,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        # ...
    )
    rasterizer = GaussianRasterizer(raster_settings)

    # 执行光栅化：3D高斯 → 2D投影
    rendered_image, radii = rasterizer(
        means3D=pc.get_xyz,        # 高斯中心位置 [N, 3]
        opacities=pc.get_density,  # 密度值 [N, 1]
        scales=pc.get_scaling,     # 尺度 [N, 3]
        rotations=pc.get_rotation, # 旋转四元数 [N, 4]
    )

    return {"render": rendered_image, "radii": radii, ...}
```

### 1.5.5 推理/测试阶段

训练完成后，可以：
1. **重建 3D CT 体积**：通过体素化查询
2. **生成新视角投影**：从任意角度渲染 X-ray 图像

**代码位置** (`test.py:37-89` 和 `r2_gaussian/gaussian/render_query.py:26-76`):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              推理阶段                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输出1：3D CT体积重建                                                        │
│  ─────────────────                                                          │
│  vol_pred = query(gaussians, center, nVoxel, sVoxel, pipe)["vol"]          │
│                                                                             │
│  原理：在规则的 3D 网格上查询每个体素位置的密度值                             │
│        将所有高斯在该位置的贡献累加起来                                       │
│                                                                             │
│  输出2：新视角X-ray投影                                                      │
│  ──────────────────                                                         │
│  new_image = render(new_camera, gaussians, pipe)["render"]                 │
│                                                                             │
│  原理：从任意新角度执行光栅化渲染                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**体素化查询函数** (`r2_gaussian/gaussian/render_query.py:26-76`):

```python
def query(pc, center, nVoxel, sVoxel, pipe, scaling_modifier=1.0):
    """
    体素化查询：将高斯点云转换为规则的 3D 体素网格

    参数:
        center: 体积中心位置
        nVoxel: 体素网格尺寸 [256, 256, 256]
        sVoxel: 体积物理大小
    """
    voxel_settings = GaussianVoxelizationSettings(
        nVoxel_x=nVoxel[0], nVoxel_y=nVoxel[1], nVoxel_z=nVoxel[2],
        sVoxel_x=sVoxel[0], sVoxel_y=sVoxel[1], sVoxel_z=sVoxel[2],
        center_x=center[0], center_y=center[1], center_z=center[2],
    )
    voxelizer = GaussianVoxelizer(voxel_settings)

    # 体素化：3D高斯 → 规则网格
    vol_pred, radii = voxelizer(
        means3D=pc.get_xyz,
        opacities=pc.get_density,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
    )

    return {"vol": vol_pred}  # [256, 256, 256] 的 CT 体积
```

### 1.5.6 高斯模型数据结构

每个 3D 高斯由以下参数定义：

```python
class GaussianModel:
    """
    3D 高斯点云模型

    每个高斯包含:
    - _xyz:      位置 [N, 3]       在 3D 空间中的中心坐标
    - _density:  密度 [N, 1]       该位置的 CT 密度值 (Hounsfield单位)
    - _scaling:  尺度 [N, 3]       高斯的三轴尺寸
    - _rotation: 旋转 [N, 4]       四元数表示的朝向

    可选 (X²-Gaussian):
    - kplanes_encoder:   K-Planes 空间编码器
    - density_decoder:   密度调制 MLP
    """
```

### 1.5.7 完整使用示例

```bash
# 步骤1：初始化点云 (可选，如果使用密度加权)
python initialize_pcd.py \
    --source data/369/foot_50_3views.pickle \
    --output data/density-369/foot_50_3views \
    --sampling_strategy density_weighted \
    --n_points 50000

# 步骤2：训练
python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/2024_12_03_foot_3views_x2gs \
    --enable_kplanes \
    --iterations 30000

# 步骤3：测试和评估
python test.py \
    -m output/2024_12_03_foot_3views_x2gs \
    --iteration 30000

# 输出文件:
# - output/.../test/iter_30000/vol_pred.npy     # 重建的3D体积
# - output/.../test/iter_30000/eval3d.yml       # 评估指标 (PSNR, SSIM)
# - output/.../test/iter_30000/vol_pred.nii.gz  # 可用3D Slicer查看
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

##### 核心数学公式

**1. 双目一致性损失**（纯图像一致性，v1.2 简化版）：

$$\mathcal{L}_{\text{bino}} = \mathcal{L}_{\text{consistency}}$$

> **v1.2 更新**：已移除边缘感知平滑损失（$\mathcal{L}_{\text{smooth}}$），仅保留纯图像一致性约束。

**2. 一致性损失（Warp 后的 L1 损失）**：

$$\mathcal{L}_{\text{consistency}} = \frac{1}{|M|} \sum_{p \in M} |I_{\text{warped}}(p) - I_{\text{gt}}(p)|$$

其中 $M$ 是有效像素掩码，$I_{\text{warped}}$ 是通过视差变形后的图像。

**3. 视差计算公式**：

$$d = \frac{f \cdot b}{D}$$

其中 $f$ 是焦距，$b$ 是基线距离，$D$ 是深度。

**5. Bino 深度估计方法**（从高斯点直接计算）：

$$D_{\text{bino}} = \frac{\sum_{i} D_i \cdot \rho_i}{\sum_{i} \rho_i}$$

其中 $D_i$ 是第 $i$ 个高斯点在相机坐标系下的深度，$\rho_i$ 是其密度值。

> **注意**：Bino 的深度是从高斯模型自身计算的粗略深度（全图同一值），而非逐像素精细深度。

##### 技术实现

**文字描述**：
1. 对于每个训练视角，生成一个小角度偏移的"虚拟双目视角"（约 3.4°）
2. 分别从两个视角渲染图像
3. 从高斯点计算密度加权平均深度
4. 用视差公式将一个视角的图像"变形"到另一个视角
5. 要求变形后的图像与 GT 图像尽量一致

**代码实现** (`r2_gaussian/utils/binocular_utils.py:229-357`)：

```python
class BinocularConsistencyLoss(nn.Module):
    """
    双目立体一致性损失（v1.2 简化版）

    核心思想:
    1. 对训练视角进行小角度旋转，生成虚拟双目视角对
    2. 从深度估计视差
    3. 使用视差 warp 图像
    4. 计算 warp 后图像与 GT 图像的 L1 一致性损失

    v1.2 更新: 已移除边缘感知平滑损失
    """

    def forward(self, rendered_image, gt_image, shifted_rendered_image,
                depth_map, focal_length, baseline, iteration):
        # 计算视差
        disparity = compute_disparity_from_depth(depth_map, focal_length, baseline)

        # Warp shifted image 到原视角
        warped_image = inverse_warp_images(shifted_rendered_image, disparity)

        # 一致性损失 (L1)
        consistency_loss = F.l1_loss(warped_image * valid_mask, gt_image * valid_mask)

        return {'consistency': consistency_loss, 'total': consistency_loss}
```

**训练流程中的调用** (`train.py:228-261`)：

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

    # 4. 估计深度图（密度加权平均）
    depth_map = estimate_depth_for_ct(gaussians, viewpoint_cam)

    # 5. 计算双目一致性损失
    bino_losses = binocular_loss_module(...)

    # 6. 加入总损失
    loss["total"] += opt.binocular_loss_weight * bino_losses["total"]
```

#### 2.2.2 FSGS：Proximity-guided 密化

> ⚠️ **v1.2 更新**：FSGS 的深度监督功能（MiDaS 预训练模型）已被移除。当前仅保留 Proximity-guided 密化策略。

##### 概述

FSGS 模块在 v1.2 版本中仅保留 **Proximity-guided 密化策略**，用于改善高斯点的空间分布。

##### 保留功能：Proximity-guided 密化

Proximity-guided 密化是一种基于邻域分析的密化策略，在高斯点稀疏的区域优先进行克隆/分裂操作。

**相关参数**：
- `enable_fsgs_proximity`: 主开关
- `proximity_threshold`: proximity score 阈值
- `enable_medical_constraints`: 医学约束开关
- `proximity_organ_type`: 器官类型

##### 已移除功能

以下功能在 v1.2 版本中已被移除：

| 已移除功能 | 原因 |
|-----------|------|
| MiDaS 深度估计 | 第三方模型依赖，增加复杂度 |
| `pseudo_view_sampler` | 与 Bino 的 `create_shifted_camera` 功能重复 |
| Pearson 深度损失 | 移除深度监督后不再需要 |

> 深度相关功能统一使用 Bino 模块的 `estimate_depth_for_ct()` 函数，基于高斯点自身计算密度加权平均深度。

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

**核心数学公式**：

**1. MLP Decoder 前向传播**：

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{f} + \mathbf{b}_1), \quad \mathbf{f} \in \mathbb{R}^{96}$$
$$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$$
$$o = \tanh(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3), \quad o \in [-1, 1]$$

其中 $\mathbf{f}$ 是 K-Planes 拼接后的 96 维特征。

**2. 密度调制公式**：

$$\rho_{\text{final}} = \rho_{\text{base}} \times m(o)$$

其中调制因子 $m(o)$ 使用 sigmoid 平滑映射：

$$m(o) = 0.7 + 0.6 \times \sigma(o) = 0.7 + \frac{0.6}{1 + e^{-o}}$$

**调制范围分析**：
- 当 $o \to -\infty$：$m \to 0.7$（密度减少 30%）
- 当 $o = 0$：$m = 1.0$（密度不变）
- 当 $o \to +\infty$：$m \to 1.3$（密度增加 30%）

```
┌─────────────────────────────────────────────────────────────────┐
│                    密度调制流程图                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  高斯点位置 (x, y, z)                                           │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │  K-Planes       │     │  基础密度        │                   │
│  │  Encoder        │     │  ρ_base         │                   │
│  │  ──────────     │     │                 │                   │
│  │  plane_xy       │     │  sigmoid(_ρ)   │                   │
│  │  plane_xz       │     │                 │                   │
│  │  plane_yz       │     └────────┬────────┘                   │
│  └────────┬────────┘              │                            │
│           │                       │                            │
│           ▼                       │                            │
│    特征 f [N, 96]                │                            │
│           │                       │                            │
│           ▼                       │                            │
│  ┌─────────────────┐              │                            │
│  │  MLP Decoder    │              │                            │
│  │  ──────────     │              │                            │
│  │  96→128→128→1   │              │                            │
│  │  + Tanh         │              │                            │
│  └────────┬────────┘              │                            │
│           │                       │                            │
│           ▼                       │                            │
│    offset o ∈ [-1, 1]            │                            │
│           │                       │                            │
│           ▼                       │                            │
│    m = 0.7 + 0.6·σ(o)            │                            │
│    调制因子 ∈ [0.7, 1.3]         │                            │
│           │                       │                            │
│           └───────────┬───────────┘                            │
│                       │                                        │
│                       ▼                                        │
│              ρ_final = ρ_base × m                              │
│              最终密度                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**文字描述**：
1. 将 96 维 K-Planes 特征输入 3 层 MLP 网络（96→128→128→1）
2. 输出一个标量值（通过 Tanh 限制在 [-1, 1]）
3. 使用 sigmoid 将输出平滑映射到调制因子 [0.7, 1.3]（即 ±30% 调制）
4. 用调制因子**乘以**基础密度（乘法调制，而非加法）

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

**核心数学公式**（X²-Gaussian 原版对齐）：

对于单个特征平面 $\mathbf{P} \in \mathbb{R}^{C \times H \times W}$，TV 损失定义为：

$$\mathcal{L}_{\text{TV}}(\mathbf{P}) = 2 \times \left( \frac{\sum_{i,j,c} (P_{c,i+1,j} - P_{c,i,j})^2}{\text{count}_h} + \frac{\sum_{i,j,c} (P_{c,i,j+1} - P_{c,i,j})^2}{\text{count}_w} \right)$$

其中：
- $\text{count}_h = C \times (H-1) \times W$：水平方向差分的元素数
- $\text{count}_w = C \times H \times (W-1)$：垂直方向差分的元素数

**三个平面的总 TV 损失**：

$$\mathcal{L}_{\text{plane-tv}} = \mathcal{L}_{\text{TV}}(\mathbf{P}_{xy}) + \mathcal{L}_{\text{TV}}(\mathbf{P}_{xz}) + \mathcal{L}_{\text{TV}}(\mathbf{P}_{yz})$$

**TV 正则化的作用**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    TV 正则化的作用                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  无 TV 正则化：                                                  │
│  ┌─────────────────┐                                            │
│  │ ▓▒░▓▒░▓▒░▓▒░   │  ← 特征平面噪声多，高频伪影                │
│  │ ░▓▒░▓▒░▓▒░▓   │                                            │
│  │ ▓▒░▓▒░▓▒░▓▒░   │                                            │
│  └─────────────────┘                                            │
│                                                                 │
│  有 TV 正则化：                                                  │
│  ┌─────────────────┐                                            │
│  │ ▓▓▓▓▓▒▒▒░░░░   │  ← 特征平面平滑，减少伪影                  │
│  │ ▓▓▓▓▒▒▒░░░░░   │                                            │
│  │ ▓▓▓▒▒▒▒░░░░░   │                                            │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> **与 R²-Gaussian 原生 3D TV 的区别**：
> - **3D TV**：作用于体素密度场 $V \in \mathbb{R}^{D \times H \times W}$，正则化最终输出
> - **Plane TV**：作用于 K-Planes 特征平面 $\mathbf{P} \in \mathbb{R}^{C \times H \times W}$，正则化中间特征

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

**TV 损失计算函数** (`r2_gaussian/utils/regulation.py:15-71`):

```python
def compute_plane_tv(plane: torch.Tensor, loss_type: str = "l2") -> torch.Tensor:
    """
    计算单个平面的 Total Variation (TV) 损失

    公式（X²-Gaussian 原版）：
        TV(P) = 2 * (Σ(P[i+1,j] - P[i,j])² / count_h
                   + Σ(P[i,j+1] - P[i,j])² / count_w)
    """
    batch_size, c, h, w = plane.shape

    # 计算精确的计数
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)

    # 计算水平梯度 [batch, C, H, W-1]
    grad_h = plane[:, :, :, 1:] - plane[:, :, :, :-1]

    # 计算垂直梯度 [batch, C, H-1, W]
    grad_w = plane[:, :, 1:, :] - plane[:, :, :-1, :]

    # L2 损失（X²-Gaussian 原版）
    h_tv = torch.square(grad_h).sum()
    w_tv = torch.square(grad_w).sum()
    tv_loss = 2 * (h_tv / count_h + w_tv / count_w)

    return tv_loss
```

---

## 三、核心公式汇总

为方便查阅，以下汇总本文涉及的所有核心数学公式。

### 3.1 总损失函数（v1.2 更新）

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{dssim}} \mathcal{L}_{\text{DSSIM}} + \lambda_{\text{tv}} \mathcal{L}_{\text{3D-TV}} + \lambda_{\text{bino}} \mathcal{L}_{\text{bino}} + \lambda_{\text{plane-tv}} \mathcal{L}_{\text{plane-tv}}$$

| 损失项 | 公式 | 默认权重 | 来源技术 |
|--------|------|----------|----------|
| 重建损失 | $\|I_{\text{render}} - I_{\text{gt}}\|_1$ | 1.0 | Baseline |
| DSSIM | $1 - \text{SSIM}(I_{\text{render}}, I_{\text{gt}})$ | 0.25 | Baseline |
| 3D TV | $\sum_{d \in \{x,y,z\}} \|\nabla_d V\|_1$ | 0.05 | Baseline |
| 双目一致性 | $\mathcal{L}_{\text{consistency}}$ | 0.08~0.15 | Bino |
| Plane TV | $\sum_{p \in \{xy,xz,yz\}} \mathcal{L}_{\text{TV}}(\mathbf{P}_p)$ | 0.002 | X²-GS |

> **v1.2 更新**：已移除 FSGS 深度监督损失和 Bino 边缘感知平滑损失。

### 3.2 Bino 双目一致性损失

$$\mathcal{L}_{\text{bino}} = \mathcal{L}_{\text{consistency}} = \frac{1}{|M|} \sum_{p \in M} |I_{\text{warped}}(p) - I_{\text{gt}}(p)|$$

其中 $M$ 是有效像素掩码，$I_{\text{warped}}$ 是通过视差变形后的图像。

> **v1.2 更新**：已移除边缘感知平滑损失（$\mathcal{L}_{\text{smooth}}$）。

### 3.3 X²-Gaussian 密度调制

$$\rho_{\text{final}} = \rho_{\text{base}} \times \left( 0.7 + 0.6 \cdot \sigma(\text{MLP}(\mathbf{f}_{\text{kplanes}})) \right)$$

调制范围：$[0.7, 1.3]$（即 ±30%）

### 3.4 Plane TV 正则化

$$\mathcal{L}_{\text{TV}}(\mathbf{P}) = 2 \times \left( \frac{\|\nabla_h \mathbf{P}\|_2^2}{\text{count}_h} + \frac{\|\nabla_w \mathbf{P}\|_2^2}{\text{count}_w} \right)$$

### 3.5 深度估计（统一方案）

v1.2 版本统一使用高斯点密度加权平均深度：

$$D = \frac{\sum_i D_i \rho_i}{\sum_i \rho_i}$$

其中 $D_i$ 是第 $i$ 个高斯点在相机坐标系下的深度，$\rho_i$ 是其密度值。

> **v1.2 更新**：已移除 MiDaS 预训练模型深度估计，统一使用自身高斯点计算。

---

