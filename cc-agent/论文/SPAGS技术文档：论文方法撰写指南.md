# SPAGS技术文档：论文方法撰写指南

> 本文档旨在让读者无需阅读源代码即可理解 R²-Gaussian 的核心算法实现，以便准确撰写论文的方法部分。

---

## 目录

1. [项目概述与研究背景](#第一章项目概述与研究背景)
2. [3D 高斯场景表示](#第二章3d-高斯场景表示)
3. [X 射线投影模型](#第三章x-射线投影模型)
4. [体渲染与光栅化](#第四章体渲染与光栅化)
5. [损失函数与正则化](#第五章损失函数与正则化)
6. [K-Planes 特征增强（扩展技术）](#第六章k-planes-特征增强扩展技术)
7. [自适应密度控制](#第七章自适应密度控制)
8. [训练流程](#第八章训练流程)
9. [核心公式汇总](#第九章核心公式汇总)
10. [附录：关键符号表](#附录关键符号表)

---

## 第一章：项目概述与研究背景

### 1.1 研究目标

**R²-Gaussian**（Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction）是一个基于 3D 高斯泼溅（3D Gaussian Splatting, 3DGS）的 CT 断层扫描重建方法，已被 **NeurIPS 2024** 接收。

**核心目标**：
- 实现快速、直接的 CT 体积重建
- 支持极稀疏视角（3/6/9 个投影视角）的高质量重建
- 将 3DGS 的高效渲染优势引入医学影像重建领域

### 1.2 CT 断层重建的挑战

传统 CT 重建方法（如 FBP、FDK）在稀疏视角下存在严重的伪影问题。基于深度学习的方法（如 NeRF）虽能改善重建质量，但存在训练和推理速度慢的问题。R²-Gaussian 通过将 X 射线成像物理模型与 3DGS 的高效表示相结合，实现了快速且高质量的稀疏视角 CT 重建。

### 1.3 与标准 3DGS 的核心区别

| 特性 | 标准 3DGS | R²-Gaussian |
|------|----------|-------------|
| **应用场景** | 自然场景新视角合成 | 医学 CT 断层重建 |
| **输出通道** | RGB 颜色（3 通道） | X 射线衰减密度（1 通道） |
| **渲染方程** | Alpha 混合（前后顺序敏感） | 积分求和（顺序无关） |
| **激活函数** | Sigmoid（不透明度 ∈ [0,1]） | Softplus（密度 ∈ [0, +∞)） |
| **投影模式** | 透视投影（固定） | 平行束/锥形束（可切换） |
| **初始化** | COLMAP 稀疏点云 | FDK 重建体积采样 |
| **场景范围** | 任意范围 | 归一化到 [-1, 1]³ |

### 1.4 论文信息

- **会议**：NeurIPS 2024
- **论文标题**：R²-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction
- **arXiv**：https://arxiv.org/abs/2405.20693

---

## 第二章：3D 高斯场景表示

### 2.1 高斯参数化定义

R²-Gaussian 使用一组 3D 高斯基元来表示 CT 体积场景。每个高斯基元由以下参数定义：

| 参数 | 符号 | 维度 | 说明 |
|------|------|------|------|
| 位置 | $\boldsymbol{\mu}$ | $(N, 3)$ | 高斯中心在 3D 空间中的坐标 |
| 缩放 | $\mathbf{s}$ | $(N, 3)$ | 各轴向的缩放因子（存储为 log 形式） |
| 旋转 | $\mathbf{q}$ | $(N, 4)$ | 四元数表示的旋转 |
| 密度 | $\rho$ | $(N, 1)$ | X 射线衰减系数（存储为 log 形式） |

其中 $N$ 为高斯基元的数量。

### 2.2 激活函数设计

为确保参数在物理上有意义，R²-Gaussian 对原始参数应用以下激活函数：

**缩放激活**：
$$\mathbf{s}_{\text{active}} = \exp(\mathbf{s}_{\text{raw}})$$

**旋转激活**（四元数归一化）：
$$\mathbf{q}_{\text{active}} = \frac{\mathbf{q}_{\text{raw}}}{\|\mathbf{q}_{\text{raw}}\|}$$

**密度激活**（Softplus，确保非负）：
$$\rho_{\text{active}} = \text{Softplus}(\rho_{\text{raw}}) = \log(1 + \exp(\rho_{\text{raw}}))$$

> **注意**：与标准 3DGS 使用 Sigmoid 将不透明度限制在 [0,1] 不同，R²-Gaussian 使用 Softplus 允许密度取任意非负值，这与 X 射线衰减系数的物理特性一致。

### 2.3 协方差矩阵构建

3D 高斯的协方差矩阵 $\Sigma$ 由缩放矩阵 $\mathbf{S}$ 和旋转矩阵 $\mathbf{R}$ 构建：

$$\Sigma = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

其中：
- $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$ 为对角缩放矩阵
- $\mathbf{R}$ 为从四元数 $\mathbf{q}$ 计算得到的 $3 \times 3$ 旋转矩阵

### 2.4 场景归一化约定

R²-Gaussian 将所有场景归一化到 $[-1, 1]^3$ 的立方体空间：

$$\text{scene\_scale} = \frac{2}{\max(\text{sVoxel})}$$

其中 sVoxel 为原始 CT 扫描的体积尺寸。所有高斯位置 $\boldsymbol{\mu}$ 在训练前通过此缩放因子进行归一化。

### 2.5 初始化策略

R²-Gaussian 采用基于 FDK 重建的智能初始化策略：

**步骤 1：FDK 体积重建**
使用传统 FDK 算法从稀疏投影重建初始体积 $V_{\text{FDK}}$。

**步骤 2：密度阈值过滤**
$$\mathcal{M} = \{(i,j,k) \mid V_{\text{FDK}}(i,j,k) > \tau_{\text{density}}\}$$

其中 $\tau_{\text{density}}$ 为密度阈值（默认 0.05）。

**步骤 3：点云采样**
从有效体素 $\mathcal{M}$ 中随机采样 $N_{\text{init}}$（默认 50,000）个点作为初始高斯位置。

**步骤 4：缩放初始化**
使用 K 近邻（KNN）距离初始化高斯缩放：
$$s_{\text{init}} = \sqrt{\text{dist}_{\text{KNN}}}$$

**步骤 5：密度初始化**
$$\rho_{\text{init}} = \text{Softplus}^{-1}(V_{\text{FDK}}(i,j,k) \times \alpha_{\text{rescale}})$$

其中 $\alpha_{\text{rescale}}$ 为密度缩放因子（默认 0.15）。

---

## 第三章：X 射线投影模型

### 3.1 辐射传输方程与 Beer-Lambert 定律

X 射线穿过物体时的衰减遵循 Beer-Lambert 定律。对于沿射线 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ 传播的 X 射线：

$$I = I_0 \exp\left(-\int_0^L \mu(\mathbf{r}(t)) \, dt\right)$$

其中：
- $I_0$：入射 X 射线强度
- $I$：透射后的 X 射线强度
- $\mu(\mathbf{x})$：位置 $\mathbf{x}$ 处的线性衰减系数
- $L$：射线穿过物体的路径长度

**对数变换后的投影值**：
$$p = -\ln\frac{I}{I_0} = \int_0^L \mu(\mathbf{r}(t)) \, dt$$

### 3.2 3D 高斯到投影的数学推导

R²-Gaussian 将体积表示为 3D 高斯混合模型：

$$\mu(\mathbf{x}) = \sum_{i=1}^{N} \rho_i \cdot G_i(\mathbf{x})$$

其中 $G_i(\mathbf{x})$ 为第 $i$ 个归一化 3D 高斯函数：

$$G_i(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T \Sigma_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)\right)$$

**投影积分**：
$$p(\mathbf{u}) = \int_{-\infty}^{+\infty} \sum_{i=1}^{N} \rho_i \cdot G_i(\mathbf{o} + t\mathbf{d}) \, dt$$

由于高斯函数沿直线的积分仍为高斯函数（高斯的线性性质），这使得投影计算可以高效进行。

### 3.3 协方差矩阵投影变换

将 3D 协方差矩阵 $\Sigma$ 投影到 2D 屏幕空间需要雅可比变换：

$$\Sigma' = \mathbf{J} \mathbf{W} \Sigma \mathbf{W}^T \mathbf{J}^T$$

其中：
- $\mathbf{W}$：视图变换矩阵（世界坐标到相机坐标）
- $\mathbf{J}$：投影雅可比矩阵（取决于投影模式）

### 3.4 平行束 vs 锥形束投影

R²-Gaussian 支持两种 X 射线投影几何：

**平行束（Parallel Beam）**：

雅可比矩阵：
$$\mathbf{J}_{\text{parallel}} = \begin{pmatrix} f_x & 0 & 0 \\ 0 & f_y & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

**锥形束（Cone Beam）**：

雅可比矩阵：
$$\mathbf{J}_{\text{cone}} = \begin{pmatrix} \frac{f_x}{z} & 0 & -\frac{f_x \cdot x}{z^2} \\ 0 & \frac{f_y}{z} & -\frac{f_y \cdot y}{z^2} \\ \frac{x}{l} & \frac{y}{l} & \frac{z}{l} \end{pmatrix}$$

其中 $(x, y, z)$ 为相机坐标系下的高斯中心位置，$l = \sqrt{x^2 + y^2 + z^2}$。

### 3.5 积分偏差因子 $\mu$

为准确计算 X 射线沿高斯椭球的积分，R²-Gaussian 引入积分偏差因子 $\mu$（论文中的 Eq. 7）：

$$\mu = \sqrt{\frac{2\pi \cdot \text{circ}}{\text{diamond}}}$$

其中：
- $\text{circ}$：高斯在投影方向上的"圆周"度量
- $\text{diamond}$：高斯在投影平面上的"菱形"度量

该因子校正了因高斯椭球形状导致的积分路径差异。

**最终投影值计算**：
$$\alpha_i = \rho_i \cdot \mu_i \cdot \exp\left(-\frac{1}{2}\mathbf{d}^T (\Sigma'_i)^{-1} \mathbf{d}\right)$$

其中 $\mathbf{d}$ 为像素位置到投影高斯中心的 2D 偏移。

---

## 第四章：体渲染与光栅化

### 4.1 体渲染积分公式

R²-Gaussian 的渲染方程与标准 3DGS 有本质区别：

**标准 3DGS（Alpha 混合）**：
$$C = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**R²-Gaussian（积分求和）**：
$$I(\mathbf{u}) = \sum_{i=1}^{N} \alpha_i$$

> **关键区别**：R²-Gaussian 采用简单求和而非 Alpha 混合，因为 X 射线投影是沿射线的线性积分，不存在遮挡关系。这也意味着渲染顺序无关紧要。

### 4.2 Tile-based 光栅化策略

R²-Gaussian 沿用 3DGS 的 Tile-based 光栅化以实现高效并行计算：

**步骤 1：预处理**
- 计算每个高斯的 2D 投影位置和协方差
- 确定屏幕空间包围矩形
- 计算积分偏差因子 $\mu$

**步骤 2：Tile 分配**
- 将屏幕划分为 $16 \times 16$ 像素的 Tile
- 为每个高斯生成 (tile_id, depth) 键值对

**步骤 3：排序**
- 使用 GPU 基数排序按 tile_id 排序

**步骤 4：渲染**
- 每个 CUDA Block 处理一个 Tile
- 每个 Thread 计算一个像素的累积投影值

### 4.3 CUDA 实现的核心 Kernel

| Kernel 名称 | 功能 |
|------------|------|
| `preprocessCUDA` | 计算 3D→2D 投影、屏幕空间协方差、$\mu$ 因子 |
| `renderCUDA` | 主渲染核心，遍历 Tile 内高斯并累积投影值 |
| `checkFrustum` | 视锥体剔除 |
| `duplicateWithKeys` | 生成 Tile 键值对 |
| `identifyTileRanges` | 标识每个 Tile 的高斯范围 |

### 4.4 前向/反向传播流程

**前向传播**：
```
输入: means3D, scales, rotations, densities
  ↓
preprocessCUDA (计算 2D 投影、协方差、μ)
  ↓
排序与 Tile 分配
  ↓
renderCUDA (累积计算投影图像)
  ↓
输出: rendered_image [H, W]
```

**反向传播**：

由于采用简单求和公式，梯度计算较标准 3DGS 简化：

$$\frac{\partial \mathcal{L}}{\partial \alpha_i} = \frac{\partial \mathcal{L}}{\partial I}$$

$$\frac{\partial \mathcal{L}}{\partial \rho_i} = \mu_i \cdot G_i \cdot \frac{\partial \mathcal{L}}{\partial \alpha_i}$$

$$\frac{\partial \mathcal{L}}{\partial \mu_i} = \rho_i \cdot G_i \cdot \frac{\partial \mathcal{L}}{\partial \alpha_i}$$

---

## 第五章：损失函数与正则化

### 5.1 投影损失

**L1 损失**：
$$\mathcal{L}_{\text{L1}} = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{u} \in \mathcal{P}} |I_{\text{pred}}(\mathbf{u}) - I_{\text{gt}}(\mathbf{u})|$$

**结构相似性损失（SSIM）**：
$$\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(I_{\text{pred}}, I_{\text{gt}})$$

**总投影损失**：
$$\mathcal{L}_{\text{proj}} = \mathcal{L}_{\text{L1}} + \lambda_{\text{SSIM}} \cdot \mathcal{L}_{\text{SSIM}}$$

其中 $\lambda_{\text{SSIM}} = 0.25$（默认值）。

### 5.2 3D 总变差（TV）正则化

为促进重建体积的平滑性并减少噪声伪影，R²-Gaussian 引入 3D 总变差正则化：

$$\mathcal{L}_{\text{TV}} = \sum_{i,j,k} \left( |V_{i+1,j,k} - V_{i,j,k}| + |V_{i,j+1,k} - V_{i,j,k}| + |V_{i,j,k+1} - V_{i,j,k}| \right)$$

其中 $V$ 为从高斯场查询得到的 3D 体积。

**实现细节**：
- 在训练过程中定期查询 3D 体积
- TV 损失权重 $\lambda_{\text{TV}} = 0.05$（默认值）

### 5.3 K-Planes TV 正则化（扩展）

当启用 K-Planes 特征增强时，对三个特征平面施加 TV 正则化：

$$\mathcal{L}_{\text{plane-TV}} = \sum_{p \in \{xy, xz, yz\}} w_p \cdot \text{TV}(P_p)$$

其中：
- $P_p$ 为平面 $p$ 的特征图
- $w_p$ 为对应权重（默认 $[0.0001, 0.0001, 0.0001]$）

**平面 TV 计算**：
$$\text{TV}(P) = \frac{2}{H \cdot W} \left( \sum_{i,j} (P_{i+1,j} - P_{i,j})^2 + \sum_{i,j} (P_{i,j+1} - P_{i,j})^2 \right)$$

### 5.4 双目一致性损失（扩展）

为提高稀疏视角下的几何一致性，引入基于角度偏移的双目一致性损失：

**原理**：对于给定的训练视角 $\theta$，生成相邻角度 $\theta \pm \Delta\theta$ 的"虚拟视角"投影，并约束它们之间的一致性。

$$\mathcal{L}_{\text{bino}} = \frac{1}{2} \left( \|I_{\theta+\Delta\theta} - \text{Warp}(I_\theta, \Delta\theta)\|_1 + \|I_{\theta-\Delta\theta} - \text{Warp}(I_\theta, -\Delta\theta)\|_1 \right)$$

其中 $\text{Warp}$ 为基于深度的视图变换函数。

### 5.5 总损失函数

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{proj}} + \lambda_{\text{TV}} \cdot \mathcal{L}_{\text{TV}} + \lambda_{\text{plane-TV}} \cdot \mathcal{L}_{\text{plane-TV}} + \lambda_{\text{bino}} \cdot \mathcal{L}_{\text{bino}}$$

---

## 第六章：K-Planes 特征增强（扩展技术）

### 6.1 动机

原始 R²-Gaussian 为每个高斯基元分配独立的密度参数，难以捕捉空间相关性。K-Planes 特征增强通过引入基于位置的特征编码，使相邻高斯能够共享空间信息，从而提高重建质量。

### 6.2 K-Planes 编码器架构

K-Planes 编码器使用三个正交的 2D 特征平面表示 3D 空间：

| 平面 | 符号 | 维度 | 说明 |
|------|------|------|------|
| XY 平面 | $P_{xy}$ | $(1, C, H, W)$ | 俯视图特征 |
| XZ 平面 | $P_{xz}$ | $(1, C, H, W)$ | 前视图特征 |
| YZ 平面 | $P_{yz}$ | $(1, C, H, W)$ | 侧视图特征 |

默认参数：
- 特征维度 $C = 32$
- 分辨率 $H = W = 64$

**特征提取**：
对于位置 $\mathbf{x} = (x, y, z)$：

$$f_{xy} = \text{BilinearSample}(P_{xy}, (x, y))$$
$$f_{xz} = \text{BilinearSample}(P_{xz}, (x, z))$$
$$f_{yz} = \text{BilinearSample}(P_{yz}, (y, z))$$

**特征拼接**：
$$\mathbf{f}_{\text{kplanes}} = [f_{xy}; f_{xz}; f_{yz}] \in \mathbb{R}^{96}$$

### 6.3 MLP 密度解码器

将 K-Planes 特征映射为密度调制因子：

$$\delta_\rho = \text{MLP}(\mathbf{f}_{\text{kplanes}})$$

**MLP 结构**：
```
输入: [N, 96]
  ↓
Linear(96, 128) + ReLU
  ↓
Linear(128, 128) + ReLU
  ↓
Linear(128, 1) + Tanh
  ↓
输出: [N, 1]，范围 [-1, 1]
```

### 6.4 密度调制公式

K-Planes 特征通过乘性调制影响最终密度：

$$\rho_{\text{final}} = \rho_{\text{base}} \cdot m(\delta_\rho)$$

其中调制函数：
$$m(\delta_\rho) = 0.7 + 0.6 \cdot \sigma(\delta_\rho)$$

由于 $\sigma(\cdot) \in (0, 1)$，因此：
$$m(\delta_\rho) \in (0.7, 1.3)$$

这意味着 K-Planes 可以对基础密度进行 ±30% 的调制。

### 6.5 训练策略与超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_kplanes` | False | 是否启用 K-Planes |
| `kplanes_resolution` | 64 | 平面分辨率 |
| `kplanes_dim` | 32 | 特征维度 |
| `kplanes_lr_init` | 0.001 | 初始学习率 |
| `kplanes_lr_final` | 0.00001 | 最终学习率 |
| `lambda_plane_tv` | 0.002 | 平面 TV 权重 |

---

## 第七章：自适应密度控制

### 7.1 概述

R²-Gaussian 沿用 3DGS 的自适应密度控制机制，根据训练过程中的梯度信息动态调整高斯基元的数量和分布。

### 7.2 密化（Densification）策略

#### 7.2.1 Clone（克隆）

**触发条件**：
- 位置梯度超过阈值：$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}}$
- 高斯尺度较小：$\max(\mathbf{s}) < \tau_{\text{scale}}$

**操作**：复制高斯基元，沿梯度方向偏移新位置。

#### 7.2.2 Split（分割）

**触发条件**：
- 位置梯度超过阈值：$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}}$
- 高斯尺度较大：$\max(\mathbf{s}) > \tau_{\text{scale}}$

**操作**：将高斯分割为两个更小的高斯，新高斯的尺度为原来的 $1/\phi$（默认 $\phi = 1.6$）。

### 7.3 剪枝（Pruning）策略

**触发条件**（满足任一）：
- 密度过低：$\rho < \tau_{\rho}$
- 尺度过大：$\max(\mathbf{s}) > \tau_{\text{world}}$（超出场景边界）

**操作**：移除满足条件的高斯基元。

### 7.4 GAR 邻近感知密化（扩展）

标准密化策略可能导致高斯聚集。GAR（Gaussian-Aware Refinement）引入邻近感知机制：

**邻近惩罚**：
对于距离过近的高斯对 $(i, j)$：
$$\text{penalty}_{ij} = \exp\left(-\frac{\|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|^2}{2\sigma^2}\right)$$

当惩罚超过阈值时，优先选择梯度较小的高斯进行剪枝或合并。

### 7.5 控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `densify_from_iter` | 500 | 开始密化的迭代 |
| `densify_until_iter` | 15000 | 停止密化的迭代 |
| `densification_interval` | 100 | 密化间隔 |
| `densify_grad_threshold` | 2.0e-4 | 梯度阈值 |
| `opacity_reset_interval` | 3000 | 不透明度重置间隔 |

---

## 第八章：训练流程

### 8.1 数据加载与预处理

**输入数据格式**：
- CT 投影图像：$(N_{\text{views}}, H, W)$
- 投影角度：$(N_{\text{views}},)$
- 扫描几何参数：源到探测器距离、体素尺寸等

**预处理步骤**：
1. 加载投影图像和几何参数
2. 场景归一化到 $[-1, 1]^3$
3. 生成初始点云（FDK + 采样）
4. 初始化高斯参数

### 8.2 优化器配置与学习率调度

**参数分组**：

| 参数组 | 初始学习率 | 最终学习率 | 调度策略 |
|--------|-----------|-----------|----------|
| 位置 $\boldsymbol{\mu}$ | 0.0002 | 0.0000002 | 指数衰减 |
| 缩放 $\mathbf{s}$ | 0.005 | - | 固定 |
| 旋转 $\mathbf{q}$ | 0.001 | - | 固定 |
| 密度 $\rho$ | 0.01 | - | 固定 |
| K-Planes | 0.001 | 0.00001 | 指数衰减 |

**学习率调度**（指数衰减）：
$$\text{lr}(t) = \text{lr}_{\text{init}} \cdot \left(\frac{\text{lr}_{\text{final}}}{\text{lr}_{\text{init}}}\right)^{t / T}$$

### 8.3 训练循环伪代码

```
输入: 投影图像 {I_gt}, 角度 {θ}, 几何参数
输出: 训练好的高斯模型

初始化:
  gaussians = 从 FDK 点云初始化()
  optimizer = Adam(gaussians.parameters())

for iter in range(max_iterations):
    # 1. 学习率更新
    update_learning_rate(iter)

    # 2. 随机选择视角
    view_idx = random.choice(num_views)
    camera = get_camera(θ[view_idx])

    # 3. 渲染投影图像
    I_pred = render(gaussians, camera)

    # 4. 计算损失
    loss = L1(I_pred, I_gt[view_idx])
    loss += λ_ssim * (1 - SSIM(I_pred, I_gt[view_idx]))

    if use_tv_regularization:
        V = query_volume(gaussians)
        loss += λ_tv * TV_3D(V)

    if use_kplanes:
        loss += λ_plane_tv * plane_tv_loss(gaussians.kplanes)

    # 5. 反向传播
    loss.backward()

    # 6. 自适应密度控制
    if densify_from_iter < iter < densify_until_iter:
        if iter % densification_interval == 0:
            gaussians.densify_and_prune(grad_threshold)

    # 7. 参数更新
    optimizer.step()
    optimizer.zero_grad()

return gaussians
```

### 8.4 评估指标

**2D 投影质量**：
- **PSNR**：$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$
- **SSIM**：基于局部均值、方差和协方差的结构相似性

**3D 体积质量**：
- 逐切片计算 PSNR 和 SSIM
- 沿三个轴向（X、Y、Z）分别评估后取平均

---

## 第九章：核心公式汇总

本章汇总论文方法部分可能用到的所有核心公式，便于直接引用。

### 9.1 3D 高斯表示

**高斯函数**：
$$G(\mathbf{x}; \boldsymbol{\mu}, \Sigma) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

**协方差矩阵**：
$$\Sigma = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

**密度激活**：
$$\rho = \text{Softplus}(\rho_{\text{raw}}) = \log(1 + e^{\rho_{\text{raw}}})$$

### 9.2 X 射线投影

**Beer-Lambert 定律**：
$$I = I_0 \exp\left(-\int_0^L \mu(\mathbf{r}(t)) \, dt\right)$$

**投影值**：
$$p(\mathbf{u}) = -\ln\frac{I}{I_0} = \int_{-\infty}^{+\infty} \mu(\mathbf{r}(t)) \, dt$$

**高斯投影贡献**：
$$\alpha_i(\mathbf{u}) = \rho_i \cdot \mu_i \cdot \exp\left(-\frac{1}{2}\mathbf{d}^T (\Sigma'_i)^{-1} \mathbf{d}\right)$$

**总投影（R²-Gaussian 渲染方程）**：
$$I(\mathbf{u}) = \sum_{i=1}^{N} \alpha_i(\mathbf{u})$$

### 9.3 协方差变换

**世界到相机坐标**：
$$\Sigma_{\text{cam}} = \mathbf{W} \Sigma \mathbf{W}^T$$

**相机到屏幕坐标（带雅可比）**：
$$\Sigma' = \mathbf{J} \Sigma_{\text{cam}} \mathbf{J}^T$$

### 9.4 损失函数

**L1 损失**：
$$\mathcal{L}_{\text{L1}} = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{u} \in \mathcal{P}} |I_{\text{pred}}(\mathbf{u}) - I_{\text{gt}}(\mathbf{u})|$$

**SSIM 损失**：
$$\mathcal{L}_{\text{SSIM}} = 1 - \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

**3D TV 正则化**：
$$\mathcal{L}_{\text{TV}} = \sum_{i,j,k} \left( |V_{i+1,j,k} - V_{i,j,k}| + |V_{i,j+1,k} - V_{i,j,k}| + |V_{i,j,k+1} - V_{i,j,k}| \right)$$

**总损失**：
$$\mathcal{L} = \mathcal{L}_{\text{L1}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}}$$

### 9.5 K-Planes 密度调制

**特征提取**：
$$\mathbf{f} = [P_{xy}(x,y); P_{xz}(x,z); P_{yz}(y,z)] \in \mathbb{R}^{3C}$$

**密度调制**：
$$\rho_{\text{final}} = \rho_{\text{base}} \cdot \left(0.7 + 0.6 \cdot \sigma(\text{MLP}(\mathbf{f}))\right)$$

### 9.6 自适应密度控制

**克隆条件**：
$$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}} \quad \text{且} \quad \max(\mathbf{s}) < \tau_{\text{scale}}$$

**分割条件**：
$$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}} \quad \text{且} \quad \max(\mathbf{s}) > \tau_{\text{scale}}$$

---

## 附录：关键符号表

| 符号 | 含义 | 维度/范围 |
|------|------|----------|
| $N$ | 高斯基元数量 | 标量 |
| $\boldsymbol{\mu}$ | 高斯中心位置 | $\mathbb{R}^3$ |
| $\Sigma$ | 3D 协方差矩阵 | $\mathbb{R}^{3 \times 3}$ |
| $\Sigma'$ | 2D 投影协方差矩阵 | $\mathbb{R}^{2 \times 2}$ |
| $\mathbf{s}$ | 缩放因子 | $\mathbb{R}^3_+$ |
| $\mathbf{q}$ | 旋转四元数 | $\mathbb{R}^4$，$\|\mathbf{q}\|=1$ |
| $\mathbf{R}$ | 旋转矩阵 | $\mathbb{R}^{3 \times 3}$，$\mathbf{R}^T\mathbf{R}=\mathbf{I}$ |
| $\mathbf{S}$ | 缩放矩阵 | $\mathbb{R}^{3 \times 3}$，对角矩阵 |
| $\rho$ | X 射线衰减密度 | $\mathbb{R}_+$ |
| $\mu$ | 积分偏差因子 | $\mathbb{R}_+$ |
| $\alpha$ | 像素投影贡献 | $\mathbb{R}_+$ |
| $I(\mathbf{u})$ | 投影图像像素值 | $\mathbb{R}_+$ |
| $\mathbf{W}$ | 视图变换矩阵 | $\mathbb{R}^{3 \times 3}$ |
| $\mathbf{J}$ | 投影雅可比矩阵 | $\mathbb{R}^{3 \times 3}$ |
| $P_{xy}, P_{xz}, P_{yz}$ | K-Planes 特征平面 | $\mathbb{R}^{C \times H \times W}$ |
| $\tau_{\text{grad}}$ | 密化梯度阈值 | 默认 $2 \times 10^{-4}$ |
| $\lambda_{\text{SSIM}}$ | SSIM 损失权重 | 默认 0.25 |
| $\lambda_{\text{TV}}$ | TV 正则化权重 | 默认 0.05 |

---

## 文档信息

- **生成日期**：2024年12月
- **基于代码版本**：R²-Gaussian（integration 分支）
- **目标用途**：论文方法部分撰写参考
