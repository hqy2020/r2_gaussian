# SPAGS 技术文档：论文方法撰写指南

> 本文档旨在让读者无需阅读源代码即可理解 **SPAGS**（Spatial-aware Progressive Adaptive Gaussian Splatting）的核心算法实现，以便准确撰写论文的方法部分。

---

## SPAGS 框架概览

**SPAGS** 是一个基于 R²-Gaussian 的三阶段渐进式优化框架，包含三个核心创新模块：

```
┌─────────────────────────────────────────────────────────────────┐
│                        SPAGS 框架                                │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: SPS (Spatial Prior Seeding)                           │
│           空间先验播种 → 密度加权初始化                           │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: GAR (Geometry-Aware Refinement)                       │
│           几何感知细化 → 双目一致性 + 邻近密度反池化              │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: ADM (Adaptive Density Modulation)                     │
│           自适应密度调制 → K-Planes 特征编码 + MLP 密度调制       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录

1. [项目概述与研究背景](#第一章项目概述与研究背景)
2. [3D 高斯场景表示](#第二章3d-高斯场景表示)
3. [X 射线投影模型](#第三章x-射线投影模型)
4. [体渲染与光栅化](#第四章体渲染与光栅化)
5. [损失函数与正则化](#第五章损失函数与正则化)
6. [**SPS：空间先验播种**](#第六章sps空间先验播种)
7. [**GAR：几何感知细化**](#第七章gar几何感知细化)
8. [**ADM：自适应密度调制**](#第八章adm自适应密度调制)
9. [自适应密度控制](#第九章自适应密度控制)
10. [训练流程](#第十章训练流程)
11. [核心公式汇总](#第十一章核心公式汇总)
12. [附录：关键符号表](#附录关键符号表)

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

## 第六章：SPS（空间先验播种）

### 6.1 动机与核心思想

传统 R²-Gaussian 使用均匀随机采样从 FDK 重建体积初始化高斯点云，这种方式忽略了 CT 体积中的密度分布先验。**SPS（Spatial Prior Seeding，空间先验播种）** 通过密度加权采样策略，使初始高斯点云的分布与实际组织密度分布一致。

**核心思想**：
- 高密度区域（骨骼、造影剂）采样更多点 → 细节保留
- 低密度区域（空气、软组织）采样较少点 → 避免噪声放大

### 6.2 密度加权采样算法

给定 FDK 重建体积 $V_{\text{FDK}} \in \mathbb{R}^{D \times H \times W}$：

**步骤 1：有效体素筛选**
$$\mathcal{M} = \{(i,j,k) \mid V_{\text{FDK}}(i,j,k) > \tau_{\text{density}}\}$$

**步骤 2：计算采样概率**
$$P(\mathbf{x}) = \frac{\rho(\mathbf{x})}{\sum_{\mathbf{x}' \in \mathcal{M}} \rho(\mathbf{x}')}$$

其中 $\rho(\mathbf{x}) = V_{\text{FDK}}(\mathbf{x})$ 为体素密度值。

**步骤 3：加权采样**
$$\{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N\} \sim \text{Multinomial}(\mathcal{M}, P, N_{\text{init}})$$

采样 $N_{\text{init}}$（默认 50,000）个点作为初始高斯位置。

### 6.3 降噪预处理

FDK 在稀疏视角下产生的重建噪声会影响采样质量。SPS 可选地在采样前进行高斯滤波降噪：

$$V_{\text{smooth}} = V_{\text{FDK}} * G_\sigma$$

其中 $G_\sigma$ 为标准差为 $\sigma$（默认 3.0）的 3D 高斯核。

### 6.4 SPS 公式汇总

**采样概率**：
$$P_i = \frac{\rho_i}{\sum_{j=1}^{|\mathcal{M}|} \rho_j}$$

**期望采样密度**：
$$\mathbb{E}[n_i] = N_{\text{init}} \cdot P_i$$

### 6.5 SPS 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_sps` | False | 是否启用 SPS |
| `sps_strategy` | "density_weighted" | 采样策略 |
| `sps_denoise` | False | 是否启用降噪 |
| `sps_denoise_sigma` | 3.0 | 高斯核标准差 |
| `density_thresh` | 0.05 | 密度阈值 |
| `n_points` | 50000 | 采样点数 |

---

## 第七章：GAR（几何感知细化）

GAR（Geometry-Aware Refinement，几何感知细化）包含两个协同工作的子模块：
1. **双目一致性损失**：利用虚拟视角约束几何一致性
2. **邻近密度反池化**：基于邻域结构的智能密化策略

### 7.1 双目一致性损失

#### 7.1.1 动机

稀疏视角 CT 重建存在严重的视角间几何不一致问题。双目一致性损失通过生成"虚拟立体视角对"，约束相邻视角之间的几何一致性。

#### 7.1.2 虚拟视角生成

对于训练视角 $\theta$，生成角度偏移为 $\pm\Delta\theta$ 的虚拟视角：

**旋转矩阵更新**：
$$\mathbf{R}_{\text{shifted}} = \begin{pmatrix} \cos\Delta\theta & 0 & \sin\Delta\theta \\ 0 & 1 & 0 \\ -\sin\Delta\theta & 0 & \cos\Delta\theta \end{pmatrix} \cdot \mathbf{R}_{\text{original}}$$

**相机位置更新**：
$$\mathbf{T}_{\text{shifted}} = \begin{pmatrix} d \cdot \sin\Delta\theta \\ T_y \\ -d \cdot \cos\Delta\theta \end{pmatrix}$$

其中 $d$ 为相机到场景中心的距离。

#### 7.1.3 深度估计

使用加权平均法估计像素深度：

$$Z(\mathbf{u}) = \frac{\sum_{i=1}^{N} \alpha_i(\mathbf{u}) \cdot z_i}{\sum_{i=1}^{N} \alpha_i(\mathbf{u})}$$

其中 $z_i$ 为第 $i$ 个高斯基元在相机坐标系下的深度，$\alpha_i(\mathbf{u})$ 为其对像素 $\mathbf{u}$ 的贡献。

#### 7.1.4 视差计算与图像变换

**视差公式**（双目立体几何）：
$$d(\mathbf{u}) = \frac{f \cdot B}{Z(\mathbf{u})}$$

其中：
- $f$：焦距
- $B$：基线（由角度偏移决定）
- $Z(\mathbf{u})$：像素深度

**图像 Warp**：
$$I_{\text{warped}}(\mathbf{u}) = I_{\text{source}}\left(\mathbf{u} + \begin{pmatrix} d(\mathbf{u}) \\ 0 \end{pmatrix}\right)$$

使用可微的 `grid_sample` 实现双线性插值。

#### 7.1.5 双目一致性损失公式

$$\mathcal{L}_{\text{bino}} = \|I_{\theta} - \text{Warp}(I_{\theta+\Delta\theta}, -d)\|_1 + \|I_{\theta+\Delta\theta} - \text{Warp}(I_{\theta}, d)\|_1$$

**带 Warmup 的完整损失**：
$$\mathcal{L}_{\text{GAR-bino}} = \mathcal{L}_{\text{bino}} + w_{\text{warmup}}(t) \cdot \mathcal{L}_{\text{smooth}}(Z)$$

其中 $w_{\text{warmup}}(t)$ 随迭代渐增，初期强调图像一致性，后期加入深度平滑约束。

### 7.2 邻近密度反池化

#### 7.2.1 动机

标准 3DGS 的 Clone/Split 策略仅依赖梯度信息，容易导致高斯点聚集在高梯度区域，而忽略稀疏但重要的区域。**邻近密度反池化**通过分析高斯点的空间邻域结构，在稀疏区域智能生成新的高斯点。

#### 7.2.2 邻近度分数计算

对于每个高斯基元 $i$，计算其邻近度分数（Proximity Score）：

$$P_i = \frac{1}{K} \sum_{j \in \mathcal{N}_K(i)} \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2$$

其中 $\mathcal{N}_K(i)$ 为高斯 $i$ 的 $K$ 近邻集合。

**物理意义**：
- $P_i$ **高** → 高斯点"孤立"，位于稀疏区域 → 需要密化
- $P_i$ **低** → 高斯点"拥挤"，位于密集区域 → 无需密化

#### 7.2.3 密化候选识别

$$\mathcal{D} = \{i \mid P_i > \tau_{\text{proximity}}\}$$

其中 $\tau_{\text{proximity}}$ 为邻近度阈值（默认 5.0）。

#### 7.2.4 邻居中点插值（反池化核心）

对于每个待密化的高斯 $i \in \mathcal{D}$，在其与邻居之间生成新高斯：

$$\boldsymbol{\mu}_{\text{new}}^{(i,j)} = \frac{\boldsymbol{\mu}_i + \boldsymbol{\mu}_j}{2}, \quad \forall j \in \mathcal{N}_K(i)$$

**新高斯属性继承**：
- **位置**：邻居中点
- **缩放**：继承自邻居高斯
- **密度**：继承自邻居高斯
- **旋转**：初始化为单位四元数

这种"反池化"操作在稀疏区域插入新点，类似于上采样操作。

#### 7.2.5 医学约束（可选）

针对 CT 数据的组织特性，为不同组织类型应用差异化的密化参数：

| 组织类型 | 密度范围 | 邻近度阈值 | K 近邻数 | 物理含义 |
|---------|---------|-----------|---------|---------|
| 背景空气 | $[0, 0.05)$ | 2.0（严格） | 6 | 空气区域，避免噪声 |
| 组织边界 | $[0.05, 0.15)$ | 1.5（最严格） | 8 | 诊断关键区域 |
| 软组织 | $[0.15, 0.40)$ | 1.0（适中） | 6 | 脏器、肌肉 |
| 高密度结构 | $[0.40, 1.0]$ | 0.8（宽松） | 4 | 骨骼、钙化 |

### 7.3 GAR 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_gar` | False | GAR 主开关 |
| `gar_loss_weight` | 0.08 | 双目损失权重 |
| `gar_max_angle` | 0.04 | 最大角度偏移（弧度） |
| `gar_start_iter` | 5000 | 起始迭代 |
| `gar_warmup_iters` | 3000 | Warmup 迭代数 |
| `enable_gar_proximity` | False | 邻近密化开关 |
| `gar_proximity_threshold` | 5.0 | 邻近度阈值 |
| `gar_proximity_k` | 5 | K 近邻数 |
| `gar_medical_constraints` | True | 医学约束开关 |

---

## 第八章：ADM（自适应密度调制）

### 8.1 动机与核心思想

原始 R²-Gaussian 为每个高斯基元分配独立的密度参数，难以捕捉空间相关性。**ADM（Adaptive Density Modulation，自适应密度调制）** 通过 K-Planes 空间编码和 MLP 解码，使相邻高斯能够共享空间信息，实现密度的自适应调制。

### 8.2 K-Planes 空间编码

K-Planes 将 3D 空间分解为三个正交的 2D 特征平面：

| 平面 | 符号 | 维度 | 说明 |
|------|------|------|------|
| XY 平面 | $P_{xy}$ | $(1, C, H, W)$ | 轴向切面特征 |
| XZ 平面 | $P_{xz}$ | $(1, C, H, W)$ | 冠状切面特征 |
| YZ 平面 | $P_{yz}$ | $(1, C, H, W)$ | 矢状切面特征 |

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

### 8.3 MLP 密度解码器

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

### 8.4 密度调制公式

K-Planes 特征通过乘性调制影响最终密度：

$$\rho_{\text{final}} = \rho_{\text{base}} \cdot m(\delta_\rho)$$

其中调制函数：
$$m(\delta_\rho) = 0.7 + 0.6 \cdot \sigma(\delta_\rho)$$

由于 $\sigma(\cdot) \in (0, 1)$，因此：
$$m(\delta_\rho) \in (0.7, 1.3)$$

这意味着 ADM 可以对基础密度进行 **±30%** 的调制。

**调制函数的数学性质**：
- 当 $\delta_\rho \to -\infty$ 时，$m \to 0.7$（↓30%）
- 当 $\delta_\rho = 0$ 时，$m = 1.0$（不调制）
- 当 $\delta_\rho \to +\infty$ 时，$m \to 1.3$（↑30%）

### 8.5 Plane TV 正则化

为防止 K-Planes 过拟合，对特征平面施加总变差（TV）正则化：

$$\mathcal{L}_{\text{plane-TV}} = \sum_{p \in \{xy, xz, yz\}} w_p \cdot \text{TV}(P_p)$$

**平面 TV 计算**：
$$\text{TV}(P) = \frac{2}{H \cdot W} \left( \sum_{i,j} (P_{i+1,j} - P_{i,j})^2 + \sum_{i,j} (P_{i,j+1} - P_{i,j})^2 \right)$$

### 8.6 ADM 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_adm` | False | ADM 主开关 |
| `adm_resolution` | 64 | 平面分辨率 |
| `adm_feature_dim` | 32 | 特征维度 |
| `adm_decoder_hidden` | 128 | MLP 隐层维度 |
| `adm_decoder_layers` | 3 | MLP 层数 |
| `adm_lr_init` | 0.002 | 初始学习率 |
| `adm_lr_final` | 0.0002 | 最终学习率 |
| `adm_lambda_tv` | 0.002 | Plane TV 权重 |

---

## 第九章：自适应密度控制

### 9.1 概述

R²-Gaussian 沿用 3DGS 的自适应密度控制机制，根据训练过程中的梯度信息动态调整高斯基元的数量和分布。SPAGS 在此基础上增加了邻近密度反池化（见第七章 GAR）。

### 9.2 密化（Densification）策略

#### 9.2.1 Clone（克隆）

**触发条件**：
- 位置梯度超过阈值：$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}}$
- 高斯尺度较小：$\max(\mathbf{s}) < \tau_{\text{scale}}$

**操作**：复制高斯基元，沿梯度方向偏移新位置。

#### 9.2.2 Split（分割）

**触发条件**：
- 位置梯度超过阈值：$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}}$
- 高斯尺度较大：$\max(\mathbf{s}) > \tau_{\text{scale}}$

**操作**：将高斯分割为两个更小的高斯，新高斯的尺度为原来的 $1/\phi$（默认 $\phi = 1.6$）。

### 9.3 剪枝（Pruning）策略

**触发条件**（满足任一）：
- 密度过低：$\rho < \tau_{\rho}$
- 尺度过大：$\max(\mathbf{s}) > \tau_{\text{world}}$（超出场景边界）

**操作**：移除满足条件的高斯基元。

### 9.4 控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `densify_from_iter` | 500 | 开始密化的迭代 |
| `densify_until_iter` | 15000 | 停止密化的迭代 |
| `densification_interval` | 100 | 密化间隔 |
| `densify_grad_threshold` | 2.0e-4 | 梯度阈值 |
| `opacity_reset_interval` | 3000 | 不透明度重置间隔 |

---

## 第十章：训练流程

### 10.1 数据加载与预处理

**输入数据格式**：
- CT 投影图像：$(N_{\text{views}}, H, W)$
- 投影角度：$(N_{\text{views}},)$
- 扫描几何参数：源到探测器距离、体素尺寸等

**预处理步骤**：
1. 加载投影图像和几何参数
2. 场景归一化到 $[-1, 1]^3$
3. 生成初始点云（SPS 密度加权采样 或 均匀随机采样）
4. 初始化高斯参数

### 10.2 优化器配置与学习率调度

**参数分组**：

| 参数组 | 初始学习率 | 最终学习率 | 调度策略 |
|--------|-----------|-----------|----------|
| 位置 $\boldsymbol{\mu}$ | 0.0002 | 0.0000002 | 指数衰减 |
| 缩放 $\mathbf{s}$ | 0.005 | - | 固定 |
| 旋转 $\mathbf{q}$ | 0.001 | - | 固定 |
| 密度 $\rho$ | 0.01 | - | 固定 |
| K-Planes (ADM) | 0.002 | 0.0002 | 指数衰减 |

**学习率调度**（指数衰减）：
$$\text{lr}(t) = \text{lr}_{\text{init}} \cdot \left(\frac{\text{lr}_{\text{final}}}{\text{lr}_{\text{init}}}\right)^{t / T}$$

### 10.3 SPAGS 训练循环伪代码

```
输入: 投影图像 {I_gt}, 角度 {θ}, 几何参数
输出: 训练好的高斯模型

# ========== Stage 1: SPS 初始化 ==========
if enable_sps:
    point_cloud = density_weighted_sampling(FDK_volume)
else:
    point_cloud = uniform_random_sampling(FDK_volume)

gaussians = initialize_from_point_cloud(point_cloud)
optimizer = Adam(gaussians.parameters())

# ========== Stage 2 & 3: GAR + ADM 训练 ==========
for iter in range(max_iterations):
    # 1. 学习率更新
    update_learning_rate(iter)

    # 2. 随机选择视角
    view_idx = random.choice(num_views)
    camera = get_camera(θ[view_idx])

    # 3. 渲染投影图像
    I_pred = render(gaussians, camera)

    # 4. 计算基础损失
    loss = L1(I_pred, I_gt[view_idx])
    loss += λ_ssim * (1 - SSIM(I_pred, I_gt[view_idx]))

    # 5. TV 正则化
    if use_tv_regularization:
        V = query_volume(gaussians)
        loss += λ_tv * TV_3D(V)

    # 6. GAR: 双目一致性损失
    if enable_gar and iter > gar_start_iter:
        shifted_cam = create_shifted_camera(camera, Δθ)
        I_shifted = render(gaussians, shifted_cam)
        depth = estimate_depth(gaussians, camera)
        loss += λ_gar * binocular_loss(I_pred, I_shifted, depth)

    # 7. ADM: K-Planes TV 正则化
    if enable_adm:
        loss += λ_plane_tv * plane_tv_loss(gaussians.kplanes)

    # 8. 反向传播
    loss.backward()

    # 9. 自适应密度控制 (含 GAR 邻近密化)
    if densify_from_iter < iter < densify_until_iter:
        if iter % densification_interval == 0:
            gaussians.densify_and_prune(grad_threshold)

            if enable_gar_proximity:
                proximity_scores = compute_proximity_scores(gaussians)
                gaussians.proximity_densify(proximity_scores)

    # 10. 参数更新
    optimizer.step()
    optimizer.zero_grad()

return gaussians
```

### 10.4 评估指标

**2D 投影质量**：
- **PSNR**：$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$
- **SSIM**：基于局部均值、方差和协方差的结构相似性

**3D 体积质量**：
- 逐切片计算 PSNR 和 SSIM
- 沿三个轴向（X、Y、Z）分别评估后取平均

---

## 第十一章：核心公式汇总

本章汇总论文方法部分可能用到的所有核心公式，便于直接引用。

### 11.1 3D 高斯表示

**高斯函数**：
$$G(\mathbf{x}; \boldsymbol{\mu}, \Sigma) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

**协方差矩阵**：
$$\Sigma = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

**密度激活**：
$$\rho = \text{Softplus}(\rho_{\text{raw}}) = \log(1 + e^{\rho_{\text{raw}}})$$

### 11.2 X 射线投影

**Beer-Lambert 定律**：
$$I = I_0 \exp\left(-\int_0^L \mu(\mathbf{r}(t)) \, dt\right)$$

**投影值**：
$$p(\mathbf{u}) = -\ln\frac{I}{I_0} = \int_{-\infty}^{+\infty} \mu(\mathbf{r}(t)) \, dt$$

**高斯投影贡献**：
$$\alpha_i(\mathbf{u}) = \rho_i \cdot \mu_i \cdot \exp\left(-\frac{1}{2}\mathbf{d}^T (\Sigma'_i)^{-1} \mathbf{d}\right)$$

**总投影（R²-Gaussian 渲染方程）**：
$$I(\mathbf{u}) = \sum_{i=1}^{N} \alpha_i(\mathbf{u})$$

### 11.3 协方差变换

**世界到相机坐标**：
$$\Sigma_{\text{cam}} = \mathbf{W} \Sigma \mathbf{W}^T$$

**相机到屏幕坐标（带雅可比）**：
$$\Sigma' = \mathbf{J} \Sigma_{\text{cam}} \mathbf{J}^T$$

### 11.4 基础损失函数

**L1 损失**：
$$\mathcal{L}_{\text{L1}} = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{u} \in \mathcal{P}} |I_{\text{pred}}(\mathbf{u}) - I_{\text{gt}}(\mathbf{u})|$$

**SSIM 损失**：
$$\mathcal{L}_{\text{SSIM}} = 1 - \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

**3D TV 正则化**：
$$\mathcal{L}_{\text{TV}} = \sum_{i,j,k} \left( |V_{i+1,j,k} - V_{i,j,k}| + |V_{i,j+1,k} - V_{i,j,k}| + |V_{i,j,k+1} - V_{i,j,k}| \right)$$

### 11.5 SPS：空间先验播种

**密度加权采样概率**：
$$P_i = \frac{\rho_i}{\sum_{j=1}^{|\mathcal{M}|} \rho_j}$$

**高斯滤波降噪**：
$$V_{\text{smooth}} = V_{\text{FDK}} * G_\sigma$$

### 11.6 GAR：双目一致性损失

**虚拟视角旋转矩阵**：
$$\mathbf{R}_{\text{shifted}} = \begin{pmatrix} \cos\Delta\theta & 0 & \sin\Delta\theta \\ 0 & 1 & 0 \\ -\sin\Delta\theta & 0 & \cos\Delta\theta \end{pmatrix} \cdot \mathbf{R}_{\text{original}}$$

**深度估计（加权平均）**：
$$Z(\mathbf{u}) = \frac{\sum_{i=1}^{N} \alpha_i(\mathbf{u}) \cdot z_i}{\sum_{i=1}^{N} \alpha_i(\mathbf{u})}$$

**视差公式**：
$$d(\mathbf{u}) = \frac{f \cdot B}{Z(\mathbf{u})}$$

**双目一致性损失**：
$$\mathcal{L}_{\text{bino}} = \|I_{\theta} - \text{Warp}(I_{\theta+\Delta\theta}, -d)\|_1 + \|I_{\theta+\Delta\theta} - \text{Warp}(I_{\theta}, d)\|_1$$

### 11.7 GAR：邻近密度反池化

**邻近度分数**：
$$P_i = \frac{1}{K} \sum_{j \in \mathcal{N}_K(i)} \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|_2$$

**密化候选集**：
$$\mathcal{D} = \{i \mid P_i > \tau_{\text{proximity}}\}$$

**邻居中点插值（反池化核心）**：
$$\boldsymbol{\mu}_{\text{new}}^{(i,j)} = \frac{\boldsymbol{\mu}_i + \boldsymbol{\mu}_j}{2}, \quad \forall j \in \mathcal{N}_K(i)$$

### 11.8 ADM：自适应密度调制

**K-Planes 特征提取**：
$$\mathbf{f} = [P_{xy}(x,y); P_{xz}(x,z); P_{yz}(y,z)] \in \mathbb{R}^{3C}$$

**密度调制**：
$$\rho_{\text{final}} = \rho_{\text{base}} \cdot \left(0.7 + 0.6 \cdot \sigma(\text{MLP}(\mathbf{f}))\right)$$

**Plane TV 正则化**：
$$\mathcal{L}_{\text{plane-TV}} = \sum_{p \in \{xy, xz, yz\}} w_p \cdot \text{TV}(P_p)$$

### 11.9 SPAGS 总损失函数

$$\mathcal{L}_{\text{SPAGS}} = \mathcal{L}_{\text{L1}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{TV}} \mathcal{L}_{\text{TV}} + \lambda_{\text{GAR}} \mathcal{L}_{\text{bino}} + \lambda_{\text{ADM}} \mathcal{L}_{\text{plane-TV}}$$

### 11.10 自适应密度控制

**克隆条件**：
$$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}} \quad \text{且} \quad \max(\mathbf{s}) < \tau_{\text{scale}}$$

**分割条件**：
$$\|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau_{\text{grad}} \quad \text{且} \quad \max(\mathbf{s}) > \tau_{\text{scale}}$$

---

## 附录：关键符号表

### A.1 基础符号

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

### A.2 SPS 相关符号

| 符号 | 含义 | 维度/范围 |
|------|------|----------|
| $V_{\text{FDK}}$ | FDK 重建体积 | $\mathbb{R}^{D \times H \times W}$ |
| $\mathcal{M}$ | 有效体素集合 | 集合 |
| $P_i$ | 采样概率 | $[0, 1]$ |
| $G_\sigma$ | 高斯滤波核 | 标准差 $\sigma$ |
| $\tau_{\text{density}}$ | 密度阈值 | 默认 0.05 |

### A.3 GAR 相关符号

| 符号 | 含义 | 维度/范围 |
|------|------|----------|
| $\Delta\theta$ | 角度偏移 | 默认 0.04 rad |
| $Z(\mathbf{u})$ | 像素深度 | $\mathbb{R}_+$ |
| $d(\mathbf{u})$ | 视差 | $\mathbb{R}$ |
| $B$ | 基线 | $\mathbb{R}_+$ |
| $f$ | 焦距 | $\mathbb{R}_+$ |
| $P_i$ | 邻近度分数 | $\mathbb{R}_+$ |
| $\mathcal{N}_K(i)$ | K 近邻集合 | 集合 |
| $\mathcal{D}$ | 密化候选集 | 集合 |
| $\tau_{\text{proximity}}$ | 邻近度阈值 | 默认 5.0 |
| $K$ | 近邻数 | 默认 5 |

### A.4 ADM 相关符号

| 符号 | 含义 | 维度/范围 |
|------|------|----------|
| $P_{xy}, P_{xz}, P_{yz}$ | K-Planes 特征平面 | $\mathbb{R}^{C \times H \times W}$ |
| $C$ | 特征维度 | 默认 32 |
| $\mathbf{f}$ | K-Planes 特征向量 | $\mathbb{R}^{96}$ |
| $\delta_\rho$ | 密度偏移量 | $[-1, 1]$ |
| $m(\cdot)$ | 调制函数 | $(0.7, 1.3)$ |

### A.5 损失权重符号

| 符号 | 含义 | 默认值 |
|------|------|--------|
| $\lambda_{\text{SSIM}}$ | SSIM 损失权重 | 0.25 |
| $\lambda_{\text{TV}}$ | 3D TV 正则化权重 | 0.05 |
| $\lambda_{\text{GAR}}$ | GAR 双目损失权重 | 0.08 |
| $\lambda_{\text{ADM}}$ | ADM Plane TV 权重 | 0.002 |
| $\tau_{\text{grad}}$ | 密化梯度阈值 | $2 \times 10^{-4}$ |

---

## 文档信息

- **生成日期**：2025年12月
- **基于代码版本**：R²-Gaussian + SPAGS（integration 分支）
- **目标用途**：SPAGS 论文方法部分撰写参考
- **核心创新点**：
  - **SPS**：空间先验播种（密度加权初始化）
  - **GAR**：几何感知细化（双目一致性 + 邻近密度反池化）
  - **ADM**：自适应密度调制（K-Planes 特征编码）
