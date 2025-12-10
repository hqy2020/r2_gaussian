# SPS 空间先验播种技术文档

> **SPS**: **S**patial **P**rior **S**eeding
>
> 利用 FDK 粗重建的密度分布进行智能初始化

---

## 1. Motivation：为什么需要空间感知初始化？

### 1.1 问题：随机初始化在稀疏视角下失效

标准 3DGS 通常使用**随机采样**或**均匀采样**来初始化点云。这在稀疏视角 CT 重建中存在严重问题：

```
随机采样的问题
├── 高密度区域（骨骼、器官边界）
│   ├── 采样点数不足
│   ├── 细节容易丢失
│   └── 需要更多训练迭代来"发现"这些区域
│
└── 低密度区域（空气、背景）
    ├── 采样点数过多
    ├── 浪费计算资源
    └── 增加不必要的优化负担
```

### 1.2 核心洞察：FDK 重建包含空间先验

即使只有 3 个视角，FDK 算法也能给出一个**粗略但有意义**的 3D 密度分布：
- 高密度区域的位置大致正确
- 密度值的相对关系基本保持
- 这是一个免费的"空间先验"

### 1.3 SPS 的解决方案

**核心思想**：让采样概率与密度成正比

$$
P(x_i) = \frac{\rho(x_i)}{\sum_j \rho(x_j)}
$$

- 高密度点 → 高采样概率 → 更多初始化点
- 低密度点 → 低采样概率 → 更少初始化点

---

## 2. 核心公式

### 2.1 密度加权采样

给定 FDK 重建的 3D 体积 $V$，对每个有效体素 $x_i$：

```
采样概率 = 该点密度 / 所有有效点密度之和
```

代码实现（`initialize_pcd.py:152-164`）：

```python
# 获取有效体素的密度值
densities_flat = vol[
    valid_indices[:, 0],
    valid_indices[:, 1],
    valid_indices[:, 2],
]

# 归一化为采样概率
probs = densities_flat / densities_flat.sum()

# 无放回采样
sampled_idx = np.random.choice(
    len(valid_indices),
    n_points,
    replace=False,
    p=probs  # 关键：使用密度作为采样概率
)
```

### 2.2 采样效果对比

假设有 50,000 个点要采样：

| 密度区间 | 随机采样 | SPS 采样 | 变化 |
|---------|---------|---------|------|
| 高密度 (≥0.25) | ~3,000 点 | ~6,000 点 | **+100%** |
| 中密度 (0.10-0.25) | ~15,000 点 | ~18,000 点 | +20% |
| 低密度 (≤0.10) | ~32,000 点 | ~26,000 点 | -19% |

---

## 3. 实现流程

### 3.1 完整数据流

```
输入：X 射线投影 + 扫描器几何参数
          ↓
┌─────────────────────────────────────┐
│ Step 1: 加载场景                     │
│   Scene() → 读取投影和相机参数        │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Step 2: FDK 重建                     │
│   projs + angles → vol [X, Y, Z]    │
│   使用 TIGRE 库的 algs.fdk()         │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Step 3: 高斯滤波降噪（可选）          │
│   vol ← gaussian_filter(vol, σ=3.0) │
│   去除 FDK 重建的噪声和伪影           │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Step 4: 密度阈值过滤                 │
│   mask = vol > density_thresh       │
│   只保留密度 > 0.05 的体素            │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Step 5: 密度加权采样                 │
│   P(x) = ρ(x) / Σρ                  │
│   采样 n_points 个点                 │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Step 6: 坐标映射                     │
│   体素索引 → 世界坐标                │
│   pos = idx × dVoxel - sVoxel/2     │
└─────────────────────────────────────┘
          ↓
输出：init_*.npy [N, 4] = [x, y, z, density]
```

### 3.2 坐标映射公式

将体素索引转换为世界坐标：

```python
# 采样位置的世界坐标
sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin

# 其中：
# - sampled_indices: 采样的体素索引 [N, 3]
# - dVoxel: 体素尺寸 (例如 0.5mm)
# - sVoxel: 体积尺寸 (例如 [128, 128, 128])
# - offOrigin: 原点偏移
```

### 3.3 密度缩放

```python
# 采样密度值并缩放
sampled_densities = vol[sampled_indices] * density_rescale

# density_rescale = 0.15（默认）
# 用于平衡初始密度值的大小
```

---

## 4. 超参数设置

### 4.1 主要参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|---------|
| `density_thresh` | 0.05 | 有效体素的密度阈值 | 若有效点<N，降低到 0.01-0.02 |
| `density_rescale` | 0.15 | 密度缩放因子 | 控制初始密度大小，范围 [0.1, 0.3] |
| `n_points` | 50000 | 初始化点数 | 与 GPU 内存相关，通常 50k 足够 |
| `sps_denoise` | False | 是否启用高斯滤波 | 稀疏视角建议开启 |
| `sps_denoise_sigma` | 3.0 | 高斯滤波标准差 | 去噪强度，3.0 通常足够 |

### 4.2 采样策略

| 策略 | 参数值 | 说明 |
|------|--------|------|
| `density_weighted` | 默认 | 密度加权采样，高密度区更多点 |
| `stratified` | 可选 | 分层采样，确保各密度区间有代表 |

---

## 5. 使用方法

### 5.1 生成 SPS 点云

```bash
# 基础用法
python initialize_pcd.py -s data/369/foot_50_3views.pickle --enable_sps

# 完整参数
python initialize_pcd.py \
    -s data/369/foot_50_3views.pickle \
    --enable_sps \
    --sps_denoise \
    --sps_denoise_sigma 3.0 \
    --density_thresh 0.05 \
    --n_points 50000

# 输出位置
# → data/density-369/init_foot_50_3views.npy
```

### 5.2 在训练中使用 SPS

```bash
# 方法 1：直接指定点云路径
python train.py \
    -s data/369/foot_50_3views.pickle \
    --ply_path data/density-369/init_foot_50_3views.npy

# 方法 2：使用消融脚本（推荐）
./cc-agent/scripts/run_spags_ablation.sh sps foot 3 0
```

### 5.3 消融脚本中的 SPS 配置

在 `run_spags_ablation.sh` 中：

```bash
# SPS 点云路径
SPS_PCD_PATH="data/density-369/init_${ORGAN}_50_${VIEWS}views.npy"

# 当配置包含 SPS 时
if [ "$USE_SPS" = true ]; then
    PLY_FLAG="--ply_path $SPS_PCD_PATH"
fi
```

---

## 6. 代码位置索引

| 功能 | 文件 | 位置 |
|------|------|------|
| 参数定义 | `initialize_pcd.py` | 第 27-75 行 `InitParams` |
| 主初始化逻辑 | `initialize_pcd.py` | 第 100-250 行 |
| 密度加权采样 | `initialize_pcd.py` | 第 152-164 行 |
| 分层采样 | `initialize_pcd.py` | 第 165-196 行 |
| 点云加载 | `r2_gaussian/gaussian/initialize.py` | 第 13-61 行 |
| GaussianModel 初始化 | `r2_gaussian/gaussian/gaussian_model.py` | 第 389-419 行 `create_from_pcd()` |

---

## 7. 注意事项

### 7.1 点云文件格式

SPS 点云保存为 `.npy` 文件，格式为 `[N, 4]`：
- 前 3 列：世界坐标 (x, y, z)
- 第 4 列：密度值

```python
# 读取点云
point_cloud = np.load("init_foot_50_3views.npy")
xyz = point_cloud[:, :3]      # [N, 3]
density = point_cloud[:, 3:4] # [N, 1]
```

### 7.2 场景归一化

R²-Gaussian 将场景归一化到 $[-1, 1]^3$，确保：
- 所有坐标在 [-1, 1] 范围内
- 密度值经过适当缩放
- 这对 GAR 和 ADM 的超参数设置很重要

### 7.3 与其他模块的协同

SPS 为后续模块奠定基础：
- **GAR**：更好的初始分布 → 更准确的邻近分数
- **ADM**：更合理的初始密度 → 更稳定的调制学习

---

*文档版本：v1.0 | 更新日期：2025-12-10*
