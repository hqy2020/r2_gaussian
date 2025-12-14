&nbsp;

<div align="center">

<p align="center"> <img src="assets/logo.png" width="250px"> </p>

<h2> SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting </h2> 

*稀疏视角 CT 重建的三阶段渐进式优化框架*

</div>

&nbsp;

### 简介

**SPAGS** 是一个基于 3D Gaussian Splatting 的稀疏视角 CT 重建方法，采用三阶段渐进式优化策略：

- **SPS (Spatial Prior Seeding)**: 空间先验播种 - 利用 FDK 密度分布指导点云初始化
- **GAR (Geometry-aware Refinement)**: 几何感知细化 - 双目一致性约束 + 邻近感知密化
- **ADM (Adaptive Density Modulation)**: 自适应密度调制 - K-Planes 空间编码器学习位置相关密度修正

本代码库基于 [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) 改进，专门针对稀疏视角（3/6/9 views）CT 重建场景。

## Baseline 性能（R²-Gaussian）

我们在 5 个器官 × 3 个视角 = 15 个场景下评估了 baseline 方法（R²-Gaussian）的性能：

### 三视角 (3 views)

| 器官 | PSNR | SSIM |
|------|------|------|
| Chest | 26.12 | 0.837 |
| Foot | 28.56 | 0.898 |
| Head | 26.67 | 0.922 |
| Abdomen | 29.24 | 0.936 |
| Pancreas | 28.60 | 0.920 |
| **平均** | **27.84** | **0.903** |

### 六视角 (6 views)

| 器官 | PSNR | SSIM |
|------|------|------|
| Chest | 33.24 | 0.927 |
| Foot | 32.51 | 0.938 |
| Head | 33.11 | 0.973 |
| Abdomen | 34.11 | 0.974 |
| Pancreas | 33.58 | 0.951 |
| **平均** | **33.31** | **0.953** |

### 九视角 (9 views)

| 器官 | PSNR | SSIM |
|------|------|------|
| Chest | 36.95 | 0.953 |
| Foot | 34.95 | 0.955 |
| Head | 35.87 | 0.982 |
| Abdomen | 37.03 | 0.981 |
| Pancreas | 35.71 | 0.961 |
| **平均** | **36.10** | **0.966** |

## 1. 安装

```sh
# 克隆代码
git clone <repository_url> --recursive

# 创建环境
conda env create --file environment.yml
conda activate r2_gaussian_new

# 安装 TIGRE（用于数据生成和初始化）
wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip
unzip v2.3.zip
pip install TIGRE-2.3/Python --no-build-isolation
```

## 2. 数据准备

支持 NAF 格式（`*.pickle`）和 NeRF 格式（`meta_data.json`）。数据组织如下：

```sh
data/
└── 369/                    # 稀疏视角数据集（3/6/9 views）
    ├── foot_50_3views.pickle
    ├── init_foot_50_3views.npy  # SPS 初始化点云
    └── ...
```

## 3. 快速开始

### 3.1 初始化点云（SPS）

```sh
# 生成 SPS 初始化点云（推荐：adaptive，随视角数自动减弱先验，避免高视角过度集中）
python initialize_pcd.py --data <path_to_data> --enable_sps --sps_strategy adaptive --n_points 50000
```

### 3.2 训练

```sh
# 使用消融实验脚本（推荐）
./cc-agent/scripts/run_spags_ablation.sh <配置> <器官> <视角数> [GPU]

# 配置选项:
#   baseline  - Baseline (无任何技术)
#   sps       - 仅 SPS
#   gar       - 仅 GAR
#   adm       - 仅 ADM
#   spags     - Full SPAGS (SPS + GAR + ADM)

# 示例
./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0
```

或直接使用 `train.py`：

```sh
python train.py -s data/369/foot_50_3views.pickle \
    -m output/experiment_name \
    --ply_path data/369/init_foot_50_3views.npy \
    --enable_fsgs_proximity \
    --enable_kplanes
```

### 3.3 评估

```sh
python test.py -m <path_to_trained_model>
```

## 4. SPAGS 核心参数

### SPS (空间先验播种)
- `--enable_sps`: 启用 SPS 初始化（需在 `initialize_pcd.py` 中生成 init_*.npy）
- `--sps_strategy`: 采样策略（`adaptive|mixed|density_weighted|stratified`，推荐 `adaptive`）
- `--sps_uniform_ratio`: mixed/adaptive 中均匀采样占比（其余为密度加权）
- `--sps_density_gamma`: 密度权重幂指数 γ（`<1` 更平滑，`>1` 更尖锐）
- `--n_points`: 初始化点云数量（默认 50000）

### GAR (几何感知细化)
- `--enable_fsgs_proximity`: 启用邻近感知密化
- `--gar_proximity_threshold`: 邻近密化阈值（默认 0.05，场景归一化到 [-1,1]^3 后邻近分数典型范围约 0.01~0.5）
- `--gar_new_per_source`: 每个候选点最多生成的新点数（默认 1；<=0 表示使用全部 K 个邻居，更贴近 FSGS）

### ADM (自适应密度调制)
- `--enable_kplanes`: 启用 K-Planes 编码器
- `--kplanes_resolution`: K-Planes 分辨率（默认 64）
- `--kplanes_dim`: 特征维度（默认 32）
- `--lambda_plane_tv`: Plane TV 正则化权重（默认 0.002）

## 5. 代码结构

```
r2_gaussian/
├── train.py                    # 主训练入口
├── test.py                     # 测试评估入口
├── initialize_pcd.py           # 点云初始化（SPS）
├── r2_gaussian/
│   ├── gaussian/
│   │   ├── gaussian_model.py   # Gaussian 参数管理
│   │   ├── kplanes.py          # K-Planes 编码器（ADM）
│   │   └── render_query.py     # 渲染和查询
│   ├── utils/
│   │   └── loss_utils.py       # 基础损失函数
│   └── innovations/fsgs/       # 邻近密集化模块（GAR）
└── cc-agent/                   # 实验脚本和工具
```

## 6. 致谢与引用

本代码基于 [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian)、[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 和 [TIGRE](https://github.com/CERN/TIGRE.git) 开发。

如果本代码库对您有帮助，请考虑引用：

```
@article{spags,
  title={SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting for Sparse-View CT Reconstruction},
  author={...},
  journal={...},
  year={2025}
}
```
