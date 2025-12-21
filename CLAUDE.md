# CLAUDE.md

## 项目概述

**SPAGS**: 基于 3D Gaussian Splatting 的 CT 断层扫描重建项目。核心目标是稀疏视角（3/6/9 views）新视角合成。

项目集成了 **9 种新视角合成方法** 用于对比实验：
- **6 种 3DGS 方法**: SPAGS (主方法)、R²-Gaussian、X-Gaussian、FSGS、DN-Gaussian、CoR-GS
- **3 种 NeRF 方法**: NAF、TensoRF、SAX-NeRF

## 重要约定

- **所有回复和写入文档的内容都是中文**
- **训练命名格式**: `yyyy_MM_dd_HH_mm_organ_{{nums}}views_{{technique}}`
- **CUDA 环境**: `r2_gaussian_new`
- **数据集位置**: `data/369/`（3/6/9 稀疏视角数据）
- **多使用 serena MCP 理解代码，修改代码**
- **尽可能确保都是有专门的助手 agent 执行具体流程**

## 集成方法概览

### 3D Gaussian Splatting 方法

| 方法 | 说明 | 核心文件 |
|------|------|----------|
| **SPAGS** | 我们的方法，包含三阶段优化 (SPS+GAR+ADM) | `r2_gaussian/gaussian/gaussian_model.py` |
| **baseline** | R²-Gaussian 基准 | `r2_gaussian/gaussian/gaussian_model.py` |
| **xgaussian** | X-Gaussian，使用球谐特征和 opacity | `r2_gaussian/baselines/xgaussian/` |
| **fsgs** | FSGS 邻近高斯密集化 | `r2_gaussian/baselines/fsgs/` |
| **dngaussian** | DN-Gaussian 深度正则化 | `r2_gaussian/baselines/dngaussian/` |
| **corgs** | CoR-GS 协同正则化 | `r2_gaussian/baselines/corgs/` |

### NeRF 方法

| 方法 | 说明 | 编码方式 | 核心文件 |
|------|------|----------|----------|
| **NAF** | Neural Attenuation Fields | Hash Grid (16 levels) | `r2_gaussian/baselines/naf/` |
| **TensoRF** | 张量分解 VM 编码 | TensoRF VM | `r2_gaussian/baselines/tensorf/` |
| **SAX-NeRF** | Lineformer Transformer 增强 | Hash Grid + Lineformer | `r2_gaussian/baselines/saxnerf/` |

### 方法路由机制

`train.py` 通过 `--method` 参数选择方法：
```python
# 3DGS 方法
--method r2_gaussian  # R²-Gaussian 基准 (baseline)
--method xgaussian    # X-Gaussian
--method fsgs         # FSGS
--method dngaussian   # DN-Gaussian
--method corgs        # CoR-GS

# NeRF 方法
--method naf          # NAF
--method tensorf      # TensoRF
--method saxnerf      # SAX-NeRF
```

## 6 种 3DGS 方法性能对比 (PSNR/SSIM)

### 三视角 (3 views) 结果

| 方法 | Chest | Foot | Head | Abdomen | Pancreas | **平均** |
|------|-------|------|------|---------|----------|----------|
| **SPAGS** | 27.03/0.847 | 28.59/0.900 | 26.75/0.919 | 29.66/0.938 | 29.13/0.924 | **28.23/0.905** |
| baseline | 26.16/0.837 | 28.82/0.898 | 26.59/0.922 | 29.24/0.936 | 28.58/0.920 | 27.88/0.903 |
| xgaussian | 20.48/0.739 | 26.03/0.848 | 21.02/0.863 | 23.56/0.906 | 24.93/0.898 | 23.20/0.851 |
| fsgs | 20.55/0.736 | 25.79/0.851 | 20.51/0.859 | 24.01/0.907 | 25.08/0.899 | 23.19/0.851 |
| dngaussian | 20.52/0.748 | 24.78/0.852 | 17.70/0.798 | 16.20/0.827 | 23.65/0.885 | 20.57/0.822 |
| corgs | 19.53/0.724 | 25.25/0.851 | 20.54/0.862 | 22.59/0.909 | 20.36/0.880 | 21.65/0.845 |

### 六视角 (6 views) 结果

| 方法 | Chest | Foot | Head | Abdomen | Pancreas | **平均** |
|------|-------|------|------|---------|----------|----------|
| **SPAGS** | 33.44/0.928 | 32.31/0.940 | 32.78/0.973 | 34.37/0.976 | 33.91/0.952 | **33.36/0.954** |
| baseline | 33.14/0.927 | 32.31/0.937 | 33.03/0.973 | 34.00/0.974 | 33.40/0.950 | 33.18/0.952 |
| xgaussian | 28.37/0.877 | 28.28/0.895 | 27.48/0.942 | 27.41/0.946 | 29.03/0.932 | 28.11/0.918 |
| fsgs | 27.61/0.876 | 28.34/0.897 | 27.77/0.944 | 28.17/0.950 | 28.82/0.933 | 28.14/0.920 |
| dngaussian | 27.34/0.863 | 27.57/0.901 | 18.24/0.836 | 19.73/0.907 | 28.97/0.933 | 24.37/0.888 |
| corgs | 27.69/0.871 | 28.45/0.901 | 27.28/0.944 | 28.27/0.951 | 29.28/0.936 | 28.19/0.921 |

### 九视角 (9 views) 结果

| 方法 | Chest | Foot | Head | Abdomen | Pancreas | **平均** |
|------|-------|------|------|---------|----------|----------|
| **SPAGS** | 36.79/0.952 | 34.48/0.955 | 36.15/0.982 | 36.74/0.982 | 35.70/0.961 | **35.97/0.967** |
| baseline | 36.92/0.953 | 34.96/0.955 | 35.80/0.982 | 36.97/0.981 | 35.80/0.960 | 36.09/0.966 |
| xgaussian | 31.90/0.915 | 29.78/0.922 | 29.41/0.956 | 30.11/0.960 | 29.81/0.944 | 30.20/0.939 |
| fsgs | 31.29/0.915 | 30.74/0.926 | 30.22/0.960 | 31.46/0.964 | 31.03/0.945 | 30.95/0.942 |
| dngaussian | 30.42/0.898 | 29.55/0.921 | 21.16/0.872 | 21.40/0.924 | 29.77/0.943 | 26.46/0.912 |
| corgs | 32.26/0.921 | 30.89/0.928 | 30.02/0.956 | 31.46/0.965 | 30.78/0.947 | 31.08/0.943 |

## SPAGS 消融实验结果

SPAGS 包含三个创新点：**SPS** (采样策略)、**GAR** (邻近密化)、**ADM** (密度调制)。
以下是 7 种正交组合的实验结果 (平均 PSNR/SSIM)：

| 配置 | 3v PSNR/SSIM | 6v PSNR/SSIM | 9v PSNR/SSIM |
|------|--------------|--------------|--------------|
| sps | 28.14/0.904 | 33.38/0.954 | 36.03/0.966 |
| adm | 28.04/0.904 | 33.38/0.953 | 36.10/0.966 |
| gar | 27.99/0.903 | 33.36/0.953 | 36.16/0.966 |
| sps_adm | 28.19/0.904 | 33.40/0.954 | 36.02/0.966 |
| sps_gar | 28.22/0.905 | 33.31/0.954 | 36.00/0.967 |
| gar_adm | 28.05/0.904 | 33.30/0.953 | 36.04/0.966 |
| **spags** | **28.35/0.906** | **33.40/0.954** | **36.03/0.967** |

## 常用命令

### 环境激活
```bash
conda activate r2_gaussian_new
```

### 训练

#### SPAGS 消融实验脚本（推荐）

```bash
# SPAGS 方法示例
./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0
./cc-agent/scripts/run_spags_ablation.sh baseline chest 6 1
./cc-agent/scripts/run_spags_ablation.sh sps_gar pancreas 9 0

# 3DGS 基准方法示例
./cc-agent/scripts/run_spags_ablation.sh xgaussian foot 3 0
./cc-agent/scripts/run_spags_ablation.sh fsgs chest 6 1
./cc-agent/scripts/run_spags_ablation.sh dngaussian head 9 0
./cc-agent/scripts/run_spags_ablation.sh corgs abdomen 3 0

# NeRF 方法示例
./cc-agent/scripts/run_spags_ablation.sh naf chest 6 1
./cc-agent/scripts/run_spags_ablation.sh tensorf head 9 0
./cc-agent/scripts/run_spags_ablation.sh saxnerf abdomen 3 0
```

### 初始化点云
```bash
python initialize_pcd.py --data <path_to_data>
python initialize_pcd.py --data <path_to_data> --evaluate  # 评估初始化质量
```

## 代码架构

```
r2_gaussian/
├── train.py                      # 主训练入口（方法路由）
├── test.py                       # 测试评估入口
├── initialize_pcd.py             # 点云初始化（SPS 实现）
├── r2_gaussian/                  # 核心代码模块
│   ├── gaussian/                 # R²-Gaussian / SPAGS 核心
│   │   ├── gaussian_model.py     # GaussianModel 类：Gaussian 参数管理
│   │   ├── render_query.py       # render() / query() 函数
│   │   ├── kplanes.py            # K-Planes 编码器（ADM 实现）
│   │   └── initialize.py         # 初始化逻辑
│   ├── baselines/                # 基准方法
│   │   ├── registry.py           # 方法注册表（核心配置）
│   │   ├── xgaussian/            # X-Gaussian (3DGS)
│   │   ├── fsgs/                 # FSGS (3DGS)
│   │   ├── dngaussian/           # DN-Gaussian (3DGS)
│   │   ├── corgs/                # CoR-GS (3DGS)
│   │   ├── naf/                  # NAF (NeRF)
│   │   ├── tensorf/              # TensoRF (NeRF)
│   │   ├── saxnerf/              # SAX-NeRF (NeRF)
│   │   └── nerf_base/            # NeRF 共享基础代码
│   ├── innovations/              # 创新点模块
│   │   └── fsgs/                 # FSGS 邻近密集化（GAR 实现）
│   │       └── proximity_densifier.py
│   ├── dataset/                  # 数据加载
│   ├── utils/                    # 工具函数
│   ├── arguments/                # 命令行参数定义
│   └── submodules/               # CUDA 扩展
├── cc-agent/                     # AI 科研助手系统
│   ├── experiment/               # 实验数据
│   │   ├── results_6methods_90experiments.md  # 6 方法对比结果
│   │   └── ablation_results.md   # 消融实验结果
│   ├── scripts/                  # 脚本工具
│   └── figures/                  # 生成的图片
├── data/369/                     # 稀疏视角数据集
└── output/                       # 训练输出
```

## 关键数据流

### SPAGS 训练流程
1. **数据加载**: `Scene` → `dataset_readers.py` → NAF/NeRF 格式解析
2. **初始化**: `initialize_pcd.py` → FDK 重建 → 密度加权采样 (SPS) → `init_*.npy`
3. **训练循环**:
   - GAR 邻近密化 (每 N 次迭代)
   - ADM 密度调制 (K-Planes)
   - 渲染投影 → 损失计算 → 优化
4. **体积查询**: `query()` → 体素化 Gaussian → 3D 体积重建

### 3DGS 基准方法训练流程
- X-Gaussian、FSGS、DN-Gaussian、CoR-GS 使用各自的 trainer.py

### NeRF 基准训练流程
1. **数据加载**: 相同的数据接口
2. **编码**: Hash Grid (NAF/SAX-NeRF) 或 TensoRF VM
3. **网络**: MLP (NAF/TensoRF) 或 Lineformer (SAX-NeRF)
4. **渲染**: 体积渲染管线

## GaussianModel 核心属性

- `_xyz`: 高斯中心位置 (world coordinates)
- `_scaling`: 3D 尺度参数
- `_rotation`: 旋转四元数
- `_density`: 密度参数（Softplus 激活）

## 关键文件速查表

| 功能 | 文件路径 |
|------|---------|
| 方法注册表 | `r2_gaussian/baselines/registry.py` |
| SPAGS / R²-Gaussian 模型 | `r2_gaussian/gaussian/gaussian_model.py` |
| X-Gaussian 模型 | `r2_gaussian/baselines/xgaussian/model.py` |
| NeRF 统一训练器 | `r2_gaussian/baselines/nerf_base/trainer.py` |
| Lineformer 网络 | `r2_gaussian/baselines/nerf_base/lineformer.py` |
| K-Planes (ADM) | `r2_gaussian/gaussian/kplanes.py` |
| FSGS 密化器 (GAR) | `r2_gaussian/innovations/fsgs/proximity_densifier.py` |
| 点云初始化 (SPS) | `initialize_pcd.py` |
| 消融实验脚本 | `cc-agent/scripts/run_spags_ablation.sh` |
| 6 方法对比结果 | `cc-agent/experiment/results_6methods_90experiments.md` |
| 消融实验结果 | `cc-agent/experiment/ablation_results.md` |

## 技术栈

- **Python 3.9** + **PyTorch 1.12.1** + **CUDA 11.6**
- **TIGRE 2.3**: CT 数据生成和 FDK 重建
- **Open3D**: 3D 数据处理
- **TensorBoard**: 训练可视化
