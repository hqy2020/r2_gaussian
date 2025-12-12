# CLAUDE.md

## 项目概述

**SPAGS**: 基于 3D Gaussian Splatting 的 CT 断层扫描重建项目。核心目标是稀疏视角（3/6/9 views）新视角合成。

项目集成了 **5 种新视角合成方法** 用于对比实验：
- **2 种 3DGS 方法**: R²-Gaussian (主方法)、X-Gaussian
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
| **R²-Gaussian** | 主方法，包含 SPAGS 三阶段优化 (SPS/GAR/ADM) | `r2_gaussian/gaussian/gaussian_model.py` |
| **X-Gaussian** | 3DGS 基准，使用球谐特征和 opacity | `r2_gaussian/baselines/xgaussian/` |

### NeRF 方法

| 方法 | 说明 | 编码方式 | 核心文件 |
|------|------|----------|----------|
| **NAF** | Neural Attenuation Fields | Hash Grid (16 levels) | `r2_gaussian/baselines/naf/` |
| **TensoRF** | 张量分解 VM 编码 | TensoRF VM | `r2_gaussian/baselines/tensorf/` |
| **SAX-NeRF** | Lineformer Transformer 增强 | Hash Grid + Lineformer | `r2_gaussian/baselines/saxnerf/` |

### 方法路由机制

`train.py` 通过 `--method` 参数选择方法：
```python
--method r2_gaussian  # 默认，主方法
--method xgaussian    # X-Gaussian
--method naf          # NAF
--method tensorf      # TensoRF
--method saxnerf      # SAX-NeRF
```

## R²-Gaussian 性能结果

### 三视角 (3 views) 结果

| 器官 | PSNR | SSIM |
|------|------|------|
| Chest | 26.12 | 0.837 |
| Foot | 28.56 | 0.898 |
| Head | 26.67 | 0.922 |
| Abdomen | 29.24 | 0.936 |
| Pancreas | 28.60 | 0.920 |
| 平均 | 27.84 | 0.903 |

### 六视角 (6 views) 结果

| 器官 | PSNR | SSIM |
|------|------|------|
| Chest | 33.24 | 0.927 |
| Foot | 32.51 | 0.938 |
| Head | 33.11 | 0.973 |
| Abdomen | 34.11 | 0.974 |
| Pancreas | 33.58 | 0.951 |
| 平均 | 33.31 | 0.953 |

### 九视角 (9 views) 结果

| 器官 | PSNR | SSIM |
|------|------|------|
| Chest | 36.95 | 0.953 |
| Foot | 34.95 | 0.955 |
| Head | 35.87 | 0.982 |
| Abdomen | 37.03 | 0.981 |
| Pancreas | 35.71 | 0.961 |
| 平均 | 36.10 | 0.966 |

## 常用命令

### 环境激活
```bash
conda activate r2_gaussian_new
```

### 训练

#### SPAGS 消融实验脚本（推荐）

# SPAGS 方法示例
./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0
./cc-agent/scripts/run_spags_ablation.sh baseline chest 6 1
./cc-agent/scripts/run_spags_ablation.sh sps_gar pancreas 9 0

# 基准方法示例
./cc-agent/scripts/run_spags_ablation.sh xgaussian foot 3 0
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
│   ├── gaussian/                 # R²-Gaussian 核心
│   │   ├── gaussian_model.py     # GaussianModel 类：Gaussian 参数管理
│   │   ├── render_query.py       # render() / query() 函数
│   │   ├── kplanes.py            # K-Planes 编码器（ADM 实现）
│   │   └── initialize.py         # 初始化逻辑
│   ├── baselines/                # 四种基准方法
│   │   ├── registry.py           # 方法注册表（核心配置）
│   │   ├── xgaussian/            # X-Gaussian (3DGS)
│   │   │   ├── model.py          # XGaussianModel 类
│   │   │   ├── renderer.py       # X 光渲染器
│   │   │   ├── trainer.py        # 训练函数
│   │   │   └── config.py         # 参数配置
│   │   ├── naf/                  # NAF (NeRF)
│   │   │   └── config.py         # Hash Grid 配置
│   │   ├── tensorf/              # TensoRF (NeRF)
│   │   │   └── config.py         # VM 编码配置
│   │   ├── saxnerf/              # SAX-NeRF (NeRF + Transformer)
│   │   │   └── config.py         # Lineformer 配置
│   │   └── nerf_base/            # NeRF 共享基础代码
│   │       ├── encoder/          # 位置编码器
│   │       │   ├── hashgrid.py   # Hash Grid 编码
│   │       │   └── tensorf.py    # TensoRF VM 编码
│   │       ├── lineformer.py     # Lineformer 网络
│   │       ├── network.py        # MLP 密度网络
│   │       ├── render.py         # NeRF 渲染函数
│   │       └── trainer.py        # 统一 NeRF 训练函数
│   ├── innovations/              # 创新点模块
│   │   └── fsgs/                 # FSGS 邻近密集化（GAR 实现）
│   │       └── proximity_densifier.py
│   ├── dataset/                  # 数据加载
│   │   ├── dataset_readers.py    # NAF/NeRF 格式解析
│   │   └── cameras.py            # Camera 类定义
│   ├── utils/                    # 工具函数
│   │   ├── loss_utils.py         # L1/SSIM/TV 损失函数
│   │   └── ...
│   ├── arguments/                # 命令行参数定义
│   └── submodules/               # CUDA 扩展
│       ├── xray-gaussian-rasterization-voxelization/  # X 射线光栅化
│       └── simple-knn/           # KNN 搜索
├── cc-agent/                     # AI 科研助手系统
├── data/369/                     # 稀疏视角数据集
└── output/                       # 训练输出
```

## 关键数据流

### R²-Gaussian 训练流程
1. **数据加载**: `Scene` → `dataset_readers.py` → NAF/NeRF 格式解析
2. **初始化**: `initialize_pcd.py` → FDK 重建 → 密度加权采样 (SPS) → `init_*.npy`
3. **训练循环**:
   - GAR 邻近密化 (每 N 次迭代)
   - ADM 密度调制 (K-Planes)
   - 渲染投影 → 损失计算 → 优化
4. **体积查询**: `query()` → 体素化 Gaussian → 3D 体积重建

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
| R²-Gaussian 模型 | `r2_gaussian/gaussian/gaussian_model.py` |
| X-Gaussian 模型 | `r2_gaussian/baselines/xgaussian/model.py` |
| NeRF 统一训练器 | `r2_gaussian/baselines/nerf_base/trainer.py` |
| Lineformer 网络 | `r2_gaussian/baselines/nerf_base/lineformer.py` |
| K-Planes (ADM) | `r2_gaussian/gaussian/kplanes.py` |
| FSGS 密化器 (GAR) | `r2_gaussian/innovations/fsgs/proximity_densifier.py` |
| 点云初始化 (SPS) | `initialize_pcd.py` |
| 消融实验脚本 | `cc-agent/scripts/run_spags_ablation.sh` |

## 技术栈

- **Python 3.9** + **PyTorch 1.12.1** + **CUDA 11.6**
- **TIGRE 2.3**: CT 数据生成和 FDK 重建
- **Open3D**: 3D 数据处理
- **TensorBoard**: 训练可视化
