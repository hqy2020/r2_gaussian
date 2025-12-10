# SPAGS 技术架构文档

> **SPAGS**: **S**patial **P**rior-**A**ware **G**aussian **S**platting
>
> 面向稀疏视角 CT 重建的空间感知 3D 高斯喷溅方法

---

## 1. 项目背景

### 1.1 问题定义

**稀疏视角 CT 重建**：仅使用 3/6/9 个视角的 X 射线投影，重建高质量的 3D CT 体积。

**核心挑战**：
- 视角数量极少（传统 CT 需要 50-360 个视角）
- 几何约束严重不足
- 容易产生伪影和形变

### 1.2 为什么需要"空间感知"？

标准 3D Gaussian Splatting (3DGS) 在稀疏视角下存在三个关键问题：

| 问题 | 现象 | 根因 |
|------|------|------|
| **初始化偏差** | 高密度区域细节丢失 | 随机采样无法反映解剖结构 |
| **几何约束缺失** | 密化位置不准确 | 梯度信号在稀疏视角下不可靠 |
| **密度估计失准** | 局部 CT 值偏离真实 | 全局统一激活无法自适应 |

**SPAGS 的解决方案**：通过三个**空间感知**模块，分别解决这三个问题。

---

## 2. SPAGS 三阶段框架

SPAGS 由三个互补的创新点组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                        SPAGS 框架                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │     SPS      │    │     GAR      │    │     ADM      │       │
│  │ 空间先验播种  │ → │ 几何感知细化  │ → │ 自适应密度调制 │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        ↓                   ↓                   ↓                 │
│  解决初始化偏差     解决几何约束缺失     解决密度估计失准         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 SPS (Spatial Prior Seeding) - 空间先验播种

**作用阶段**：训练前（点云初始化）

**核心思想**：
- 利用 FDK 粗重建的密度分布作为**空间先验**
- 高密度区域（如骨骼）获得更多初始化点
- 低密度区域（如空气）减少浪费的点

**详细文档**：[SPS_空间先验播种.md](./SPS_空间先验播种.md)

### 2.2 GAR (Geometry-Aware Refinement) - 几何感知细化

**作用阶段**：训练中（迭代 1000-15000）

**核心思想**：
- 使用**邻近分数**直接感知几何稀疏性
- 高斯距离邻居越远 → 越需要密化
- 不依赖可能不可靠的梯度信号

**详细文档**：[GAR_几何感知细化.md](./GAR_几何感知细化.md)

### 2.3 ADM (Adaptive Density Modulation) - 自适应密度调制

**作用阶段**：训练全程

**核心思想**：
- 使用 K-Planes 学习**位置相关的密度修正**
- 不同解剖结构获得不同的密度响应
- 视角越少，调制越强

**详细文档**：[ADM_自适应密度调制.md](./ADM_自适应密度调制.md)

---

## 3. 训练流程

### 3.1 数据流

```
输入数据
├── 投影数据: data/369/{organ}_50_{views}views.pickle
│   └── 包含 X 射线投影和相机参数
│
└── 初始点云（可选 SPS）:
    ├── Baseline: data/369/init_*.npy（随机采样）
    └── SPS: data/density-369/init_*.npy（密度加权采样）

    ↓ train.py

训练循环 (30000 iterations)
├── 1. 随机采样一个训练视角
├── 2. 渲染：GaussianRasterizer → X 射线投影
├── 3. 损失计算：L1 + DSSIM + TV
├── 4. 反向传播
├── 5. 密化/剪枝：
│   ├── 标准密化（梯度驱动）
│   └── GAR 密化（邻近驱动）  ← 每 500 iter
├── 6. ADM 调制（每次迭代生效）
└── 7. 优化器更新

    ↓

输出
└── output/{timestamp}_{organ}_{views}views_{config}/
    ├── point_cloud/  # 保存的点云
    ├── ckpt/         # 检查点
    └── eval/         # 评估结果
```

### 3.2 关键函数调用

| 阶段 | 函数 | 文件位置 |
|------|------|---------|
| 初始化 | `initialize_gaussian()` | `r2_gaussian/gaussian/initialize.py` |
| 渲染 | `render()` | `r2_gaussian/gaussian/render_query.py` |
| GAR 密化 | `compute_proximity_scores()` | `r2_gaussian/innovations/fsgs/proximity_densifier.py` |
| ADM 调制 | `get_density` property | `r2_gaussian/gaussian/gaussian_model.py:244-273` |
| TV 正则化 | `compute_plane_tv_loss()` | `r2_gaussian/utils/regulation.py` |

---

## 4. 消融实验配置

SPAGS 支持 8 种配置组合，通过脚本 `cc-agent/scripts/run_spags_ablation.sh` 运行。

### 4.1 配置矩阵

| 配置 | SPS | GAR | ADM | 说明 |
|------|:---:|:---:|:---:|------|
| `baseline` | - | - | - | 标准 R²-Gaussian |
| `sps` | ✓ | - | - | 仅空间先验播种 |
| `gar` | - | ✓ | - | 仅几何感知细化 |
| `adm` | - | - | ✓ | 仅自适应密度调制 |
| `sps_gar` | ✓ | ✓ | - | SPS + GAR |
| `sps_adm` | ✓ | - | ✓ | SPS + ADM |
| `gar_adm` | - | ✓ | ✓ | GAR + ADM |
| `spags` | ✓ | ✓ | ✓ | **完整 SPAGS** |

### 4.2 命令行参数映射

```bash
# SPS：通过 --ply_path 指定密度加权点云
--ply_path data/density-369/init_${organ}_50_${views}views.npy

# GAR：启用邻近密化及优化参数
--enable_fsgs_proximity \
--gar_adaptive_threshold \
--gar_adaptive_percentile 85 \
--gar_progressive_decay \
--gar_decay_start_ratio 0.7 \
--gar_final_strength 0.5

# ADM：启用 K-Planes 密度调制
--enable_kplanes
```

---

## 5. 快速上手

### 5.1 环境准备

```bash
conda activate r2_gaussian_new
cd /home/qyhu/Documents/r2_ours/r2_gaussian
```

### 5.2 运行完整 SPAGS

```bash
# 用法: ./cc-agent/scripts/run_spags_ablation.sh <配置> <器官> <视角数> [GPU]

# 示例：在 foot 数据集上运行 3 视角完整 SPAGS
./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0

# 示例：在 chest 数据集上运行 6 视角 baseline
./cc-agent/scripts/run_spags_ablation.sh baseline chest 6 1
```

### 5.3 支持的器官和视角

- **器官**: foot, chest, head, abdomen, pancreas
- **视角**: 3, 6, 9

### 5.4 输出目录

训练输出位于：
```
output/{时间戳}_{器官}_{视角数}views_{配置}/
├── cfg_args                # 配置参数
├── point_cloud/            # 迭代保存的点云
│   └── iteration_{N}/
├── ckpt/                   # 训练检查点
├── eval/                   # 评估结果（PSNR/SSIM）
└── training.log            # 训练日志
```

---

## 6. 代码结构索引

```
r2_gaussian/
├── train.py                    # 主训练入口
├── test.py                     # 测试评估
├── initialize_pcd.py           # SPS 点云初始化
│
├── r2_gaussian/
│   ├── arguments/
│   │   └── __init__.py         # ModelParams, OptimizationParams
│   │
│   ├── gaussian/
│   │   ├── gaussian_model.py   # GaussianModel 类
│   │   ├── initialize.py       # 点云加载
│   │   ├── kplanes.py          # ADM: K-Planes 编码器
│   │   └── render_query.py     # 渲染和查询
│   │
│   ├── innovations/
│   │   └── fsgs/
│   │       └── proximity_densifier.py  # GAR: 邻近密化器
│   │
│   └── utils/
│       ├── loss_utils.py       # 损失函数
│       └── regulation.py       # TV 正则化
│
└── cc-agent/
    ├── scripts/
    │   └── run_spags_ablation.sh  # 消融实验脚本
    └── docs/                      # 技术文档（本目录）
```

---

## 7. 相关文档

- [SPS_空间先验播种.md](./SPS_空间先验播种.md) - 详细介绍空间先验初始化
- [GAR_几何感知细化.md](./GAR_几何感知细化.md) - 详细介绍邻近密化策略
- [ADM_自适应密度调制.md](./ADM_自适应密度调制.md) - 详细介绍 K-Planes 调制

---

*文档版本：v1.0 | 更新日期：2025-12-10*
