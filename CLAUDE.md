# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPOERTANT
neo4j 的mcp一定是在neo4j数据库里面的
## 项目概述

**SPAGS**: 基于 3D Gaussian Splatting 的 CT 断层扫描重建项目。核心目标是稀疏视角（3/6/9 views）新视角合成。

## 重要约定

- **所有回复和写入文档的内容都是中文**
- **训练命名格式**: `yyyy_MM_dd_HH_mm_organ_{{nums}}views_{{technique}}`
- **CUDA 环境**: `r2_gaussian_new`
- **数据集位置**: `data/369/`（3/6/9 稀疏视角数据）
- **多使用 serena MCP 理解代码，修改代码**
- **尽可能确保都是有专门的助手 agent 执行具体流程**

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
IMPORTANT
### 训练
```bash
# SPAGS 消融实验脚本（推荐）
# 用法: ./cc-agent/scripts/run_spags_ablation.sh <配置> <器官> <视角数> [GPU]
#
# 配置选项:
#   baseline  - Baseline (无任何技术)
#   sps       - 仅 SPS (空间先验播种)
#   gar       - 仅 GAR (几何感知细化)
#   adm       - 仅 ADM (自适应密度调制)
#   sps_gar   - SPS + GAR
#   sps_adm   - SPS + ADM
#   gar_adm   - GAR + ADM
#   spags     - Full SPAGS (SPS + GAR + ADM)
#
# 器官: foot, chest, head, abdomen, pancreas
# 视角: 3, 6, 9
#
# 示例:
./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0
./cc-agent/scripts/run_spags_ablation.sh baseline chest 6 1
./cc-agent/scripts/run_spags_ablation.sh sps_gar pancreas 9 0

# 直接使用 train.py（不推荐，仅用于调试）
python train.py -s data/369/foot_50_3views.pickle -m output/experiment_name
```

### 初始化点云
```bash
python initialize_pcd.py --data <path_to_data>
python initialize_pcd.py --data <path_to_data> --evaluate  # 评估初始化质量
```


## 代码架构

```
r2_gaussian/
├── train.py                    # 主训练入口
├── test.py                     # 测试评估入口
├── initialize_pcd.py           # 点云初始化（FDK 采样）
├── r2_gaussian/                # 核心代码模块
│   ├── gaussian/
│   │   ├── gaussian_model.py   # GaussianModel 类：Gaussian 参数管理
│   │   ├── render_query.py     # render() / query() 函数
│   │   ├── kplanes.py          # K-Planes 编码器（可选特征增强）
│   │   └── initialize.py       # 初始化逻辑
│   ├── dataset/
│   │   ├── dataset_readers.py  # NAF/NeRF 格式数据加载
│   │   └── cameras.py          # Camera 类定义
│   ├── utils/
│   │   ├── loss_utils.py       # L1/SSIM/TV 损失函数
│   │   ├── binocular_utils.py  # 双目一致性损失
│   │   └── ...
│   ├── arguments/              # 命令行参数定义（ModelParams/OptimizationParams）
│   ├── innovations/fsgs/       # FSGS 邻近密集化模块
│   └── submodules/             # CUDA 扩展
│       ├── xray-gaussian-rasterization-voxelization/  # X 射线光栅化
│       └── simple-knn/         # KNN 搜索
├── cc-agent/                   # AI 科研助手系统
├── data/369/                   # 稀疏视角数据集
└── output/                     # 训练输出
```

### 关键数据流

1. **数据加载**: `Scene` → `dataset_readers.py` → NAF/NeRF 格式解析
2. **初始化**: `initialize_pcd.py` → FDK 重建 → 点云采样 → `init_*.npy`
3. **训练循环**: `train.py` → `GaussianModel` → `render()` → 投影 → 损失计算 → 优化
4. **体积查询**: `query()` → 体素化 Gaussian → 3D 体积重建

### GaussianModel 核心属性

- `_xyz`: 高斯中心位置 (world coordinates)
- `_scaling`: 3D 尺度参数
- `_rotation`: 旋转四元数
- `_density`: 密度参数（Softplus 激活）

## 技术栈

- **Python 3.9** + **PyTorch 1.12.1** + **CUDA 11.6**
- **TIGRE 2.3**: CT 数据生成和 FDK 重建
- **Open3D**: 3D 数据处理
- **TensorBoard**: 训练可视化

### CUDA 扩展编译
```bash
pip install -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
pip install -e r2_gaussian/submodules/simple-knn
```

## Memory 工具使用

- 使用 Neo4j 数据库存储项目记忆
- 每次会话开始时：(1) 切换到 neo4j 数据库 (2) 搜索相关记忆

### 记忆存储规范
- 单个 observation < 150 字
- 每个 memory ≤ 3 个 observations
- 保持类型纯度（不混合 issue/decision/implementation）
- 详细指南见 `cc-agent/MCP工具使用指南.md` 和 `cc-agent/记忆模板.md`

## 三阶段工作流

### 阶段一：分析问题
**声明格式**: `【分析问题】`
- 使用 `memory_find` 查找相关记忆
- 使用 serena MCP 的 `find_symbol` / `find_referencing_symbols` 分析代码
- **禁止**：修改代码、急于给方案

### 阶段二：制定方案
**声明格式**: `【制定方案】`
- 使用 `memory_store` 存储新发现的知识/决策
- 列出变更文件和描述
- 需要决策时向用户提问

### 阶段三：执行方案
**声明格式**: `【执行方案】`
- 严格按方案实现
- 执行后记录实现结果为 `implementation` 类型记忆
- **禁止**：自动提交代码

## 科研助手团队

```
cc-agent/
├── medical_expert/     # 医学 CT 影像专家
├── 3dgs_expert/        # 3D Gaussian Splatting 专家
├── code/               # PyTorch/CUDA 编程专家
├── experiments/        # 深度学习调参专家
├── records/            # 进度跟踪（progress.md）
└── 论文/               # 论文库
```

## 详细指南

- **完整工作流**: `cc-agent/工作流详细指南.md`
- **MCP 工具使用**: `cc-agent/MCP工具使用指南.md`
- **代码结构**: `.serena/memories/codebase_structure.md`
- **常用命令**: `.serena/memories/suggested_commands.md`
