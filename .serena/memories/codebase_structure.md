# R²-Gaussian 代码库结构

## 项目根目录

```
r2_gaussian/
├── r2_gaussian/              # 核心代码模块
├── cc-agent/                 # AI 科研助手系统
├── data/                     # 训练数据集
├── output/                   # 训练输出
├── scripts/                  # 工具脚本
├── model/                    # 预训练模型
├── data_generator/           # 数据生成工具
├── TIGRE-2.3/               # TIGRE CT 重建库
├── assets/                   # 项目资源文件
└── [各种工具脚本和配置文件]
```

## 核心代码模块 (`r2_gaussian/`)

### 主要子模块

```
r2_gaussian/
├── gaussian/                 # Gaussian 模型核心
│   ├── gaussian_model.py    # 高斯模型定义
│   ├── render_query.py      # 渲染查询接口
│   ├── initialize.py        # 初始化逻辑
│   └── __init__.py
├── utils/                    # 工具函数库
│   ├── loss_utils.py        # 损失函数
│   ├── camera_utils.py      # 相机工具
│   ├── depth_utils.py       # 深度处理
│   ├── graphics_utils.py    # 图形学工具
│   ├── gaussian_utils.py    # 高斯工具
│   ├── image_utils.py       # 图像工具
│   ├── general_utils.py     # 通用工具
│   ├── system_utils.py      # 系统工具
│   ├── log_utils.py         # 日志工具
│   ├── cfg_utils.py         # 配置工具
│   ├── argument_utils.py    # 参数工具
│   ├── pose_utils.py        # 位姿工具
│   ├── warp_utils.py        # 变形工具
│   ├── depth_estimator.py   # 深度估计
│   ├── sghmc_optimizer.py   # SGHMC 优化器
│   ├── fsgs_proximity_optimized.py  # FSGS 邻近优化
│   ├── sss_helpers.py       # SSS 辅助函数
│   ├── sss_utils.py         # SSS 工具
│   ├── corgs_metrics.py     # CoR-GS 指标
│   └── advanced_pseudo_label.py  # 高级伪标签
├── dataset/                  # 数据集加载器
├── arguments/                # 命令行参数定义
└── submodules/              # CUDA 子模块
    ├── xray-gaussian-rasterization-voxelization/
    └── simple-knn/
```

## 顶层脚本

### 训练与测试
- **train.py**：主训练脚本
- **test.py**：测试评估脚本
- **train_examples.py**：训练示例

### 初始化
- **initialize_pcd.py**：点云初始化（从 FDK 重建采样）

### 可视化与分析
- **generate_depth_maps.py**：生成深度图
- **visualize_depth_map.py**：深度图可视化
- **plot_eval_curve_psnr_ssim.py**：绘制评估曲线
- **depth_usage_example.py**：深度使用示例
- **test_depth.py**：深度测试
- **test_opacity_decay.py**：不透明度衰减测试

### 数据转换
- **convert_abdomen_to_r2_format.py**：腹部数据格式转换

### 批处理脚本
- **batch_run_2views.sh**：2 视角批量训练
- **batch_run_4views.sh**：4 视角批量训练
- **run_fsgs_fixed.sh**：FSGS 训练脚本

## cc-agent 科研助手系统

```
cc-agent/
├── medical_expert/          # 医学 CT 影像专家
├── 3dgs_expert/             # 3D Gaussian Splatting 专家
├── code/                    # PyTorch/CUDA 编程专家
├── experiments/             # 深度学习调参专家
├── records/                 # 项目进度跟踪
│   ├── progress.md          # 当前进度
│   ├── decision_log.md      # 决策日志
│   ├── knowledge_base.md    # 知识库
│   ├── project_timeline.md  # 项目时间线
│   └── archives/            # 历史归档
└── 论文/                    # 论文库
    ├── 待读/
    ├── 正在读/
    └── 已归档/
```

## 数据目录

```
data/
├── 369/                      # 3/6/9 视角数据集
│   ├── foot_50_3views.pickle
│   ├── init_foot_50_3views.npy
│   └── ...
├── synthetic_dataset/        # 合成数据集
└── real_dataset/            # 真实数据集
```

## 输出目录

```
output/
└── {experiment_name}/       # 如 footdd_3_1117
    ├── cfg_args             # 训练配置
    ├── cfg_args.yml
    ├── ckpt/                # 检查点
    ├── eval/                # 评估结果
    ├── depth_maps/          # 深度图输出
    ├── point_cloud/         # 点云数据
    └── events.out.tfevents.*  # TensorBoard 日志
```

## 关键入口点

1. **训练入口**：`train.py`
2. **测试入口**：`test.py`
3. **初始化入口**：`initialize_pcd.py`
4. **Gaussian 模型**：`r2_gaussian/gaussian/gaussian_model.py`
5. **渲染器**：`r2_gaussian/gaussian/render_query.py`