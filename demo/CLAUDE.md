# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**智能辅助诊断平台**：毕业论文《面向稀疏数据的医学影像场景理解和重建方法研究》的演示系统。

### 核心功能模块
1. **CT 三维重建**：基于 SPAGS (3DGS) 的稀疏视角 CT 重建（3/6/9 views）
2. **语义分割**：基于 UNet + SAM 的半监督医学图像分割（心脏 ACDC 数据）

## 重要约定

- **所有回复和写入文档的内容都是中文**
- **CUDA 环境**: `r2_gaussian_new`
- **端口**: 7860 (Gradio 默认)
- **优先使用预设案例测试**，避免频繁上传大文件

## 常用命令

### 启动系统
```bash
conda activate r2_gaussian_new
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 简化版（推荐）
python demo/app_simple.py

# 完整版
python demo/app.py
```

### 安装依赖
```bash
pip install -r requirements_demo.txt
python -c "import gradio; import plotly; print('依赖安装成功!')"
```

### 访问地址
- 本地: http://localhost:7860
- 局域网: http://<服务器IP>:7860

## 代码架构

```
demo/
├── app.py                    # 完整版主应用
├── app_simple.py             # 简化版主应用（推荐）
├── config.py                 # 配置：路径、预设案例、Gradio主题
│
├── backend/                  # 后端推理模块
│   ├── reconstruction.py     # ReconstructionInference 类
│   ├── segmentation.py       # SegmentationInference 类
│   └── data_manager.py       # DataManager 预设案例管理
│
├── visualization/            # 可视化模块
│   ├── slice_viewer.py       # 切片浏览器 (Plotly)
│   ├── volume_render.py      # 体绘制 (Marching Cubes)
│   └── surface_render.py     # 表面重建 (多器官3D)
│
├── ui/                       # Gradio UI 组件
│   ├── reconstruction_tab.py # 重建界面 Tab
│   └── segmentation_tab.py   # 分割界面 Tab
│
└── data/                     # 数据目录
    ├── presets/              # 预设案例
    └── cache/                # 缓存文件
```

### 核心类说明

| 类名 | 文件 | 职责 |
|------|------|------|
| `ReconstructionInference` | backend/reconstruction.py | 加载 R²-Gaussian 模型，执行 3D 体积重建 |
| `SegmentationInference` | backend/segmentation.py | 加载 UNet 模型，执行心脏 MRI 分割 |
| `DataManager` | backend/data_manager.py | 管理预设案例，处理数据加载 |
| `SliceViewer` | visualization/slice_viewer.py | 交互式切片浏览 |
| `VolumeRenderer` | visualization/volume_render.py | 3D 等值面体绘制 |
| `SurfaceRenderer` | visualization/surface_render.py | 多器官表面重建 |

### 数据流

1. **重建流程**: 选择预设 → `DataManager.load_preset()` → `ReconstructionInference.load_model()` → `query()` 体积重建 → 可视化
2. **分割流程**: 选择 ACDC 案例 → 加载 H5 数据 → `SegmentationInference.segment()` → 可视化 + Dice 评估

## 配置文件说明 (config.py)

```python
# 关键路径配置
SEGMENTATION_MODEL_PATH = "/home/qyhu/Documents/new_PG_semi/exp/acdc/CP_SAM/unet/7/ema_0/best.pth"
ACDC_DATA_PATH = "/home/qyhu/ACDC/data"

# 预设案例
RECONSTRUCTION_PRESETS = [...]  # 足部 3/6/9 视角
SEGMENTATION_PRESETS = [...]    # ACDC 患者案例
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 前端框架 | Gradio 4.x | 快速构建 Web UI |
| 3D 可视化 | Plotly | 交互式切片和体绘制 |
| 后端推理 | PyTorch | 深度学习模型推理 |
| 图像处理 | NumPy, scikit-image | Marching Cubes 等 |
| 医学格式 | nibabel | NIfTI 导出支持 |

## 开发进度追踪

### 已完成功能 ✅
- 项目结构搭建、重建推理接口、分割推理接口
- 切片浏览/体绘制/表面重建可视化
- Gradio UI (重建 Tab + 分割 Tab)
- NPY/NIfTI 导出、使用说明文档

### 待开发功能（按优先级）

| 优先级 | 功能 | 说明 |
|--------|------|------|
| P0 | DICOM 格式支持 | pydicom 解析 |
| P0 | 批量处理功能 | 批量上传和处理 |
| P0 | 错误处理完善 | 友好的异常提示 |
| P1 | 交互式分割编辑 | 画笔工具修正 |
| P1 | GT 对比功能 | 真值与预测对比 |
| P2 | Flask 后端重构 | 前后端分离 |
| P2 | Docker 部署 | 容器化 |

详细进度见 `todolist.md`

## 技术债务

| 问题 | 优先级 | 解决方案 |
|------|--------|----------|
| Gradio 版本兼容性 | 高 | 锁定版本 |
| 全局状态管理 | 中 | 改用会话状态 |
| 硬编码路径 | 中 | 使用 config.py |
| 缺少单元测试 | 低 | 添加 pytest |

## 调试技巧

### 常见问题排查
```bash
# 检查 GPU 状态
nvidia-smi

# 检查模型文件是否存在
ls -la output/2025_11_17_foot_3views_baseline_30k/

# 检查分割模型
ls -la /home/qyhu/Documents/new_PG_semi/exp/acdc/CP_SAM/unet/7/ema_0/best.pth
```

### 日志输出
- Gradio 会在终端输出详细的请求日志
- 模型加载和推理错误会显示在终端

## 相关文档

- **使用说明书**: `README.md`
- **开发计划**: `todolist.md`
- **父项目 CLAUDE.md**: `../CLAUDE.md`
- **R²-Gaussian 核心代码**: `../r2_gaussian/`
