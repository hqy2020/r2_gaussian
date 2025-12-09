# SPS (Spatial Prior Seeding) 代码审查报告

> 审查日期: 2025-12-09
> 审查人: Claude Code

## 1. 概述

SPS (空间先验播种) 是 SPAGS 的初始化模块，通过密度加权采样从 FDK 重建体积中提取初始点云，为后续训练提供更好的空间先验。

## 2. 相关文件

| 文件 | 用途 |
|------|------|
| `initialize_pcd.py` | **核心实现** - InitParams 类和 init_pcd() 函数 |
| `r2_gaussian/arguments/__init__.py` | 参数定义 (ModelParams.ply_path) |
| `r2_gaussian/gaussian/initialize.py` | 点云加载逻辑 |
| `r2_gaussian/gaussian/gaussian_model.py` | create_from_pcd() 方法 |
| `cc-agent/scripts/run_spags_ablation.sh` | 训练脚本 (SPS 点云路径配置) |

## 3. 超参数列表

### 3.1 InitParams 类定义 (`initialize_pcd.py:27-75`)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_sps` | bool | `False` | **主开关**: 启用 SPS 密度加权初始化 |
| `sps_denoise` | bool | `False` | 启用高斯滤波降噪预处理 |
| `sps_denoise_sigma` | float | `3.0` | 高斯滤波核标准差 |
| `sps_strategy` | str | `"density_weighted"` | 采样策略 |
| `recon_method` | str | `"fdk"` | 重建方法 (fdk/random) |
| `n_points` | int | `50000` | 初始化点数 |
| `density_thresh` | float | `0.05` | 密度阈值，过滤低密度体素 |
| `density_rescale` | float | `0.15` | 密度缩放因子 |

### 3.2 向下兼容参数

| 新参数 | 旧参数 | 说明 |
|--------|--------|------|
| `sps_denoise` | `enable_denoise` | 降噪开关 |
| `sps_denoise_sigma` | `denoise_sigma` | 降噪 sigma |
| `sps_strategy` | `sampling_strategy` | 采样策略 |

## 4. 采样策略

### 4.1 Random (Baseline)
```python
sampled_idx = np.random.choice(len(valid_indices), n_points, replace=False)
```
均匀随机采样，不利用密度信息。

### 4.2 Density-Weighted (推荐)
```python
probs = densities_flat / densities_flat.sum()
sampled_idx = np.random.choice(len(valid_indices), n_points, replace=False, p=probs)
```
概率 `P(x) = ρ(x) / Σρ`，高密度区域获得更多采样点。

### 4.3 Stratified
按密度百分位数分 5 层，每层均匀采样 `n_points // 5` 个点。

## 5. 数据流

```
initialize_pcd.py --enable_sps -s <data>
    ↓
FDK 重建 → vol (3D 体积)
    ↓
[可选] 高斯滤波降噪
    ↓
密度阈值掩码: mask = vol > density_thresh
    ↓
采样策略选择 → sampled_idx
    ↓
保存 init_*.npy [N, 4] = [x, y, z, density]
    ↓
train.py --ply_path init_*.npy
    ↓
GaussianModel.create_from_pcd()
```

## 6. 训练脚本配置

`run_spags_ablation.sh` 第 77 行:
```bash
SPS_PCD_PATH="data/density-369/init_${ORGAN}_50_${VIEWS}views.npy"
```

当 `USE_SPS=true` 时，添加 `--ply_path $SPS_PCD_PATH`。

## 7. 发现的问题

### 7.1 参数兼容性逻辑错误 (中等风险) - ✅ 已修复

**位置**: `initialize_pcd.py:109-113`

**原问题**: 使用 `or` 操作符在新参数值为 `0` 或 `False` 时会错误回退到旧参数。

**修复方案** (2025-12-09):
```python
# 修复：使用 None 检查而非 or，避免值为 0/False 时错误回退
_sps_denoise = getattr(args, 'sps_denoise', None)
enable_denoise = _sps_denoise if _sps_denoise is not None else getattr(args, 'enable_denoise', False)
_sps_sigma = getattr(args, 'sps_denoise_sigma', None)
denoise_sigma = _sps_sigma if _sps_sigma is not None else getattr(args, 'denoise_sigma', 3.0)
```

### 7.2 密度阈值可能过高 (低风险)

**位置**: `initialize_pcd.py:54`

**问题**: 默认 `density_thresh=0.05` 对于稀疏器官 (如 pancreas) 可能导致有效体素数量不足。

**现象**: 如果有效体素 < n_points，会抛出异常。

**诊断方法**: 查看初始化输出:
```
[SPS] 有效体素数(>0.05): X
[SPS] 有效体素占比: Y%
```

**建议**:
- 降低默认值到 0.01-0.02
- 或按器官自适应调整

### 7.3 SPS 点云路径硬编码 (中等风险)

**位置**: `run_spags_ablation.sh:77`

**问题**: 路径 `data/density-369/init_${ORGAN}_50_${VIEWS}views.npy` 硬编码，如果用户使用不同目录会出错。

**建议**: 添加自动检测或参数化配置。

### 7.4 分层采样补充逻辑 (低风险)

**位置**: `initialize_pcd.py:184-193`

**问题**: 当某层级点数不足时，使用随机补充破坏了分层设计目的。

**建议**: 使用加权补充或从相邻层级借用。

## 8. 代码质量评估

### 优点
- 完善的诊断输出（体积统计、采样信息）
- 支持多种采样策略
- 向下兼容旧参数名

### 改进点
- 参数兼容性逻辑需要修复
- 缺少对不同器官的自适应阈值
- 可添加点云质量验证

## 9. 当前配置状态

**训练脚本中 SPS 配置**:
- 使用预生成的密度加权点云 (`data/density-369/`)
- 无需运行时参数（仅 `--ply_path`）

**状态**: 功能正常，但存在边缘情况下的潜在问题。
