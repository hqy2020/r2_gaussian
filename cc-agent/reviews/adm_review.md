# ADM (Adaptive Density Modulation) 代码审查报告

> 审查日期: 2025-12-09
> 审查人: Claude Code

## 1. 概述

ADM (自适应密度调制) 通过 K-Planes 编码器和 MLP 解码器实现空间自适应的密度调制，是 SPAGS 的核心创新之一。

## 2. 相关文件

| 文件 | 用途 |
|------|------|
| `r2_gaussian/arguments/__init__.py` | 参数定义 (ModelParams + OptimizationParams) |
| `r2_gaussian/gaussian/gaussian_model.py` | ADM 集成到 GaussianModel |
| `r2_gaussian/gaussian/kplanes.py` | KPlanesEncoder + DensityMLPDecoder |
| `r2_gaussian/utils/regulation.py` | Planes TV 损失函数 |
| `train.py` | ADM 调用和训练循环集成 |

## 3. 超参数列表

### 3.1 ModelParams 中的参数 (`arguments/__init__.py`)

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_adm` | bool | `False` | **主开关**: 启用 ADM |
| `adm_resolution` | int | `64` | K-Planes 平面分辨率 |
| `adm_feature_dim` | int | `32` | K-Planes 特征维度 |
| `adm_decoder_hidden` | int | `128` | MLP 隐藏层维度 |
| `adm_decoder_layers` | int | `3` | MLP 层数 |
| `adm_max_range` | float | `0.3` | 最大调制范围 (±30%) |
| `adm_view_adaptive` | bool | `True` | 视角自适应调制 |

### 3.2 OptimizationParams 中的参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `adm_lr_init` | float | `0.002` | K-Planes 初始学习率 |
| `adm_lr_final` | float | `0.0002` | K-Planes 最终学习率 |
| `adm_lr_max_steps` | int | `30000` | 学习率衰减步数 |
| `adm_lambda_tv` | float | `0.002` | Planes TV 正则化权重 |
| `adm_tv_type` | str | `"l2"` | TV 损失类型 |
| `adm_warmup_iters` | int | `3000` | Warmup 迭代数 |
| `adm_decay_start` | int | `20000` | 调制衰减开始迭代 |
| `adm_final_strength` | float | `0.5` | 最终调制强度 |

### 3.3 向下兼容参数

| 新参数 | 旧参数 |
|--------|--------|
| `enable_adm` | `enable_kplanes` |
| `adm_resolution` | `kplanes_resolution` |
| `adm_feature_dim` | `kplanes_dim` |
| `adm_lambda_tv` | `lambda_plane_tv` |

## 4. 核心架构

### 4.1 KPlanesEncoder

```
xyz [N, 3]
    ↓ 坐标归一化到 [-1, 1]
    ↓ 双线性插值
plane_xy [1, 32, 64, 64] → feat_xy [N, 32]
plane_xz [1, 32, 64, 64] → feat_xz [N, 32]
plane_yz [1, 32, 64, 64] → feat_yz [N, 32]
    ↓ concat
features [N, 96]
```

### 4.2 DensityMLPDecoder (双头)

```
features [N, 96]
    ↓ backbone (3 层 Linear + ReLU)
hidden [N, 128]
    ↓
    ├── offset_head → Tanh → offset [N, 1] ∈ [-1, 1]
    └── confidence_head → Sigmoid → confidence [N, 1] ∈ [0, 1]
```

### 4.3 密度调制公式

```python
effective_offset = offset * confidence * max_range * strength * view_scale
modulation = 1.0 + effective_offset
final_density = base_density * modulation
```

其中:
- `offset` ∈ [-1, 1]: 调制方向
- `confidence` ∈ [0, 1]: 置信度权重
- `max_range` = 0.3: 最大调制幅度
- `strength` ∈ [0, 1]: 训练进程调度
- `view_scale` ∈ [0.3, 1.0]: 视角自适应

## 5. 训练进程调度

```
strength
1.0  |   ╱─────────╲
     |  ╱           ╲
0.5  | ╱             ╲___
     |╱__________________
     0   3000  20000  30000  (iterations)
          ↑     ↑
       warmup  decay_start
```

| 阶段 | 迭代范围 | strength | 目的 |
|------|---------|----------|------|
| Warmup | 0-3000 | 0→1 | 避免初期干扰 |
| Hold | 3000-20000 | 1.0 | 正常调制 |
| Decay | 20000-30000 | 1→0.5 | 稳定最终结果 |

## 6. 视角自适应

### 6.1 ADM 调制强度缩放

```python
view_scale = 1.0 / sqrt(num_views / 3.0)
view_scale = max(view_scale, 0.3)  # 下限保护
```

| 视角数 | view_scale | 说明 |
|--------|------------|------|
| 3 | 1.0 | 强调制 (基准) |
| 6 | 0.707 | 中等调制 |
| 9 | 0.577 | 弱调制 |

### 6.2 TV 正则化权重缩放

```python
tv_scale = 1.0 / view_scale  # 反向缩放
effective_lambda_tv = lambda_tv * tv_scale
```

| 视角数 | tv_scale | 有效 lambda_tv |
|--------|----------|---------------|
| 3 | 1.0 | 0.002 |
| 6 | 1.414 | 0.00283 |
| 9 | 1.732 | 0.00346 |

**逻辑**: 视角越多 → TV 权重越大 → 防止过拟合

## 7. 训练脚本配置

`run_spags_ablation.sh:94-96`:
```bash
ADM_FLAGS_COMPAT="--enable_kplanes"
```

**使用默认值**:
- resolution=64, dim=32
- lambda_plane_tv=0.002
- tv_type=l2

## 8. 发现的问题

### 8.1 Confidence 初始化过于保守 (低风险) - ✅ 已修复

**位置**: `kplanes.py:228-233`

**原问题**: 初始 confidence ≈ 0.27 (sigmoid(-1))，导致早期调制较弱。

**修复方案** (2025-12-09):
```python
# confidence_head 初始化（修复：提高初始值以加速收敛）
# 旧值 sigmoid(-1) ≈ 0.27 过于保守，需要较长 warmup 才能生效
# 新值 sigmoid(0) = 0.5，让网络从中性状态开始学习
# ADM 已有 warmup 机制保护，不需要 confidence 额外保守
nn.init.normal_(self.confidence_head[0].weight, std=0.01)
nn.init.constant_(self.confidence_head[0].bias, 0.0)  # sigmoid(0) = 0.5
```

### 8.2 ADM 最大范围硬编码 (中等风险)

**位置**: `gaussian_model.py:111`

```python
self.adm_max_range = getattr(args, 'adm_max_range', 0.3)
```

**问题**: 固定 ±30% 可能不适合所有器官:
- 困难器官 (pancreas): 30% 可能不够
- 简单器官 (foot): 30% 可能过度调制

**建议**: 按器官或视角数自适应调整

### 8.3 view_scale 最小值限制 (低风险)

**位置**: `gaussian_model.py:226`

```python
return max(scale, 0.3)
```

**问题**: 12+ 视角时，scale 被限制在 0.3，可能导致 ADM 作用不足。

**建议**: 对于高视角场景，可能需要重新评估限制值

### 8.4 Decoder 学习率设置 (低风险)

**位置**: `gaussian_model.py:482-487`

```python
decoder_lr_init = kplanes_lr_init * 0.5  # 0.001
```

**问题**: Decoder 参数量少但学习率也低，可能学习速度过慢。

**建议**: 监控 Decoder 梯度流和参数更新速度

### 8.5 TV 权重均匀设置 (低风险)

**位置**: `arguments/__init__.py:197`

```python
self.plane_tv_weight_proposal = [1.0, 1.0, 1.0]
```

**问题**: 三个平面重要性可能不同 (CT 轴向扫描 → XY 平面更重要)

**建议**: 考虑非均匀权重或自动学习

## 9. 代码质量评估

### 优点
- 完善的三阶段调度 (warmup → hold → decay)
- 双头输出设计 (offset + confidence)
- 视角自适应机制
- 完善的诊断输出 (`get_adm_diagnostics()`)

### 改进点
- 初始化策略可以更激进
- max_range 可以自适应
- TV 权重可以非均匀

## 10. 诊断输出示例

`train.py` 每 1000 次迭代输出:
```
[Iter 1000] === ADM 诊断 ===
  调度参数: strength=0.333, view_scale=1.000, max_range=0.300
  Gaussians数量: 50,000
  offset:     mean=+0.0012, std=0.0234
  confidence: mean=0.2845, std=0.0123
  eff_offset: mean=+0.000034, std=0.002145
  modulation: mean=1.0000, std=0.0021
  密度变化%:  mean=+0.00%, std=0.21%
```

## 11. 当前配置状态

**训练脚本中 ADM 配置**:
- `--enable_kplanes` (使用旧参数名)
- 全部使用默认值

**状态**: 功能正常，设计合理，存在一些可优化的边缘情况。
