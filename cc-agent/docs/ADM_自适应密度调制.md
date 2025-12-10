# ADM 自适应密度调制技术文档

> **ADM**: **A**daptive **D**ensity **M**odulation
>
> 使用 K-Planes 学习位置相关的密度修正

---

## 1. Motivation：为什么需要自适应密度调制？

### 1.1 问题：全局统一的密度激活无法自适应

标准 3DGS 使用全局统一的激活函数（如 Softplus）将密度参数映射到实际密度值。

但在 CT 重建中，不同解剖结构对密度的需求不同：

```
密度响应的差异
├── 骨骼区域
│   ├── 需要更高的密度值
│   ├── 边界需要更清晰
│   └── 细节需要更丰富
│
├── 软组织区域
│   ├── 密度变化更平缓
│   ├── 需要平滑过渡
│   └── 避免产生伪边界
│
└── 空气/背景区域
    ├── 密度应该接近零
    ├── 不需要额外调制
    └── 避免引入噪声
```

### 1.2 核心洞察：密度调整应该是位置相关的

每个空间位置应该有自己的密度修正因子，而不是统一的变换。

### 1.3 ADM 的解决方案

**核心思想**：使用 K-Planes 学习**位置相关的密度调制因子**

$$
\rho' = \rho \times (1 + \text{offset} \times \text{confidence} \times \text{max\_range})
$$

- **K-Planes**：3 个正交平面编码空间位置特征
- **MLP 解码器**：将特征映射到 (offset, confidence)
- **调制范围**：限制在 ±30%，避免极端值

---

## 2. 架构设计

### 2.1 整体结构

```
高斯位置 (x, y, z)
        ↓
┌───────────────────────────────────────┐
│           K-Planes 编码器              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Plane XY │ │ Plane XZ │ │ Plane YZ │  │
│  │ (x, y)   │ │ (x, z)   │ │ (y, z)   │  │
│  └────┬────┘ └────┬────┘ └────┬────┘  │
│       │           │           │       │
│       └───────────┼───────────┘       │
│                   ↓                   │
│            Concat [96-dim]            │
└───────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────┐
│           MLP 解码器                   │
│  ┌─────────────────────────────────┐  │
│  │        Backbone (3 层)           │  │
│  └───────────┬─────────────────────┘  │
│              │                        │
│    ┌─────────┴─────────┐              │
│    ↓                   ↓              │
│ ┌────────┐       ┌────────────┐       │
│ │ Offset │       │ Confidence │       │
│ │ [-1,1] │       │   [0,1]    │       │
│ └────────┘       └────────────┘       │
└───────────────────────────────────────┘
                    ↓
            密度调制因子
```

### 2.2 K-Planes 编码器

将 3D 坐标分解到 3 个 2D 平面：

```python
class KPlanesEncoder(nn.Module):
    def __init__(
        self,
        grid_resolution: int = 64,    # 平面分辨率
        feature_dim: int = 32,         # 每个平面的特征维度
        bounds: Tuple[float, float] = (-1.0, 1.0),  # 场景范围
    ):
        # 初始化 3 个可学习的平面
        self.plane_xy = nn.Parameter(...)  # [1, 32, 64, 64]
        self.plane_xz = nn.Parameter(...)  # [1, 32, 64, 64]
        self.plane_yz = nn.Parameter(...)  # [1, 32, 64, 64]

    def forward(self, xyz):
        # 归一化坐标到 [-1, 1]
        xyz_normalized = self._normalize_coords(xyz)

        # 双线性插值提取每个平面的特征
        feat_xy = F.grid_sample(self.plane_xy, ...)  # [N, 32]
        feat_xz = F.grid_sample(self.plane_xz, ...)  # [N, 32]
        feat_yz = F.grid_sample(self.plane_yz, ...)  # [N, 32]

        # 拼接
        features = torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)
        return features  # [N, 96]
```

### 2.3 MLP 解码器（双头输出）

```python
class DensityMLPDecoder(nn.Module):
    def __init__(self, input_dim=96, hidden_dim=128, num_layers=3):
        # 共享 backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 双头输出
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, kplanes_feat):
        features = self.backbone(kplanes_feat)
        offset = self.offset_head(features)      # [-1, 1]
        confidence = self.confidence_head(features)  # [0, 1]
        return offset, confidence
```

**双头设计的意义**：
- **offset**：控制密度调制的方向（+增加，-减少）
- **confidence**：网络自动学习的调制强度
  - 困难区域 → confidence ≈ 1 → 强调制
  - 简单区域 → confidence ≈ 0 → 弱调制

---

## 3. 核心公式

### 3.1 密度调制

```python
# 在 GaussianModel.get_density 中

# 1. 获取基础密度
base_density = self.density_activation(self._density)

# 2. 获取 K-Planes 特征
kplanes_feat = self.get_kplanes_features()  # [N, 96]

# 3. 解码得到 offset 和 confidence
offset, confidence = self.density_decoder(kplanes_feat)

# 4. 计算有效调制
effective_offset = offset * confidence * max_range * strength * view_scale

# 5. 应用调制
modulation = 1.0 + effective_offset
adjusted_density = base_density * modulation
```

简化公式：

```
调制后密度 = 原始密度 × (1 + offset × confidence × max_range × strength × view_scale)
```

### 3.2 各因子的含义

| 因子 | 范围 | 说明 |
|------|------|------|
| `offset` | [-1, 1] | 调制方向，+增加/-减少密度 |
| `confidence` | [0, 1] | 网络学习的调制强度 |
| `max_range` | 0.3 | 最大调制范围（±30%） |
| `strength` | 0→1→0.5 | 训练进程调度（见下文） |
| `view_scale` | 1.0/0.71/0.58 | 视角自适应（3/6/9 views） |

---

## 4. 训练调度

### 4.1 三阶段调度

ADM 使用三阶段调度策略：

```
迭代次数:  0        3000      20000     30000
           │         │          │         │
强度值:    0 ──────→ 1 ──────── 1 ──────→ 0.5
           │         │          │         │
阶段:    Warmup    正常调制    渐进衰减
```

代码实现（`gaussian_model.py`）：

```python
def _get_adm_strength(self):
    it = self.current_iteration
    warmup = 3000
    decay_start = 20000
    final = 0.5
    total = 30000

    if it < warmup:
        # 阶段 1：线性 warmup
        return it / warmup
    elif it < decay_start:
        # 阶段 2：保持 1.0
        return 1.0
    else:
        # 阶段 3：线性衰减
        progress = (it - decay_start) / (total - decay_start)
        return 1.0 - (1.0 - final) * min(1.0, progress)
```

**阶段设计原理**：
- **Warmup**：避免初期调制干扰基础收敛
- **正常调制**：充分利用 ADM 的自适应能力
- **渐进衰减**：后期减弱调制，让模型稳定收敛

### 4.2 视角自适应

```python
def get_view_adaptive_scale(self) -> float:
    """
    视角越少 → 需要更强的调制（补偿信息不足）
    视角越多 → 需要更弱的调制（避免干扰监督信号）
    """
    scale = 1.0 / math.sqrt(self.num_train_views / 3.0)
    return max(scale, 0.3)  # 最小值 0.3

# 效果：
# 3-views: scale = 1.0（基准，强调制）
# 6-views: scale ≈ 0.71（中等调制）
# 9-views: scale ≈ 0.58（弱调制）
```

---

## 5. TV 正则化

### 5.1 为什么需要 TV 正则化？

K-Planes 有大量可学习参数（64×64×32×3 ≈ 400k），容易过拟合。

TV（Total Variation）正则化鼓励平面特征平滑，防止过拟合。

### 5.2 TV 损失公式

```python
def compute_plane_tv(plane: torch.Tensor) -> torch.Tensor:
    """
    计算单个平面的 TV 损失
    """
    batch_size, c, h, w = plane.shape

    # 水平梯度
    grad_w = plane[:, :, :, 1:] - plane[:, :, :, :-1]
    # 垂直梯度
    grad_h = plane[:, :, 1:, :] - plane[:, :, :-1, :]

    # L2 TV 损失
    tv_w = torch.square(grad_w).sum()
    tv_h = torch.square(grad_h).sum()
    tv_loss = 2 * (tv_w / count_w + tv_h / count_h)

    return tv_loss
```

### 5.3 在训练中应用

```python
# train.py

if opt.lambda_plane_tv > 0 and gaussians.enable_kplanes:
    planes = gaussians.kplanes_encoder.get_plane_params()
    tv_loss = compute_plane_tv_loss(planes)

    # 视角自适应 TV 权重
    view_scale = gaussians.get_view_adaptive_scale()
    effective_lambda = opt.lambda_plane_tv * view_scale

    loss["total"] += effective_lambda * tv_loss
```

---

## 6. 超参数设置

### 6.1 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_adm` / `enable_kplanes` | False | 启用 ADM |
| `adm_resolution` | 64 | K-Planes 平面分辨率 |
| `adm_feature_dim` | 32 | K-Planes 特征维度 |
| `adm_decoder_hidden` | 128 | MLP 隐藏层维度 |
| `adm_decoder_layers` | 3 | MLP 层数 |
| `adm_max_range` | 0.3 | 最大调制范围（±30%）|
| `adm_view_adaptive` | True | 启用视角自适应 |

### 6.2 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adm_lr_init` | 0.002 | K-Planes 初始学习率 |
| `adm_lr_final` | 0.0002 | K-Planes 最终学习率 |
| `adm_lr_max_steps` | 30000 | 学习率衰减步数 |
| `adm_lambda_tv` | 0.002 | TV 正则化权重 |
| `adm_tv_type` | "l2" | TV 损失类型 |
| `adm_warmup_iters` | 3000 | Warmup 迭代数 |
| `adm_decay_start` | 20000 | 调制衰减开始迭代 |
| `adm_final_strength` | 0.5 | 最终调制强度 |

---

## 7. 使用方法

### 7.1 在训练中启用 ADM

```bash
# 方法 1：手动指定参数
python train.py \
    -s data/369/foot_50_3views.pickle \
    --enable_kplanes \
    --adm_resolution 64 \
    --adm_feature_dim 32 \
    --adm_lambda_tv 0.002

# 方法 2：使用消融脚本（推荐）
./cc-agent/scripts/run_spags_ablation.sh adm foot 3 0
```

### 7.2 与其他模块组合

```bash
# SPS + ADM
./cc-agent/scripts/run_spags_ablation.sh sps_adm foot 3 0

# GAR + ADM
./cc-agent/scripts/run_spags_ablation.sh gar_adm foot 3 0

# 完整 SPAGS (SPS + GAR + ADM)
./cc-agent/scripts/run_spags_ablation.sh spags foot 3 0
```

### 7.3 监控 ADM 效果

ADM 会在 TensorBoard 中记录诊断信息：

| 指标 | 说明 |
|------|------|
| `adm/strength` | 当前调度强度 |
| `adm/view_scale` | 视角自适应缩放因子 |
| `adm/offset_mean` | offset 均值 |
| `adm/confidence_mean` | confidence 均值 |
| `adm/modulation_std` | 调制因子标准差 |
| `adm/density_change_pct` | 密度变化百分比 |

---

## 8. 代码位置索引

| 功能 | 文件 | 位置 |
|------|------|------|
| 参数定义 | `r2_gaussian/arguments/__init__.py` | ModelParams 和 OptimizationParams 中的 ADM 参数 |
| KPlanesEncoder | `r2_gaussian/gaussian/kplanes.py` | 完整实现 |
| DensityMLPDecoder | 同上 | 完整实现 |
| 密度调制应用 | `r2_gaussian/gaussian/gaussian_model.py` | 第 244-273 行 `get_density` |
| 训练调度 | 同上 | 第 275-298 行 `_get_adm_strength` |
| 视角自适应 | 同上 | `get_view_adaptive_scale()` |
| TV 正则化 | `r2_gaussian/utils/regulation.py` | `compute_plane_tv_loss()` |
| 诊断信息 | `r2_gaussian/gaussian/gaussian_model.py` | 第 300-361 行 `get_adm_diagnostics()` |

---

## 9. 注意事项

### 9.1 调制范围的选择

`adm_max_range = 0.3` 表示密度最多变化 ±30%。

- 太大：可能产生极端密度值，破坏重建质量
- 太小：调制效果不明显

### 9.2 与 SPS 的协同

SPS 提供合理的初始密度分布，ADM 在此基础上学习微调：
- SPS：决定每个点的初始密度量级
- ADM：根据空间位置进行 ±30% 的精细调整

### 9.3 训练稳定性

如果训练不稳定：
1. 增加 `adm_warmup_iters`（延长 warmup）
2. 减小 `adm_max_range`（限制调制范围）
3. 增加 `adm_lambda_tv`（更强的正则化）

---

*文档版本：v1.0 | 更新日期：2025-12-10*
