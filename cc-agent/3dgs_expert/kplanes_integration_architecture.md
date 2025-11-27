# K-Planes 编码器集成架构说明

## 🎯 核心问题：K-Planes 加在哪里？

**简短回答**：K-Planes 编码器是加在 **`GaussianModel`** 类内部的，用于**增强每个 3D Gaussian 的 density（密度）表示**。

---

## 📊 架构对比图

### Baseline（原始 R²-Gaussian）

```
┌─────────────────────────────────────────────────────────┐
│              GaussianModel 类                            │
├─────────────────────────────────────────────────────────┤
│  存储的参数（每个 Gaussian）：                            │
│  • _xyz        : [N, 3]  位置                           │
│  • _scaling    : [N, 3]  尺度                           │
│  • _rotation   : [N, 4]  旋转（四元数）                  │
│  • _density    : [N, 1]  密度（原始值）                  │
├─────────────────────────────────────────────────────────┤
│  渲染时使用：                                            │
│  density = softplus(_density)  ← 直接激活                │
└─────────────────────────────────────────────────────────┘
                      ↓
            ┌─────────────────┐
            │  CUDA 渲染器     │
            │  (渲染 2D 投影)  │
            └─────────────────┘
```

### X²-Gaussian（加入 K-Planes）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GaussianModel 类（增强版）                          │
├─────────────────────────────────────────────────────────────────────────┤
│  原始参数（每个 Gaussian）：                                              │
│  • _xyz        : [N, 3]  位置                                           │
│  • _scaling    : [N, 3]  尺度                                           │
│  • _rotation   : [N, 4]  旋转（四元数）                                  │
│  • _density    : [N, 1]  密度（原始值）                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  🆕 新增模块（可选）：                                                    │
│  ┌───────────────────────────────────────────────────────┐              │
│  │  K-Planes Encoder（空间特征提取器）                    │              │
│  │  • plane_xy : [1, 32, 64, 64]  XY 平面特征            │              │
│  │  • plane_xz : [1, 32, 64, 64]  XZ 平面特征            │              │
│  │  • plane_yz : [1, 32, 64, 64]  YZ 平面特征            │              │
│  │                                                       │              │
│  │  forward(xyz) → features [N, 96]                     │              │
│  └───────────────────────────────────────────────────────┘              │
│                        ↓                                                │
│  ┌───────────────────────────────────────────────────────┐              │
│  │  MLP Decoder（特征解码器）                             │              │
│  │  • Linear(96 → 128) + ReLU                           │              │
│  │  • Linear(128 → 128) + ReLU                          │              │
│  │  • Linear(128 → 1) + Tanh                            │              │
│  │                                                       │              │
│  │  forward(features) → density_offset [-1, 1]          │              │
│  └───────────────────────────────────────────────────────┘              │
├─────────────────────────────────────────────────────────────────────────┤
│  渲染时使用（增强版）：                                                   │
│  base_density = softplus(_density)                                      │
│  ↓                                                                      │
│  kplanes_feat = kplanes_encoder(_xyz)        ← 提取空间特征             │
│  ↓                                                                      │
│  density_offset = density_decoder(kplanes_feat)  ← 解码为调制因子       │
│  ↓                                                                      │
│  modulation = 0.7 + 0.6 * sigmoid(density_offset)  ← 映射到 [0.7, 1.3] │
│  ↓                                                                      │
│  final_density = base_density * modulation   ← 调制原始密度             │
└─────────────────────────────────────────────────────────────────────────┘
                      ↓
            ┌─────────────────┐
            │  CUDA 渲染器     │
            │  (渲染 2D 投影)  │
            └─────────────────┘
```

---

## 🔍 详细集成位置

### 1. GaussianModel 类的 `__init__` 方法

**文件位置**：`r2_gaussian/gaussian/gaussian_model.py:66-102`

```python
class GaussianModel:
    def __init__(self, scale_bound=None, args=None):
        # ===== 原始参数（Baseline 就有的）=====
        self._xyz = torch.empty(0)       # [N, 3] 位置
        self._scaling = torch.empty(0)   # [N, 3] 尺度
        self._rotation = torch.empty(0)  # [N, 4] 旋转
        self._density = torch.empty(0)   # [N, 1] 密度（原始值）

        # ===== 🆕 新增：K-Planes 模块（可选）=====
        self.enable_kplanes = getattr(args, 'enable_kplanes', False)

        if self.enable_kplanes:
            from r2_gaussian.gaussian.kplanes import KPlanesEncoder, DensityMLPDecoder

            # 1️⃣ 创建 K-Planes 编码器
            self.kplanes_encoder = KPlanesEncoder(
                grid_resolution=64,      # 平面分辨率 64×64
                feature_dim=32,          # 每个平面 32 维特征
                num_levels=1,
                bounds=(-1.0, 1.0),      # 场景边界 [-1, 1]³
            ).cuda()

            # 2️⃣ 创建 MLP 解码器（96 维 → 1 维）
            self.density_decoder = DensityMLPDecoder(
                input_dim=96,            # 3 个平面 × 32 维 = 96
                hidden_dim=128,          # 隐藏层 128 维
                num_layers=3             # 3 层 MLP
            ).cuda()
        else:
            # 不启用时设为 None（向下兼容）
            self.kplanes_encoder = None
            self.density_decoder = None
```

**关键点**：
- K-Planes 是作为 `GaussianModel` 的**成员变量**添加的
- 使用 `enable_kplanes` 标志控制是否启用（默认 `False`）
- 不影响原始的 `_xyz`, `_scaling`, `_rotation`, `_density` 参数

---

### 2. GaussianModel 的 `get_density` 属性

**文件位置**：`r2_gaussian/gaussian/gaussian_model.py:148-170`

这是 **K-Planes 真正起作用的地方**！

```python
class GaussianModel:
    @property
    def get_density(self):
        """
        获取每个 Gaussian 的密度（渲染时调用）
        """
        # ===== 步骤 1：原始密度激活（Baseline 就有的）=====
        base_density = self.density_activation(self._density)  # softplus(_density)
        # base_density: [N, 1]，范围 [0, +∞)

        # ===== 步骤 2：🆕 K-Planes 调制（仅当启用时）=====
        if self.enable_kplanes and self.kplanes_encoder and self.density_decoder:

            # 2.1 从 K-Planes 提取空间特征
            kplanes_feat = self.get_kplanes_features()
            # 输入：self._xyz [N, 3]
            # 输出：kplanes_feat [N, 96]

            # 2.2 MLP 解码为 density offset
            density_offset = self.density_decoder(kplanes_feat)
            # 输出：density_offset [N, 1]，范围 [-1, 1]（Tanh 约束）

            # 2.3 Sigmoid 映射到 [0.7, 1.3] 调制范围
            modulation = 0.7 + 0.6 * torch.sigmoid(density_offset)
            # modulation: [N, 1]，范围 [0.7, 1.3]

            # 2.4 调制原始密度
            base_density = base_density * modulation
            # 最终密度在 [0.7 × base_density, 1.3 × base_density] 范围内

        return base_density  # [N, 1]
```

**工作流程**：

```
原始密度 _density [N, 1]
         ↓ softplus 激活
base_density [N, 1] (范围 [0, +∞))
         ↓
         ├─ 如果 enable_kplanes=False ────→ 直接返回（Baseline 模式）
         │
         └─ 如果 enable_kplanes=True
                   ↓
            提取位置坐标 _xyz [N, 3]
                   ↓
            K-Planes Encoder：双线性插值提取空间特征
                   ↓
            kplanes_feat [N, 96]
                   ↓
            MLP Decoder：映射到 offset
                   ↓
            density_offset [N, 1] (范围 [-1, 1])
                   ↓
            Sigmoid 映射到调制因子
                   ↓
            modulation [N, 1] (范围 [0.7, 1.3])
                   ↓
            调制密度：base_density × modulation
                   ↓
         final_density [N, 1]
```

---

### 3. `get_kplanes_features` 辅助方法

**文件位置**：`r2_gaussian/gaussian/gaussian_model.py:135-146`

```python
class GaussianModel:
    def get_kplanes_features(self) -> torch.Tensor:
        """
        从 K-Planes 编码器提取当前所有 Gaussians 的特征

        返回:
            features: [N, 96] 拼接后的空间特征
        """
        if self.kplanes_encoder is None:
            raise RuntimeError("K-Planes encoder not initialized!")

        # 调用 K-Planes 编码器前向传播
        return self.kplanes_encoder(self._xyz)
        #      输入：self._xyz [N, 3] 所有 Gaussians 的位置
        #      输出：features [N, 96] 每个 Gaussian 的空间特征
```

---

## 🔗 与渲染流程的集成

### 渲染调用链

```
train.py 或 render.py
    ↓ 调用
render_query.py: render()
    ↓ 访问
gaussians.get_density  ← @property 装饰器自动调用
    ↓ 内部执行
├─ base_density = softplus(self._density)
├─ kplanes_feat = self.kplanes_encoder(self._xyz)  ← K-Planes 编码
├─ density_offset = self.density_decoder(kplanes_feat)  ← MLP 解码
├─ modulation = 0.7 + 0.6 * sigmoid(density_offset)  ← Sigmoid 调制
└─ final_density = base_density * modulation
    ↓ 传递给
CUDA 渲染器
    ↓ 输出
2D 投影图像
```

**关键设计**：
- 使用 `@property` 装饰器使 K-Planes 集成对外部透明
- `render_query.py` **无需修改**任何代码
- 渲染器调用 `gaussians.get_density` 时自动应用 K-Planes 调制

---

## 🆚 参数量对比

### Baseline（原始 R²-Gaussian）

| 参数 | 形状 | 数量（N=50K Gaussians） |
|------|------|------------------------|
| `_xyz` | [N, 3] | 150,000 |
| `_scaling` | [N, 3] | 150,000 |
| `_rotation` | [N, 4] | 200,000 |
| `_density` | [N, 1] | 50,000 |
| **总计** | - | **550,000** |

### X²-Gaussian（加入 K-Planes）

| 参数 | 形状 | 数量 |
|------|------|------|
| **原始 Gaussians** | - | **550,000** |
| `kplanes_encoder.plane_xy` | [1, 32, 64, 64] | 131,072 |
| `kplanes_encoder.plane_xz` | [1, 32, 64, 64] | 131,072 |
| `kplanes_encoder.plane_yz` | [1, 32, 64, 64] | 131,072 |
| `density_decoder` (3 层 MLP) | - | 16,769 |
| **K-Planes 总计** | - | **409,985** |
| **总计** | - | **959,985** |

**增加量**：+409,985 参数（+74.5%）

---

## 💡 设计优势

### 1. 向下兼容

```python
# Baseline 模式（不启用 K-Planes）
python train.py --source_path data/foot.pickle
# → enable_kplanes=False，退化为标准 3D-GS

# X²-Gaussian 模式（启用 K-Planes）
python train.py --source_path data/foot.pickle --enable_kplanes
# → enable_kplanes=True，使用 K-Planes 增强
```

### 2. 透明集成

```python
# 渲染代码（render_query.py）无需修改
density = gaussians.get_density  # ← 自动应用 K-Planes（如果启用）
```

### 3. 独立优化

K-Planes 参数有独立的学习率：

```python
# train.py 优化器配置
optimizer = torch.optim.Adam([
    {'params': gaussians._xyz, 'lr': 0.0002, 'name': 'xyz'},
    {'params': gaussians._density, 'lr': 0.01, 'name': 'density'},
    # ... 其他原始参数

    # 🆕 K-Planes 独立学习率
    {'params': gaussians.kplanes_encoder.parameters(), 'lr': 0.002, 'name': 'kplanes'},
    {'params': gaussians.density_decoder.parameters(), 'lr': 0.001, 'name': 'kplanes_decoder'},
])
```

---

## 📐 空间关系示意图

### K-Planes 如何"看到"Gaussians 的位置

```
3D 场景空间 [-1, 1]³
┌─────────────────────────────────────┐
│                                     │
│    ● Gaussian 1: xyz = [0.2, 0.5, -0.3]
│                       ↓
│           ┌───────────────────────┐
│           │  K-Planes 编码器      │
│           │  ├─ XY 平面: (0.2, 0.5) → feat_xy [32]
│           │  ├─ XZ 平面: (0.2, -0.3) → feat_xz [32]
│           │  └─ YZ 平面: (0.5, -0.3) → feat_yz [32]
│           └───────────────────────┘
│                       ↓
│           拼接: [feat_xy, feat_xz, feat_yz] → [96]
│                       ↓
│           ┌───────────────────────┐
│           │  MLP Decoder          │
│           │  96 → 128 → 128 → 1   │
│           └───────────────────────┘
│                       ↓
│           density_offset = -0.35 (例子)
│                       ↓
│           modulation = 0.7 + 0.6 * sigmoid(-0.35) = 0.93
│                       ↓
│    ● Gaussian 1: final_density = base_density × 0.93
│
└─────────────────────────────────────┘
```

**关键点**：
- 每个 Gaussian 的位置 `xyz` 会被投影到 3 个正交平面
- 从 3 个平面分别提取特征，然后拼接
- MLP 将拼接特征映射为该 Gaussian 的**个性化调制因子**

---

## 🎯 总结

### K-Planes 加在哪里？

**直接回答**：
1. **物理位置**：集成在 `GaussianModel` 类内部（`gaussian_model.py`）
2. **作用对象**：增强每个 3D Gaussian 的 `density` 属性
3. **集成方式**：通过 `@property` 在 `get_density` 方法中透明调制

### 为什么这样设计？

| 优点 | 说明 |
|------|------|
| ✅ **无侵入性** | 不修改渲染器代码（`render_query.py`） |
| ✅ **向下兼容** | `enable_kplanes=False` 退化为 baseline |
| ✅ **灵活扩展** | 未来可以加更多特征（scale、rotation） |
| ✅ **独立优化** | K-Planes 有自己的学习率和正则化 |

### 与 NeRF/InstantNGP 的区别

| 方法 | 空间表示 | K-Planes 作用 |
|------|---------|--------------|
| **NeRF** | 隐式 MLP | 直接输出 density + color |
| **InstantNGP** | 多分辨率哈希表 | 提供特征给 MLP |
| **3D-GS (Baseline)** | 显式 Gaussians | ❌ 无 K-Planes |
| **3D-GS + K-Planes (Ours)** | 显式 Gaussians | ✅ K-Planes 调制 density |

**核心思想**：K-Planes 提供**空间几何先验**，帮助 Gaussians 更好地学习密度分布，尤其在稀疏视角（3 views）场景下。

---

**相关文件**：
- K-Planes 编码器实现：`r2_gaussian/gaussian/kplanes.py`
- 集成代码：`r2_gaussian/gaussian/gaussian_model.py`
- TV 正则化：`r2_gaussian/utils/regulation.py`
- 训练脚本：`scripts/train_foot3_x2_v3_ultrathink.sh`
