# X²-Gaussian 创新点迁移实施计划（详细版）

**创建时间**：2025-01-18
**目标**：将 X²-Gaussian 的 K-Planes 和 TV 正则化迁移到 R²-Gaussian
**验证数据集**：Foot 3 views（目标：PSNR > 28.49, SSIM > 0.9005）

---

## 📐 阶段一：K-Planes + TV 正则化

### 1. 新增模块：K-Planes 实现

**文件**：`r2_gaussian/gaussian/kplanes.py`

**类定义**：`KPlanesEncoder`

```python
class KPlanesEncoder(nn.Module):
    """
    K-Planes 空间分解编码器（仅空间维度，无时间维度）

    将 3D 空间 (x,y,z) 分解为 3 个正交平面特征网格：
    - plane_xy: 特征平面 [N_xy, resolution, resolution, feature_dim]
    - plane_xz: 特征平面 [N_xz, resolution, resolution, feature_dim]
    - plane_yz: 特征平面 [N_yz, resolution, resolution, feature_dim]

    其中 N_xy/N_xz/N_yz 是多分辨率层数（默认 1）
    """

    def __init__(
        self,
        grid_resolution: int = 64,       # 单平面分辨率
        feature_dim: int = 32,           # 特征维度
        num_levels: int = 1,             # 多分辨率层数
        bounds: tuple = (-1.0, 1.0),     # 空间边界（R²-GS 归一化到 [-1,1]³）
    ):
        """初始化 3 个空间平面"""
        pass

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        输入：xyz [N, 3] - 高斯中心坐标（世界坐标系）
        输出：features [N, feature_dim * 3] - 3 个平面的特征拼接

        实现步骤：
        1. 将 xyz 归一化到 [-1, 1]（用于 grid_sample）
        2. 对每个平面执行双线性插值：
           - plane_xy: 从 (x, y) 提取特征
           - plane_xz: 从 (x, z) 提取特征
           - plane_yz: 从 (y, z) 提取特征
        3. 拼接 3 个平面的特征 -> [N, 32*3=96]
        """
        pass

    def get_plane_params(self) -> List[nn.Parameter]:
        """返回所有平面参数（用于优化器配置和 TV 正则化）"""
        return [self.plane_xy, self.plane_xz, self.plane_yz]
```

**关键实现细节**：
- 使用 `nn.Parameter` 存储平面特征，初始化为小的随机值（如 Xavier 初始化）
- `grid_sample` 使用 `align_corners=True` 确保边界对齐
- 边界裁剪：超出 [-1, 1] 的坐标需要裁剪

---

### 2. 新增模块：TV 正则化

**文件**：`r2_gaussian/utils/regulation.py`

**函数定义**：`compute_plane_tv()`

```python
def compute_plane_tv(
    plane: torch.Tensor,          # [1, resolution, resolution, feature_dim]
    loss_type: str = "l1",        # "l1" 或 "l2"
) -> torch.Tensor:
    """
    计算单个平面的 Total Variation 损失

    公式：
    TV(P) = Σ |P[i+1,j] - P[i,j]| + |P[i,j+1] - P[i,j]|

    实现：
    1. 计算水平梯度：plane[:, 1:, :] - plane[:, :-1, :]
    2. 计算垂直梯度：plane[:, :, 1:] - plane[:, :, :-1]
    3. 对梯度求 L1/L2 范数并求和
    """
    pass

def compute_plane_tv_loss(
    planes: List[torch.Tensor],   # [plane_xy, plane_xz, plane_yz]
    weights: List[float],          # [w_xy, w_xz, w_yz]
    loss_type: str = "l1",
) -> torch.Tensor:
    """
    计算所有平面的加权 TV 损失

    返回：weighted_tv_loss = Σ weights[i] * TV(planes[i])
    """
    total_loss = 0.0
    for plane, weight in zip(planes, weights):
        total_loss += weight * compute_plane_tv(plane, loss_type)
    return total_loss
```

---

### 3. 修改：GaussianModel 集成 K-Planes

**文件**：`r2_gaussian/gaussian/gaussian_model.py`

**修改位置**：`GaussianModel.__init__()` (行 65-76)

```python
class GaussianModel:
    def __init__(self, sh_degree: int, args):
        # ... 原有初始化代码 ...

        # 新增：K-Planes 编码器（可选）
        self.enable_kplanes = args.enable_kplanes if hasattr(args, 'enable_kplanes') else False
        if self.enable_kplanes:
            self.kplanes_encoder = KPlanesEncoder(
                grid_resolution=args.kplanes_resolution,
                feature_dim=args.kplanes_dim,
                num_levels=1,
                bounds=(-1.0, 1.0),
            ).cuda()
        else:
            self.kplanes_encoder = None
```

**新增方法**：`get_kplanes_features()`

```python
def get_kplanes_features(self, xyz: torch.Tensor) -> torch.Tensor:
    """
    获取指定位置的 K-Planes 特征

    输入：xyz [N, 3] - 高斯中心坐标
    输出：features [N, 96] - 如果启用 K-Planes
           或 None - 如果未启用
    """
    if not self.enable_kplanes or self.kplanes_encoder is None:
        return None
    return self.kplanes_encoder(xyz)
```

**修改方法**：`training_setup()` (行 187-226)

```python
def training_setup(self, training_args):
    # ... 原有参数组 ...

    # 新增：K-Planes 参数组
    if self.enable_kplanes and self.kplanes_encoder is not None:
        l.append({
            "params": self.kplanes_encoder.parameters(),
            "lr": training_args.kplanes_lr_init,
            "name": "kplanes"
        })

    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
```

---

### 4. 修改：参数配置系统

**文件**：`r2_gaussian/arguments/__init__.py`

**ModelParams 新增参数** (行 20-104)：

```python
class ModelParams(UserScalarType):
    def __init__(self, parser, sentinel=False):
        # ... 原有参数 ...

        # K-Planes 参数
        self.enable_kplanes = False
        self.kplanes_resolution = 64
        self.kplanes_dim = 32

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        # ... 原有提取逻辑 ...

        # 提取 K-Planes 参数
        g = super().extract(args)
        self.enable_kplanes = args.enable_kplanes
        self.kplanes_resolution = args.kplanes_resolution
        self.kplanes_dim = args.kplanes_dim
        return g
```

**OptimizationParams 新增参数** (行 114-152)：

```python
class OptimizationParams(UserScalarType):
    def __init__(self, parser, sentinel=False):
        # ... 原有参数 ...

        # K-Planes 学习率
        self.kplanes_lr_init = 0.00016
        self.kplanes_lr_final = 0.0000016

        # TV 正则化参数
        self.lambda_plane_tv = 0.0
        self.plane_tv_weight_proposal = [0.0001, 0.0001, 0.0001]  # [xy, xz, yz]
        self.tv_loss_type = "l1"

        super().__init__(parser, "Optimization Parameters", sentinel)
```

**在 train.py 中注册参数**：

```python
# train.py, 参数注册部分（顶部）
parser.add_argument("--enable_kplanes", action="store_true",
                    help="启用 K-Planes 空间分解")
parser.add_argument("--kplanes_resolution", type=int, default=64,
                    help="K-Planes 平面分辨率")
parser.add_argument("--kplanes_dim", type=int, default=32,
                    help="K-Planes 特征维度")
parser.add_argument("--kplanes_lr_init", type=float, default=0.00016,
                    help="K-Planes 初始学习率")
parser.add_argument("--kplanes_lr_final", type=float, default=0.0000016,
                    help="K-Planes 最终学习率")
parser.add_argument("--lambda_plane_tv", type=float, default=0.0,
                    help="TV 正则化权重（0 表示不启用）")
parser.add_argument("--plane_tv_weight_proposal", nargs='+', type=float,
                    default=[0.0001, 0.0001, 0.0001],
                    help="每个平面的 TV 权重 [xy, xz, yz]")
parser.add_argument("--tv_loss_type", type=str, default="l1", choices=["l1", "l2"],
                    help="TV 损失类型")
```

---

### 5. 修改：训练循环集成新损失

**文件**：`train.py`

**修改位置 1**：导入新模块（顶部）

```python
from r2_gaussian.gaussian.kplanes import KPlanesEncoder
from r2_gaussian.utils.regulation import compute_plane_tv_loss
```

**修改位置 2**：训练循环中添加 TV 损失 (行 121-138 之后)

```python
# train.py, 行 ~140（在现有 TV 损失之后）
if opt.lambda_plane_tv > 0 and gaussians.enable_kplanes:
    # 获取 K-Planes 参数
    planes = gaussians.kplanes_encoder.get_plane_params()

    # 计算 TV 损失
    tv_loss_planes = compute_plane_tv_loss(
        planes=planes,
        weights=opt.plane_tv_weight_proposal,
        loss_type=opt.tv_loss_type,
    )

    loss["plane_tv"] = tv_loss_planes
    loss["total"] = loss["total"] + opt.lambda_plane_tv * tv_loss_planes
```

**修改位置 3**：TensorBoard 日志记录 (行 173-213)

```python
# 记录 K-Planes TV 损失
if "plane_tv" in loss:
    tb_writer.add_scalar("train_loss_patches/plane_tv",
                         loss["plane_tv"].item(), iteration)
```

---

## 🧪 测试与验证

### 向下兼容性测试

**测试 1**：不启用 K-Planes（默认行为）
```bash
python train.py -s data/foot -m output/test_baseline
# 预期：正常训练，无 K-Planes 相关日志
```

**测试 2**：启用 K-Planes（无 TV）
```bash
python train.py -s data/foot -m output/test_kplanes \
  --enable_kplanes --kplanes_resolution 64
# 预期：创建 K-Planes 编码器，但 TV 损失为 0
```

**测试 3**：启用 K-Planes + TV
```bash
python train.py -s data/foot -m output/test_kplanes_tv \
  --enable_kplanes --kplanes_resolution 64 \
  --lambda_plane_tv 0.0002
# 预期：K-Planes + TV 损失都生效
```

### 功能测试

**测试点 1**：K-Planes 特征提取
```python
# 测试脚本
xyz = torch.randn(1000, 3).cuda() * 0.5  # [-0.5, 0.5] 范围
features = gaussians.get_kplanes_features(xyz)
assert features.shape == (1000, 96), "特征维度错误"
```

**测试点 2**：TV 损失计算
```python
# 测试脚本
planes = gaussians.kplanes_encoder.get_plane_params()
tv_loss = compute_plane_tv_loss(planes, [1.0, 1.0, 1.0], "l1")
assert tv_loss.item() > 0, "TV 损失应为正数"
```

---

## 📊 实验验证计划

### 数据集：Foot 3 views
- **Baseline**：PSNR=28.4873, SSIM=0.9005
- **目标**：PSNR > 28.49, SSIM > 0.9005

### 消融实验

| 实验 ID | 配置 | 命令 | 预期提升 |
|--------|------|------|---------|
| EXP-1 | Baseline | `--test_iterations 30000` | 0 dB（对照） |
| EXP-2 | +K-Planes | `--enable_kplanes` | +0.3~0.5 dB |
| EXP-3 | +K-Planes+TV | `--enable_kplanes --lambda_plane_tv 0.0002` | +0.5~1.0 dB |

### 评估指标
- **2D 渲染**：PSNR, SSIM（在测试视角）
- **3D 体积**：体积重建的 PSNR, SSIM（如果可用）
- **训练时间**：对比训练速度

### 超参数搜索（如果 EXP-3 效果不佳）
- `kplanes_resolution`: [32, 64, 128]
- `lambda_plane_tv`: [0.0001, 0.0002, 0.0005]
- `plane_tv_weight_proposal`: 尝试不同平面权重比例

---

## ✅ 检查点

- ✋ **检查点 1**：代码实现完成 → 用户审核修改范围
- ✋ **检查点 2**：EXP-2/3 初步结果 → 决定是否继续优化
- ✋ **检查点 3**：消融实验完成 → 决定是否进入阶段二（多头解码器）

---

## 📝 交付物

1. **代码模块**：
   - `r2_gaussian/gaussian/kplanes.py`
   - `r2_gaussian/utils/regulation.py`
   - 修改后的 `gaussian_model.py`, `arguments/__init__.py`, `train.py`

2. **文档**：
   - `cc-agent/code/code_review.md`（代码审核文档）
   - `cc-agent/experiments/experiment_plan.md`（实验计划）
   - `cc-agent/experiments/result_analysis.md`（结果分析）

3. **实验数据**：
   - TensorBoard 日志
   - 训练曲线对比图
   - 定量指标表格

---

## 🚀 当前状态

**等待编程专家开始实现代码模块**
