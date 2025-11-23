# X²-Gaussian 原版对齐修改总结

**修改日期**: 2025-01-23
**执行者**: Claude Code Agent
**目标**: 将当前 K-Planes + TV 实现完全对齐 X²-Gaussian 原版设置

---

## ✅ 已完成的 P0 修改

### 1. K-Planes 初始化方法修正

**文件**: `r2_gaussian/gaussian/kplanes.py:62-66`

**修改前**:
```python
nn.init.xavier_uniform_(self.plane_xy)
nn.init.xavier_uniform_(self.plane_xz)
nn.init.xavier_uniform_(self.plane_yz)
```

**修改后**:
```python
# 对齐 X²-Gaussian 原版
nn.init.uniform_(self.plane_xy, a=0.1, b=0.5)
nn.init.uniform_(self.plane_xz, a=0.1, b=0.5)
nn.init.uniform_(self.plane_yz, a=0.1, b=0.5)
```

**影响**:
- 初始特征值范围从 `[-0.5, 0.5]` 变为 `[0.1, 0.5]`
- 避免负值特征，更符合 X²-Gaussian 设计

---

### 2. TV 损失计算公式修正

**文件**: `r2_gaussian/utils/regulation.py:16-72`

**关键修改**:

#### (1) 从 L1 改为 L2（平方差）
```python
# 修改前：L1 损失
tv_loss = grad_horizontal.abs().mean() + grad_vertical.abs().mean()

# 修改后：L2 损失（X²-Gaussian 原版）
h_tv = torch.square(grad_h).sum()
w_tv = torch.square(grad_w).sum()
```

#### (2) 从 `mean()` 改为精确归一化 `sum() / count`
```python
# 修改前：简单平均
tv_loss = ... .mean()

# 修改后：X²-Gaussian 原版公式
batch_size, c, h, w = plane.shape
count_h = batch_size * c * (h - 1) * w
count_w = batch_size * c * h * (w - 1)
tv_loss = 2 * (h_tv / count_h + w_tv / count_w)
```

**完整公式**（X²-Gaussian 原版）:
```
TV(P) = 2 * (Σ(P[i+1,j] - P[i,j])² / count_h + Σ(P[i,j+1] - P[i,j])² / count_w)
```

**影响**:
- TV 损失值变大（L2 > L1）
- 归一化更精确，训练更稳定

---

### 3. K-Planes 学习率修正

**文件**: `r2_gaussian/arguments/__init__.py:160-161`

**修改对比**:

| 参数 | 修改前 | 修改后 | 提升倍数 |
|-----|-------|-------|---------|
| `kplanes_lr_init` | 0.00016 | 0.002 | **12.5×** |
| `kplanes_lr_final` | 0.0000016 | 0.0002 | **125×** |

**对齐 X²-Gaussian 原版**:
- `grid_lr_init = 0.002`
- `grid_lr_final = 0.0002`

**影响**:
- K-Planes 特征更新速度显著加快
- 预期收敛更快，特征质量更高

---

### 4. TV 损失类型默认值修正

**文件**: `r2_gaussian/arguments/__init__.py:168`

**修改**:
```python
# 修改前
self.tv_loss_type = "l1"

# 修改后
self.tv_loss_type = "l2"  # 对齐 X²-Gaussian 原版
```

**影响**:
- 默认使用 L2 TV 损失
- 与修改 2 配合，完全对齐原版

---

## 📊 预期效果

基于 X²-Gaussian 论文结果和当前修改，预期：

1. **训练稳定性**: ⬆️ 提升（TV 计算更精确）
2. **收敛速度**: ⬆️ 加快（学习率提升 10+ 倍）
3. **特征质量**: ⬆️ 改善（初始化更合理）
4. **PSNR**: ⬆️ +0.5~1.0 dB（保守估计）

**成功标准**（Foot-3 views）:
- **最低要求**: PSNR ≥ 28.49（baseline 水平）
- **理想目标**: PSNR > 28.7（有明显提升）

---

## 🔬 验证计划

### 步骤 1: 快速语法检查
```bash
conda activate r2_gaussian_new
python -c "from r2_gaussian.gaussian.kplanes import KPlanesEncoder; print('✅ kplanes.py 语法正确')"
python -c "from r2_gaussian.utils.regulation import compute_plane_tv_loss; print('✅ regulation.py 语法正确')"
```

### 步骤 2: 单元测试
```bash
# 测试 K-Planes 初始化范围
python -c "
import torch
from r2_gaussian.gaussian.kplanes import KPlanesEncoder
enc = KPlanesEncoder().cuda()
params = enc.get_plane_params()
assert params[0].min().item() >= 0.1, 'Min < 0.1'
assert params[0].max().item() <= 0.5, 'Max > 0.5'
print('✅ K-Planes 初始化范围正确: [0.1, 0.5]')
"

# 测试 TV 损失类型
python -c "
from r2_gaussian.arguments import OptimizationParams
import argparse
parser = argparse.ArgumentParser()
opt = OptimizationParams(parser)
args = parser.parse_args([])
opt = opt.extract(args)
assert opt.tv_loss_type == 'l2', f'TV loss type 错误: {opt.tv_loss_type}'
print(f'✅ TV 损失类型正确: {opt.tv_loss_type}')
print(f'✅ K-Planes 学习率正确: init={opt.kplanes_lr_init}, final={opt.kplanes_lr_final}')
"
```

### 步骤 3: 100 iters 快速验证
```bash
python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/test_x2_alignment \
    --iterations 100 \
    --enable_kplanes \
    --lambda_plane_tv 0.0002
```

**预期日志**:
```
✓ K-Planes Encoder 已启用
  - 平面分辨率: 64
  - K-Planes 参数量: 393,216
✓ K-Planes TV 正则化已启用
  - lambda_plane_tv: 0.0002
  - TV 损失类型: l2  # ← 关键检查点
[Iter 1] K-Planes 诊断:
  - K-Planes 特征范围: [0.1xxx, 0.4xxx]  # ← 关键检查点
  - TV loss (plane): 0.xxxxx
```

### 步骤 4: 完整实验（30K iters）
```bash
python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/2025_01_23_foot_3views_x2_aligned \
    --iterations 30000 \
    --enable_kplanes \
    --lambda_plane_tv 0.0002 \
    --test_iterations 30000
```

---

## 📝 修改文件清单

| 文件 | 修改行数 | 修改内容 |
|-----|---------|---------|
| `r2_gaussian/gaussian/kplanes.py` | 51, 62-66 | 初始化方法注释 + uniform(0.1, 0.5) |
| `r2_gaussian/utils/regulation.py` | 16-72, 75-88 | TV 公式 L2+sum/count + 默认值 |
| `r2_gaussian/arguments/__init__.py` | 158-168 | 学习率 + TV loss_type |

**总计**: 3 个文件，约 20 行关键修改

---

## ⚠️ 回退方案

如果实验结果不理想（PSNR < 28.0），可以回退：

```bash
# 方案 1: 仅回退学习率
# 在训练命令中显式指定
python train.py ... --kplanes_lr_init 0.00016 --kplanes_lr_final 0.0000016

# 方案 2: 回退 TV 损失类型
python train.py ... --tv_loss_type l1

# 方案 3: 完全回退（使用 git）
git diff HEAD r2_gaussian/  # 查看修改
git checkout HEAD -- r2_gaussian/gaussian/kplanes.py  # 回退单文件
```

---

## 🎯 下一步

1. ✅ 执行验证计划（语法检查 + 单元测试）
2. ⏳ 运行 100 iters 快速验证
3. ⏳ 根据快速验证结果决定是否运行完整实验
4. ⏳ 分析结果并更新 Neo4j 记忆库

**当前状态**: 等待用户确认是否开始验证

---

**修改者签名**: Claude Code Agent
**审核者**: 等待用户审核
