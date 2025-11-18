# SSS Bug 修复方案（基于官方代码研究）

**研究日期：** 2025-11-18
**官方仓库：** https://github.com/realcrane/3D-student-splatting-and-scooping
**目标：** 修复用户 SSS 实现中的 5 个致命 bug，恢复性能从 -8.39 dB 回到 Baseline 水平或更好

---

## 第一部分：官方代码研究结果

### 1. 训练参数设置

从官方 `arguments/__init__.py` 和配置文件 `configs/bicycle.json` 提取的关键参数：

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `nu_degree` | 10 (默认) / 100 (bicycle) | Student-t 分布自由度初始值 |
| `opacity_threshold` | 0.005 | Opacity 过滤阈值（用于组件回收） |
| `opacity_lr` | 0.005 | Opacity 学习率 |
| `opacity_reset_interval` | 3000 | Opacity 重置间隔 |
| `opacity_reg` | 0.01 | **Balance Loss 权重** |
| `C_burnin` | 5e8 (bicycle) / 5e5 (默认) | SGHMC burn-in 阶段噪声系数 |
| `C` | 1.2e2 (bicycle) / 1e2 (默认) | SGHMC 主训练阶段噪声系数 |
| `burnin_iterations` | 15000 (bicycle) / 7000 (默认) | Burn-in 迭代次数 |
| `cap_max` | 3000000 | 最大组件数 |

**关键发现：**
- ✅ 官方**没有** `use_student_t` 参数，Student-t 是**默认启用**的
- ✅ Balance Loss 是简单的 **`opacity_reg * torch.abs(opacity).mean()`**（L1 正则）
- ✅ 不存在"渐进式 Scooping 限制"或"正负值比例控制"

### 2. Opacity 激活函数

从官方 `scene/nt_model.py` 的实现：

```python
# 官方实现
self.opacity_activation = torch.tanh
self.inverse_opacity_activation = inverse_tanh

# 初始化
opacities = self.inverse_opacity_activation(0.5 * torch.ones(...))

# Clamping (在 get_opacity 中)
opacity = torch.clamp(opacity, -1 + 1e-5, 1 - 1e-5)
```

**关键特性：**
- ✅ 使用 **`tanh`** 激活函数，值域严格 **[-1, 1]**
- ✅ 初始化为 **0.5**（经过 `inverse_tanh` 映射到参数空间）
- ✅ Clamp 范围：**[-1 + ε, 1 - ε]**（避免数值不稳定）
- ⚠️ **不是用户实现的 `[-0.2, 1.0]` 范围！**

### 3. 组件回收机制（Component Recycling）

从官方 `scene/nt_model.py` 的 `recycle_components` 方法：

```python
def recycle_components(self):
    # 1. 识别 dead components
    opacity = self.get_opacity
    alive_mask = opacity > self.opacity_threshold  # 0.005
    dead_mask = ~alive_mask

    # 2. 限制回收数量（5% cap）
    max_recycle = int(0.05 * opacity.shape[0])
    dead_indices = torch.where(dead_mask)[0]
    if len(dead_indices) > max_recycle:
        dead_indices = dead_indices[:max_recycle]

    # 3. 从存活组件中重新采样（基于 opacity）
    alive_indices = torch.where(alive_mask)[0]
    sample_weights = opacity[alive_mask].squeeze()
    sample_indices = torch.multinomial(sample_weights, len(dead_indices), replacement=True)
    source_indices = alive_indices[sample_indices]

    # 4. 重新初始化 dead components
    self._xyz[dead_indices] = self._xyz[source_indices] + torch.randn_like(...) * 0.01
    self._opacity[dead_indices] = self.inverse_opacity_activation(torch.ones_like(...) * 0.5)
    self._nu[dead_indices] = self._nu[source_indices].clone()
    # ... 复制其他参数 ...

    # 5. 重置优化器动量
    self.optimizer.reset_state(dead_indices)
```

**关键逻辑：**
- ✅ **每次最多回收 5% 总组件数**
- ✅ 低 opacity 阈值：**0.005**
- ✅ 零 opacity 重新初始化到 **0.5**
- ✅ 从高 opacity 组件重新采样
- ⚠️ **完全替代传统 densification**，不是并存

### 4. 渲染逻辑

从官方 `t_renderer/__init__.py`：

```python
def render(viewpoint_camera, pc, pipe, bg_color, ...):
    # 获取参数
    opacity = pc.get_opacity  # 已经过 tanh 激活
    nu_degree = pc.get_nu_degree
    negative_value = pc.get_negative

    # 传递给 CUDA 光栅化器
    rendered_image, radii = rasterizer(
        means3D=xyz,
        opacity=opacity,  # [-1, 1] 范围
        nu_degree=nu_degree,
        negative_value=negative_value,
        ...
    )
```

**关键发现：**
- ✅ Opacity 直接传递给光栅化器，**不在渲染函数中 clamp**
- ✅ **Clamp 在模型的 `get_opacity` 中完成**
- ✅ 引入 `negative_value` 参数支持负密度

### 5. Balance Loss（官方实现）

从官方 `train.py` 和 `arguments/__init__.py`：

```python
# 官方 Balance Loss（L1 正则）
opacity = primitives.get_opacity
balance_loss = args.opacity_reg * torch.abs(opacity).mean()

loss = L1_loss + ssim_loss + balance_loss
```

**关键公式：**
- ✅ **Balance Loss = `λ * Σ|o_i|_1`**（论文公式）
- ✅ 默认权重：**λ = 0.01**
- ❌ **不是用户实现的复杂"正负值比例控制"**

---

## 第二部分：Bug 修复详细方案

### Bug 1: 启用 SSS

**问题描述：**
- 文件：`train.py`
- 行号：142
- 当前代码：`use_student_t = False  # 强制禁用 SSS`

**修复方案：**

```python
# 修改前
use_student_t = False  # 强制禁用 SSS

# 修改后
use_student_t = args.enable_sss  # 允许通过命令行参数启用 SSS
```

**影响：**
- 启用后将使用 Student-t 分布和 SGHMC 优化器
- 需确保 `--enable_sss` 命令行参数正确传递

---

### Bug 2: Opacity 激活函数错误

**问题描述：**
- 文件：`r2_gaussian/gaussian/gaussian_model.py`
- 行号：72-78
- 当前代码：`opacity_activation = lambda x: torch.sigmoid(x) * 1.2 - 0.2  # [-0.2, 1.0]`
- **错误：** 值域 `[-0.2, 1.0]` 与论文的 `[-1, 1]` 不符

**修复方案：**

```python
# 修改前（行 72-78）
self.opacity_activation = lambda x: torch.sigmoid(x) * 1.2 - 0.2  # [-0.2, 1.0]
self.opacity_inverse_activation = lambda x: inverse_sigmoid(
    (torch.clamp(x, -0.19, 0.99) + 0.2) / 1.2
)

# 修改后
self.opacity_activation = torch.tanh  # [-1, 1] 范围（官方实现）
self.opacity_inverse_activation = lambda x: 0.5 * torch.log(
    (1 + torch.clamp(x, -0.99, 0.99)) / (1 - torch.clamp(x, -0.99, 0.99))
)  # inverse_tanh with numerical stability
```

**同时修改初始化逻辑（行 291-295）：**

```python
# 修改前
opacity_vals = torch.sigmoid(fused_density.clone()) * 0.9  # [0, 0.9]
opacity_init = self.opacity_inverse_activation(opacity_vals)

# 修改后
opacity_vals = torch.ones_like(fused_density) * 0.5  # 初始化为 0.5（官方策略）
opacity_init = self.opacity_inverse_activation(opacity_vals)
```

**添加 `get_opacity` 中的 Clamp（如果还没有）：**

```python
@property
def get_opacity(self):
    if self.use_student_t:
        opacity = self.opacity_activation(self._opacity)
        # 官方 clamp 逻辑
        return torch.clamp(opacity, -1.0 + 1e-5, 1.0 - 1e-5)
    else:
        return self.opacity_activation(self._opacity)
```

---

### Bug 3: 移除渐进式 Scooping 限制

**问题描述：**
- 文件：`train.py`
- 行号：792-843
- **自创逻辑：** 复杂的正负值比例控制（`balance_loss`），论文中不存在

**修复方案：**

将行 792-843 的整个自创 Balance Loss 逻辑替换为官方的简单 L1 正则：

```python
# 完全删除行 792-843 的代码
# 删除：
#   - negative_penalty
#   - positive_encouragement
#   - balance_loss 复杂公式
#   - nu_diversity_loss
#   - 所有相关的 debug logging

# 替换为官方实现（插入到行 792 位置）
if hasattr(GsDict[f"gs{i}"], 'use_student_t') and GsDict[f"gs{i}"].use_student_t:
    opacity = GsDict[f"gs{i}"].get_opacity
    # 官方 Balance Loss: L1 正则化
    opacity_reg_weight = 0.01  # 官方默认权重
    balance_loss = opacity_reg_weight * torch.abs(opacity).mean()
    LossDict[f"loss_gs{i}"] += balance_loss

    # 简化的日志（每 2000 次迭代）
    if iteration % 2000 == 0:
        pos_ratio = (opacity > 0).float().mean()
        neg_ratio = (opacity < 0).float().mean()
        print(f"🎯 [SSS-Official] Iter {iteration}: "
              f"Opacity range [{opacity.min():.3f}, {opacity.max():.3f}], "
              f"Balance: {pos_ratio*100:.1f}% pos / {neg_ratio*100:.1f}% neg, "
              f"Balance Loss: {balance_loss.item():.6f}")
```

---

### Bug 4: Balance Loss 公式错误

**问题：** 已在 Bug 3 中修复。

**总结：**
- ❌ 用户自创：`negative_penalty + positive_encouragement`
- ✅ 官方实现：**`0.01 * torch.abs(opacity).mean()`**

---

### Bug 5: 组件回收机制缺失

**问题描述：**
- 文件：`train.py` 和 `gaussian_model.py`
- 当前代码：使用传统 `densify_and_prune`
- **缺失：** 论文核心的组件回收（Component Recycling）机制

**修复方案（分 2 步）：**

#### 步骤 1：在 `gaussian_model.py` 中添加 `recycle_components` 方法

在 `GaussianModel` 类中添加以下方法（建议插入到 `densify_and_prune` 方法之后）：

```python
def recycle_components(self, opacity_threshold=0.005, max_recycle_ratio=0.05):
    """
    组件回收机制（官方实现）

    参数：
        opacity_threshold: 低 opacity 阈值，低于此值视为 dead component
        max_recycle_ratio: 每次最多回收的组件比例（默认 5%）
    """
    if not self.use_student_t:
        return  # 仅 SSS 启用

    with torch.no_grad():
        # 1. 识别 dead components
        opacity = self.get_opacity
        alive_mask = torch.abs(opacity) > opacity_threshold  # 使用绝对值
        dead_mask = ~alive_mask

        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return

        # 2. 限制回收数量（5% cap）
        max_recycle = int(max_recycle_ratio * opacity.shape[0])
        dead_indices = torch.where(dead_mask)[0]
        if len(dead_indices) > max_recycle:
            # 随机选择要回收的组件
            perm = torch.randperm(len(dead_indices), device=dead_indices.device)
            dead_indices = dead_indices[perm[:max_recycle]]

        num_to_recycle = len(dead_indices)

        # 3. 从存活组件中重新采样（基于 opacity 权重）
        alive_indices = torch.where(alive_mask)[0]
        if len(alive_indices) == 0:
            print("⚠️ [SSS-Recycle] No alive components, skipping recycling")
            return

        # 使用 opacity 绝对值作为采样权重
        sample_weights = torch.abs(opacity[alive_mask].squeeze())
        sample_weights = sample_weights / sample_weights.sum()  # 归一化

        # 重新采样源组件
        sample_indices = torch.multinomial(sample_weights, num_to_recycle, replacement=True)
        source_indices = alive_indices[sample_indices]

        # 4. 重新初始化 dead components
        # Position: 添加小噪声
        self._xyz[dead_indices] = self._xyz[source_indices].clone() + torch.randn_like(self._xyz[dead_indices]) * 0.01

        # Opacity: 重置为 0.5（官方策略）
        opacity_init_val = 0.5 * torch.ones(num_to_recycle, 1, device="cuda")
        self._opacity[dead_indices] = self.opacity_inverse_activation(opacity_init_val)

        # Nu: 继承源组件
        self._nu[dead_indices] = self._nu[source_indices].clone()

        # Scaling: 继承源组件
        self._scaling[dead_indices] = self._scaling[source_indices].clone()

        # Rotation: 继承源组件
        self._rotation[dead_indices] = self._rotation[source_indices].clone()

        # Density: 继承源组件
        self._density[dead_indices] = self._density[source_indices].clone()

        # 5. 重置优化器状态（重要！）
        # 清除回收组件的梯度和动量
        for param_group in self.optimizer.param_groups:
            for param_name, param in [
                ('xyz', self._xyz),
                ('opacity', self._opacity),
                ('nu', self._nu),
                ('scaling', self._scaling),
                ('rotation', self._rotation),
                ('density', self._density)
            ]:
                if param_group['name'] == param_name:
                    state = self.optimizer.state[param]
                    if len(state) > 0:
                        # 清除动量
                        if 'exp_avg' in state:
                            state['exp_avg'][dead_indices] = 0
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'][dead_indices] = 0

        print(f"🔄 [SSS-Recycle] Recycled {num_to_recycle}/{num_dead} dead components "
              f"(threshold={opacity_threshold}, cap={max_recycle})")
```

#### 步骤 2：在 `train.py` 中启用组件回收（替换 densification）

修改训练循环中的密化逻辑（行 865-980 区域）：

```python
# 在行 865 附近，densification 循环开始前添加：

# SSS: Component Recycling（替代传统 densification）
if iteration < opt.densify_until_iter:
    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
        for i in range(gaussiansN):
            if hasattr(GsDict[f"gs{i}"], 'use_student_t') and GsDict[f"gs{i}"].use_student_t:
                # 🎯 [SSS-Official] 使用组件回收机制（不是传统 densification）
                print(f"🔄 [SSS-Recycle] Iter {iteration}: Applying component recycling for GS{i}")
                GsDict[f"gs{i}"].recycle_components(
                    opacity_threshold=0.005,  # 官方阈值
                    max_recycle_ratio=0.05    # 每次最多 5%
                )
            else:
                # 标准 Gaussian 模型：使用传统 densification
                # ... 保留原有 densify_and_prune 逻辑 ...
```

**完整修改建议：**

将行 910-979（标准 densification 逻辑）修改为：

```python
# 标准密化和剪枝流程
for i in range(gaussiansN):
    # SSS: 使用组件回收机制
    if hasattr(GsDict[f"gs{i}"], 'use_student_t') and GsDict[f"gs{i}"].use_student_t:
        print(f"🔄 [SSS-Official] Iter {iteration}: GS{i} using component recycling (not densification)")
        GsDict[f"gs{i}"].recycle_components(
            opacity_threshold=0.005,
            max_recycle_ratio=0.05
        )
    else:
        # 非 SSS 模型：传统 densification
        if hasattr(GsDict[f"gs{i}"], 'enhanced_densify_and_prune'):
            print(f"✅ [Densify] Iter {iteration}: GS{i} 使用 FSGS enhanced_densify_and_prune")
            GsDict[f"gs{i}"].enhanced_densify_and_prune(
                opt.densify_grad_threshold,
                opt.density_min_threshold,
                opt.max_screen_size,
                max_scale,
                opt.max_num_gaussians,
                densify_scale_threshold,
                bbox,
                enable_proximity_densify=enable_fsgs_proximity,
            )
        else:
            # 回退到标准密化
            print(f"⚠️ [Densify] Iter {iteration}: GS{i} 回退到标准 densify_and_prune (无FSGS)")
            GsDict[f"gs{i}"].densify_and_prune(
                opt.densify_grad_threshold,
                opt.density_min_threshold,
                opt.max_screen_size,
                max_scale,
                opt.max_num_gaussians,
                densify_scale_threshold,
                bbox,
            )
```

---

## 第三部分：修复执行计划

### 阶段 1：简单修复（Bug 1-3）

**优先级：🔥 高（立即执行）**

#### 修复 1.1：启用 SSS（Bug 1）

```python
# 文件: train.py
# 行号: 142
# 操作: Edit

# 旧代码:
use_student_t = False  # 强制禁用 SSS

# 新代码:
use_student_t = args.enable_sss  # 允许通过命令行参数启用
```

#### 修复 1.2：Opacity 激活函数（Bug 2）

```python
# 文件: r2_gaussian/gaussian/gaussian_model.py
# 行号: 72-78
# 操作: Edit

# 旧代码:
self.opacity_activation = lambda x: torch.sigmoid(x) * 1.2 - 0.2  # [-0.2, 1.0]
self.opacity_inverse_activation = lambda x: inverse_sigmoid(
    (torch.clamp(x, -0.19, 0.99) + 0.2) / 1.2
)

# 新代码:
# 🎯 [SSS-Official] 使用 tanh 激活函数 [-1, 1]（官方实现）
self.opacity_activation = torch.tanh
self.opacity_inverse_activation = lambda x: 0.5 * torch.log(
    (1 + torch.clamp(x, -0.99, 0.99)) / (1 - torch.clamp(x, -0.99, 0.99))
)  # inverse_tanh
```

```python
# 文件: r2_gaussian/gaussian/gaussian_model.py
# 行号: 291-295
# 操作: Edit

# 旧代码:
opacity_vals = torch.sigmoid(fused_density.clone()) * 0.9  # [0, 0.9]
opacity_init = self.opacity_inverse_activation(opacity_vals)
self._opacity = nn.Parameter(opacity_init.requires_grad_(True))

# 新代码:
# 🎯 [SSS-Official] 初始化为 0.5（官方策略）
opacity_vals = torch.ones_like(fused_density) * 0.5
opacity_init = self.opacity_inverse_activation(opacity_vals)
self._opacity = nn.Parameter(opacity_init.requires_grad_(True))
```

```python
# 文件: r2_gaussian/gaussian/gaussian_model.py
# 行号: 201-206（get_opacity 属性）
# 操作: 修改或添加 clamp

# 确保 get_opacity 中有官方的 clamp 逻辑
@property
def get_opacity(self):
    if self.use_student_t:
        opacity = self.opacity_activation(self._opacity)
        # 🎯 [SSS-Official] Clamp 到 [-1+ε, 1-ε]
        return torch.clamp(opacity, -1.0 + 1e-5, 1.0 - 1e-5)
    else:
        return self.density_activation(self._density)
```

#### 修复 1.3：移除自创 Balance Loss（Bug 3）

```python
# 文件: train.py
# 行号: 792-843
# 操作: 删除旧代码，替换为官方实现

# 删除所有行 792-843 的代码（包括注释）

# 在行 792 位置插入官方 Balance Loss:
# 🎯 [SSS-Official] Balance Loss: L1 正则化
if hasattr(GsDict[f"gs{i}"], 'use_student_t') and GsDict[f"gs{i}"].use_student_t:
    opacity = GsDict[f"gs{i}"].get_opacity
    opacity_reg_weight = 0.01  # 官方默认权重
    balance_loss = opacity_reg_weight * torch.abs(opacity).mean()
    LossDict[f"loss_gs{i}"] += balance_loss

    # 简化日志（每 2000 次迭代）
    if iteration % 2000 == 0:
        pos_ratio = (opacity > 0).float().mean()
        neg_ratio = (opacity < 0).float().mean()
        nu = GsDict[f"gs{i}"].get_nu
        print(f"🎯 [SSS-Official] Iter {iteration}: "
              f"Opacity [{opacity.min():.3f}, {opacity.max():.3f}], "
              f"Pos/Neg: {pos_ratio*100:.1f}%/{neg_ratio*100:.1f}%, "
              f"Nu: [{nu.min():.2f}, {nu.max():.2f}], "
              f"Balance Loss: {balance_loss.item():.6f}")
```

### 阶段 2：组件回收实现（Bug 5）

**优先级：🔥 关键（核心功能）**

#### 修复 2.1：添加 `recycle_components` 方法

在 `r2_gaussian/gaussian/gaussian_model.py` 中添加完整方法（见上文"步骤 1"的完整代码）

#### 修复 2.2：在训练循环中启用

修改 `train.py` 行 910-979 的 densification 逻辑（见上文"步骤 2"的完整代码）

---

## 第四部分：验证与测试

### 修复后的训练命令

```bash
# 启用 SSS 的完整训练命令
python train.py \
    --source_path data/foot/foot_3views \
    --model_path output/2025_11_18_foot_3views_sss_fixed \
    --config configs/foot_3views.yaml \
    --enable_sss \
    --iterations 20000 \
    --test_iterations 5000 10000 20000 \
    --save_iterations 5000 10000 20000 \
    --gaussiansN 1 \
    --coreg False
```

### 预期效果

| 指标 | Baseline | 用户 Bug 版本 | 修复后预期 |
|------|----------|---------------|------------|
| **PSNR (3D)** | ~18.99 dB | 10.60 dB | **≥ 18.99 dB** |
| **SSIM (3D)** | ~0.88 | 0.83 | **≥ 0.88** |
| **训练稳定性** | 稳定 | 易崩溃 | 稳定 |
| **Opacity 平衡** | - | 全负值 | **~70% 正值 / 30% 负值** |

### 监控指标

在训练过程中观察以下关键指标（TensorBoard）：

1. **Opacity 分布**
   - 范围应在 **[-1, 1]**
   - 正值比例应在 **60%-80%**
   - 极端值（< -0.9 或 > 0.9）应少于 **5%**

2. **Balance Loss**
   - 应在 **0.001 - 0.01** 之间
   - 趋势应平稳下降

3. **组件回收**
   - 每次回收 **≤ 5% 总组件数**
   - 回收频率：每 **100-500 iterations**

4. **Nu (自由度)**
   - 范围应在 **[2, 8]**
   - 平均值约 **4-6**

### 诊断命令

如果修复后仍有问题，运行以下命令诊断：

```bash
# 检查初始化
python train.py --source_path data/foot/foot_3views \
    --model_path output/test_init \
    --enable_sss \
    --iterations 1 \
    --test_iterations 1

# 查看日志输出中的初始化信息：
# ✅ "SSS-v6-FIX] Initialize N Student's t distributions"
# ✅ "Initialized nu: [...], opacity: [...], positive: ...%"

# 检查组件回收
grep "SSS-Recycle" output/test_sss/log.txt
```

---

## 第五部分：风险与注意事项

### 已知风险

1. **CUDA 光栅化器兼容性**
   - 用户可能没有支持 Student-t 的 CUDA 光栅化器
   - **解决方案：** 检查 `submodules/diff-t-rasterization` 是否存在

2. **优化器状态重置**
   - 组件回收需要清除优化器动量
   - **风险：** 如果用户使用的不是 Adam/SGHMC，可能失败
   - **缓解：** 添加 try-except 保护

3. **内存消耗**
   - Student-t 分布需要额外的 `nu` 和 `opacity` 参数
   - **预期增加：** ~20% 内存

### 回滚计划

如果修复后性能仍差，按以下顺序回滚：

1. **仅保留 Bug 1-2 修复**（tanh + 启用 SSS）
2. **禁用组件回收**，使用传统 densification
3. **调整 Balance Loss 权重**（从 0.01 减少到 0.001）

---

## 第六部分：后续优化

修复完成后，可考虑以下优化：

1. **SGHMC 优化器**
   - 官方使用 SGHMC 而非 Adam
   - 需要在 `training_setup` 中切换

2. **两阶段训练**
   - Burn-in 阶段（iter 0-7000）：高噪声 C_burnin
   - 主训练阶段（iter 7000+）：低噪声 C

3. **Opacity Reset**
   - 官方每 3000 次迭代重置低 opacity 组件
   - 在 `recycle_components` 中集成

4. **Scale Regularization**
   - 官方还有 `scale_reg * torch.abs(scaling).mean()`
   - 可添加到损失函数

---

## 附录：官方代码关键片段

### A. Opacity 激活函数（`nt_model.py`）

```python
class NTModel:
    def __init__(self, ...):
        # Activation functions
        self.opacity_activation = torch.tanh
        self.inverse_opacity_activation = self._inverse_tanh

    def _inverse_tanh(self, x):
        x = torch.clamp(x, -0.99, 0.99)
        return 0.5 * torch.log((1 + x) / (1 - x))

    @property
    def get_opacity(self):
        opacity = self.opacity_activation(self._opacity)
        return torch.clamp(opacity, -1.0 + 1e-5, 1.0 - 1e-5)
```

### B. Balance Loss（`train.py`）

```python
# Line 104 in official train.py
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(...))

# Regularization terms
opacity_reg = args.opacity_reg * torch.abs(primitives.get_opacity).mean()
scale_reg = args.scale_reg * torch.abs(primitives.get_scaling).mean()

loss += opacity_reg + scale_reg
```

### C. 组件回收核心逻辑（`nt_model.py`）

```python
def recycle_components(self):
    # Dead component detection
    alive_mask = self.get_opacity > self.opacity_threshold
    dead_indices = torch.where(~alive_mask)[0]

    # 5% cap
    max_recycle = int(0.05 * self._xyz.shape[0])
    if len(dead_indices) > max_recycle:
        dead_indices = dead_indices[:max_recycle]

    # Resample from alive components
    alive_indices = torch.where(alive_mask)[0]
    weights = self.get_opacity[alive_mask].squeeze()
    sample_indices = torch.multinomial(weights, len(dead_indices), replacement=True)

    # Reinitialize
    self._xyz[dead_indices] = self._xyz[alive_indices[sample_indices]] + noise
    self._opacity[dead_indices] = self.inverse_opacity_activation(torch.ones(...) * 0.5)
```

---

**总结：**

本修复方案基于官方代码的深入研究，修复了用户实现的 5 个致命 bug：

1. ✅ **Bug 1：** 启用 SSS（`use_student_t = True`）
2. ✅ **Bug 2：** 修正 Opacity 激活函数（`tanh` 替代 `sigmoid * 1.2 - 0.2`）
3. ✅ **Bug 3：** 移除自创的渐进式 Scooping 限制
4. ✅ **Bug 4：** 使用论文的 Balance Loss（`0.01 * |opacity|`）
5. ✅ **Bug 5：** 实现组件回收机制（替代传统 densification）

预期修复后，性能将从 **10.60 dB** 恢复到 Baseline **18.99 dB** 或更好。

**下一步：** 等待用户确认后执行修复。
