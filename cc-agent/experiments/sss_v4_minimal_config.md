# SSS v4-minimal 诊断实验配置

**创建时间**: 2025-11-17 15:33
**实验目标**: 诊断 v2/v3 性能崩溃的根本原因，使用最小化正则化策略

---

## 🔬 实验背景

### 历史版本问题分析

| 版本 | iter 1000 PSNR | 核心问题 | 正则化策略 | 初始化策略 |
|------|---------------|---------|-----------|-----------|
| **v1** | 21.32 dB | ✅ 性能优秀但 iter 5000 崩溃 (99.8% 负 opacity) | balance_loss=0.001 (过弱) | 无 clamp |
| **v2** | 8.97 dB | ❌ 性能崩溃 (下降 12 dB) | balance_loss=0.1 + 多重惩罚 (过强) | clamp [0.3, 2.0] |
| **v3** | 8.16 dB | ❌ 性能仍崩溃 (下降 13 dB) | balance_loss=0.01 + 温和惩罚 (仍过强) | clamp [0.0, 3.0] |

### 诊断假设

v2/v3 失败的可能原因：
1. **累积惩罚过重**：即使单个权重降低，4 个惩罚项累积效果仍过强
2. **初始化冲突**：强制正初始化 + 惩罚负值 → 优化器"卡"在局部最小值
3. **训练动态破坏**：负 opacity 可能在早期训练起关键作用，双重限制破坏学习

---

## 🔧 v4-minimal 配置

### 1. 正则化策略 (最小化)

**保留项**：
```python
# 🔬 [v4-MINIMAL] 仅保留最温和的 balance_loss
pos_count = (opacity > 0).float().mean()
pos_target = 0.70  # 降低目标 (90% → 70%)，允许 30% 负值
balance_loss = torch.abs(pos_count - pos_target)
LossDict[f"loss_gs{i}"] += 0.001 * balance_loss  # v1 原版权重

# Nu diversity loss (不影响 opacity)
nu_diversity_loss = -torch.std(nu) * 0.1
nu_range_loss = torch.mean(torch.relu(nu - 8.0)) + torch.mean(torch.relu(2.0 - nu))
LossDict[f"loss_gs{i}"] += 0.001 * (nu_diversity_loss + nu_range_loss)
```

**移除项** (与 v2/v3 对比)：
- ❌ `neg_opacity_penalty` (直接惩罚负值)
- ❌ `mild_neg_penalty` (温和惩罚负值)
- ❌ `extreme_penalty` (严厉惩罚极端负值)
- ❌ 分阶段策略 (phase 1: 90%, phase 2: 85%)

### 2. 初始化策略 (自由学习)

```python
# 🔬 [v4-MINIMAL] 完全移除 clamp 限制
opacity_vals = torch.sigmoid(fused_density.clone()) * 0.7 + 0.15  # [0.15, 0.85]
opacity_init = self.opacity_inverse_activation(opacity_vals)  # 转换为 tanh 的逆
# ❌ 移除 clamp: 不限制初始化范围，允许 tanh 的完整 [-inf, inf] 输入空间
self._opacity = nn.Parameter(opacity_init.requires_grad_(True))
```

**对比 v2/v3**：
- v2: `clamp(min=0.3, max=2.0)` - 强制正值
- v3: `clamp(min=0.0, max=3.0)` - 温和正值
- **v4: 无 clamp** - 完全自由

### 3. 调试日志

```python
# 🔬 [v4-MINIMAL] 每 2000 轮记录
print(f"🔬 [SSS-v4-MINIMAL] Iter {iteration}")
print(f"   Opacity: [{opacity.min():.3f}, {opacity.max():.3f}]")
print(f"   Balance: {pos_ratio*100:.1f}% pos / {neg_ratio*100:.1f}% neg (target: 70% pos)")
print(f"   Nu: mean={nu_mean:.2f}, std={nu_std:.2f}, range=[{nu.min():.1f}, {nu.max():.1f}]")
print(f"   Extremes: {extreme_pos*100:.1f}% >0.9, {extreme_neg*100:.1f}% <-0.5")
```

---

## 🎯 训练配置

```bash
conda run -n r2_gaussian_new python train.py \
  --source_path data/369/foot_50_3views.pickle \
  --model_path output/2025_11_17_foot_3views_sss_v4 \
  --iterations 10000 \
  --enable_sss \
  --nu_lr_init 0.001 \
  --opacity_lr_init 0.01 \
  --test_iterations 1000 2000 5000 10000 \
  --save_iterations 10000 \
  --eval
```

**关键参数**：
- `--enable_sss`: 启用 Student's t 分布 + 负 opacity
- `nu_lr_init=0.001`: ν 参数学习率
- `opacity_lr_init=0.01`: opacity 学习率
- 评估点: iter 1000, 2000, 5000, 10000

---

## 📊 预期结果

### 成功指标 (v4 应恢复 v1 性能)
- ✅ iter 1000: PSNR ≥ 20 dB (接近 v1 的 21.32 dB)
- ✅ iter 2000: PSNR ≥ 22 dB
- ✅ iter 5000: PSNR ≥ 25 dB (不崩溃)
- ✅ 正 opacity 比例: 70-80% (目标 70%)

### 失败指标 (如果 v4 仍失败)
- ❌ iter 1000: PSNR < 15 dB → 初始化策略也有问题
- ❌ iter 5000: 崩溃 → 需要更强的正则化（但不能用 v2/v3 的方式）
- ❌ 负 opacity > 50% → balance_loss 0.001 仍太弱

---

## 🔍 核心假设验证

| 假设 | 验证方法 | 预期结果 |
|------|---------|---------|
| **H1: 累积惩罚过重** | 移除所有直接惩罚，仅保留 balance_loss | PSNR 恢复到 20+ dB |
| **H2: 初始化冲突** | 移除 clamp 限制，自由初始化 | 早期训练更稳定 |
| **H3: 需要负 opacity** | 允许 30% 负值，观察是否提升性能 | 负值在特定区域改善重建 |

---

## 📝 下一步行动

### 如果 v4 成功 (PSNR ≥ 20 dB @ iter 1000)
1. 确认 iter 5000 是否仍崩溃
2. 如果崩溃，逐步增加 balance_loss 权重 (0.001 → 0.002 → 0.005)
3. 目标：找到最小有效正则化强度

### 如果 v4 仍失败 (PSNR < 15 dB @ iter 1000)
1. 放弃负 opacity 方案，仅保留 Student-t 分布
2. 或尝试动态正则化 (iter < 5000: 0.0001, iter ≥ 5000: 0.01)
3. 或回退到标准 Gaussian (放弃 SSS)

---

## ⏱ 时间线

- **15:33** - v4 训练启动 (PID 4088006)
- **预计 16:00** - iter 1000 评估完成 (30 分钟)
- **预计 16:30** - iter 2000 评估完成 (60 分钟)
- **预计 17:30** - iter 5000 评估完成 (120 分钟)

---

## 🔬 实验结论 (待填写)

### iter 1000 结果
- PSNR: ___ dB (目标: ≥20 dB)
- SSIM: ___ (目标: ≥0.20)
- 正 opacity 比例: ___% (目标: 70-80%)
- **结论**: ___

### iter 5000 结果
- PSNR: ___ dB (目标: ≥25 dB)
- 是否崩溃: ___
- **最终结论**: ___
