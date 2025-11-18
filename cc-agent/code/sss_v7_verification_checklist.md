# SSS-v7-OFFICIAL 验证检查清单

**生成日期：** 2025-11-18
**版本：** v7-OFFICIAL（官方实现）
**修复内容：** 全部 5 个 bug

---

## 🎯 快速启动

### 立即开始训练

```bash
# 进入项目目录
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 激活环境
conda activate r2_gaussian_new

# 运行训练（30k 迭代，约 8-10 小时）
bash scripts/train_foot3_sss_v7_official.sh
```

---

## ✅ 训练启动检查清单

### 1. 环境检查（训练前）

- [ ] **Conda 环境**：已激活 `r2_gaussian_new`
- [ ] **CUDA 可用**：运行 `nvidia-smi` 确认 GPU 可用
- [ ] **数据集存在**：`data/369/foot_50_3views.pickle` 存在
- [ ] **磁盘空间**：输出目录 `output/` 有足够空间（至少 5 GB）

**验证命令：**
```bash
conda activate r2_gaussian_new
nvidia-smi
ls -lh data/369/foot_50_3views.pickle
df -h output/
```

### 2. 代码修复验证（训练前）

- [ ] **Bug 1 修复**：检查 `train.py:142` 是否为 `use_student_t = args.enable_sss`
- [ ] **Bug 2 修复**：检查 `gaussian_model.py:73` 是否使用 `torch.tanh`
- [ ] **Bug 3+4 修复**：检查 `train.py:796-810` 是否为官方 L1 正则
- [ ] **Bug 5 修复**：检查 `gaussian_model.py` 是否包含 `recycle_components` 方法

**快速验证命令：**
```bash
# Bug 1
grep "use_student_t = args.enable_sss" train.py

# Bug 2
grep "self.opacity_activation = torch.tanh" r2_gaussian/gaussian/gaussian_model.py

# Bug 3+4
grep "opacity_reg_weight = 0.01" train.py

# Bug 5
grep "def recycle_components" r2_gaussian/gaussian/gaussian_model.py
```

---

## 📊 训练过程监控

### 3. 启动日志检查（训练开始后 30 秒）

**预期输出：**
```
🎓 [SSS-Official] Using Student's t-distribution model
   Opacity activation: torch.tanh, range [-1, 1]
   Balance Loss: L1 regularization (weight=0.01)
   Component Recycling: enabled (threshold=0.005)
```

**检查方法：**
```bash
# 实时查看训练日志
tail -f output/2025_11_18_foot_3views_sss_v7_official_train.log
```

**❌ 如果没看到上述日志，说明 SSS 未启用，立即停止训练检查！**

### 4. 初始化验证（Iteration 1）

**预期输出（迭代 1）：**
```
🎯 [SSS-Official] Iter 2000: Opacity [-0.95, 0.98], Balance: 55% pos / 45% neg
```

**关键指标：**
- ✅ Opacity 范围：应在 `[-1, 1]` 之间（接近边界）
- ✅ Positive Ratio：应在 **40-60%**（不是 100%！）
- ✅ 无 NaN、无 Inf

**❌ 异常情况：**
- 如果 Positive Ratio = 100%：SSS 未生效
- 如果 Opacity 范围在 `[-0.2, 1.0]`：Bug 2 未修复
- 如果出现 NaN：立即停止，检查学习率

### 5. 组件回收验证（Iteration ~100）

**预期输出（每次 densification 间隔）：**
```
♻️ [SSS-Recycle] Recycled 234/456 dead components (4.7% of total)
```

**关键指标：**
- ✅ 每次回收 **~5%** 的总组件数
- ✅ Dead components 应该存在（不应该是 0）

**检查方法：**
```bash
# 查看组件回收记录
grep "SSS-Recycle" output/2025_11_18_foot_3views_sss_v7_official_train.log
```

**❌ 异常情况：**
- 如果没有任何 "SSS-Recycle" 日志：Bug 5 未生效，使用了传统 densification
- 如果回收比例 > 10%：阈值设置过高

### 6. 训练稳定性监控（持续）

**监控指标（每 2000 次迭代）：**

| 指标 | 健康范围 | 异常阈值 |
|------|---------|---------|
| **PSNR** | 持续上升 | 下降或停滞 |
| **Opacity 平衡** | 40-60% 正值 | <20% 或 >80% |
| **Balance Loss** | 0.3-0.5 | >1.0 或 <0.1 |
| **组件回收频率** | 每 100 次迭代 | 从不回收 |

**监控命令：**
```bash
# 实时监控关键指标
watch -n 60 'grep -E "SSS-Official.*Iter [0-9]+000:" output/2025_11_18_foot_3views_sss_v7_official_train.log | tail -5'
```

---

## 🎯 训练完成验证

### 7. 最终性能评估（Iteration 30000）

**预期性能（Foot 3 views）：**

| 指标 | 目标值 | Baseline 对比 |
|------|--------|--------------|
| **2D PSNR** | **≥ 28.0 dB** | Baseline: 28.31 dB |
| **2D SSIM** | **≥ 0.88** | Baseline: 0.898 |
| **Positive Ratio** | **40-60%** | - |
| **训练稳定性** | **无 NaN/Inf** | - |

**检查方法：**
```bash
# 查看最终结果
cat output/2025_11_18_foot_3views_sss_v7_official/eval/iter_030000/eval2d_render_test.yml
```

**✅ 成功标准：**
- PSNR ≥ 28 dB（接近或超过 Baseline）
- SSIM ≥ 0.88
- Opacity 平衡在 40-60%
- 无 NaN/Inf

**🟡 部分成功（需优化）：**
- PSNR 25-28 dB（接近但未达标）
- 可调整超参数：
  - 降低 `opacity_lr`（0.005 → 0.002）
  - 调整 `opacity_reg`（0.01 → 0.001）

**❌ 失败（需深入诊断）：**
- PSNR < 25 dB
- 全部 opacity 为负值或正值
- 训练崩溃

### 8. 对比 Baseline 验证

**运行 Baseline 对比实验：**
```bash
# 运行标准 3DGS Baseline（不启用 SSS）
python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/test_baseline \
    --iterations 30000 \
    --eval
```

**对比指标：**
- SSS-v7 vs Baseline：PSNR 差距应 ≤ 0.5 dB
- 如果 SSS-v7 < Baseline - 1 dB：可能仍存在问题

---

## 🔍 常见问题诊断

### 问题 1：SSS 未启用（仍在运行 Baseline）

**症状：**
- 日志中没有 "SSS-Official" 字样
- Positive Ratio 始终 100%
- 没有 "SSS-Recycle" 日志

**诊断：**
```bash
# 检查训练启动参数
grep "enable_sss" output/2025_11_18_foot_3views_sss_v7_official_train.log
```

**解决：**
- 确认训练脚本包含 `--enable_sss` 参数
- 检查 `train.py:142` 是否正确修复

### 问题 2：Opacity 全部为负值或正值

**症状：**
- Positive Ratio 接近 0% 或 100%
- PSNR 很低（< 20 dB）

**诊断：**
```bash
# 检查 opacity 激活函数
grep "opacity_activation" r2_gaussian/gaussian/gaussian_model.py
```

**解决：**
- 确认使用 `torch.tanh`
- 检查初始化是否正确（应为 0.5）
- 降低 `opacity_lr`（0.005 → 0.002）

### 问题 3：组件回收未执行

**症状：**
- 日志中没有 "SSS-Recycle" 字样
- 组件数量持续增长（类似传统 densification）

**诊断：**
```bash
# 检查是否定义了 recycle_components 方法
grep -A 5 "def recycle_components" r2_gaussian/gaussian/gaussian_model.py
```

**解决：**
- 确认 `recycle_components` 方法已添加
- 检查训练循环是否调用了该方法
- 查看代码修复摘要：`cc-agent/code/sss_bug_fix_summary.md`

### 问题 4：训练崩溃或 NaN

**症状：**
- Loss 出现 NaN
- 程序异常终止

**诊断：**
```bash
# 查找 NaN 相关日志
grep -i "nan\|inf" output/2025_11_18_foot_3views_sss_v7_official_train.log
```

**解决：**
- 降低所有学习率（减半）
- 增加 `opacity_reg`（0.01 → 0.05）
- 检查数据集是否正常

---

## 📋 性能基准对比

### Foot 3 views（30k 迭代）

| 方法 | PSNR (dB) | SSIM | 备注 |
|------|-----------|------|------|
| **R²-Gaussian Baseline** | 28.31 | 0.898 | 参考基线 |
| **FSGS** | 28.45 | 0.901 | 增强版 |
| **SSS-v5（Bug 版）** | 20.16 | 0.778 | ❌ 完全失败 |
| **SSS-v6（部分修复）** | 训练中断 | - | ⏸️ 未完成 |
| **SSS-v7（官方实现）** | **待验证** | **待验证** | 🎯 预期 ≥28 dB |

---

## 📖 参考文档

1. **修复方案详细版**：`cc-agent/code/sss_bug_fix_plan.md`
2. **修复摘要报告**：`cc-agent/code/sss_bug_fix_summary.md`
3. **论文分析**：`cc-agent/3dgs_expert/sss_innovation_analysis.md`
4. **官方代码仓库**：https://github.com/realcrane/3D-student-splatting-and-scooping

---

## 🚀 下一步行动

### 立即执行

**1. 启动训练（推荐）**
```bash
bash scripts/train_foot3_sss_v7_official.sh
```

**2. 监控训练（另开一个终端）**
```bash
# 实时查看关键指标
watch -n 30 'tail -n 20 output/2025_11_18_foot_3views_sss_v7_official_train.log'
```

**3. 定期检查（每 2 小时）**
- 查看 Opacity 平衡是否正常（40-60%）
- 确认组件回收正在执行
- 观察 PSNR 是否上升

### 训练完成后

**1. 分析结果**
```bash
# 查看详细结果
cat output/2025_11_18_foot_3views_sss_v7_official/eval/iter_030000/eval2d_render_test.yml
```

**2. 对比 Baseline**
- 如果 PSNR ≥ 28 dB：✅ 修复成功！
- 如果 PSNR 25-28 dB：🟡 需微调超参数
- 如果 PSNR < 25 dB：❌ 需深入诊断

**3. 记录结果**
```bash
# 调用项目协调员记录
/record "SSS-v7-OFFICIAL 训练完成，PSNR=XX.XX dB"
```

---

**预祝训练成功！** 🎉

如遇到任何问题，请参考本文档的"常见问题诊断"部分，或查看详细修复报告。
