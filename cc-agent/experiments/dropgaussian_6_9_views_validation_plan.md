# DropGaussian 6/9 Views 验证实验计划

**实验日期：** 2025-11-19
**目的：** 验证 DropGaussian 在足够视角数下的有效性
**假设：** DropGaussian 在 6+ views 下有效，3 views 失败是因为低于适用下限

---

## 🎯 实验目的

### **核心假设**
根据之前的失败分析：
- **3 views：** 所有 DropGaussian 策略均失败（PSNR 28.12-28.34 < Baseline 28.50）
- **根本原因：** 3 views < DropGaussian 论文最低要求（6 views）

### **验证目标**
1. **6 views（论文下限）：** DropGaussian 是否开始有效？
2. **9 views（论文典型）：** DropGaussian 是否显著提升性能？

如果实验证明 DropGaussian 在 6/9 views 下有效，则说明：
- ✅ 我们的分析是正确的（3 views 太少）
- ✅ DropGaussian 本身没问题
- ✅ 需要至少 6 views 才能使用 DropGaussian

---

## 📊 实验设计

### **实验矩阵**

| 视角数 | 策略 | 预期结果 | 关键指标 |
|-------|------|---------|---------|
| **3 views** | Curriculum (γ=0.1) | ❌ 失败（已验证）| PSNR 28.34 < 28.50 |
| **6 views** | Curriculum (γ=0.1) | ⚠️  可能有效？ | 目标：≥ Baseline |
| **9 views** | Curriculum (γ=0.1) | ✅ 应该有效 | 目标：> Baseline |

### **统一配置**

所有实验使用相同的 DropGaussian 配置（已证明是最优策略）：

```bash
--use_drop_gaussian \
--drop_gamma 0.1 \
--drop_start_iter 5000 \
--drop_end_iter 30000
```

**策略特点：**
- 前 5000 轮不 drop（保证前期稳定）
- 5000-30000 轮线性增长到 10%
- 最大 drop rate = 0.1（比论文 0.2 降低 50%）

---

## 📈 预期结果

### **场景 A：6 views 实验**

#### **如果成功（PSNR ≥ Baseline）**
- ✅ 证明 6 views 是 DropGaussian 的适用下限
- ✅ 验证我们的分析：3 views < 6 views 论文要求
- ✅ DropGaussian 本身没问题

#### **如果失败（PSNR < Baseline）**
- ⚠️  6 views 仍然不够
- ⚠️  可能需要至少 8-10 views
- ⚠️  或者 CT 场景与 RGB 场景不同

### **场景 B：9 views 实验**

#### **如果成功（PSNR > Baseline）**
- ✅ 证明 DropGaussian 在典型场景下有效
- ✅ 验证论文效果（论文使用 9-10 views）
- ✅ 为稀疏场景提供参考（需要多少视角）

#### **如果失败（PSNR < Baseline）**
- ❌ DropGaussian 可能不适用于 CT 场景
- ❌ CT 与 RGB 的差异超出预期
- ❌ 需要重新考虑正则化策略

---

## 🔍 对比分析计划

### **3/6/9 Views 完整对比表**

| 指标 | 3 Views | 6 Views | 9 Views | 趋势 |
|------|---------|---------|---------|------|
| **Baseline PSNR** | 28.50 | ? | ? | - |
| **DropGaussian PSNR** | 28.34 | ? | ? | ? |
| **差距** | -0.17 | ? | ? | ? |
| **改善图像占比** | 40% | ? | ? | ? |
| **DropGaussian 效果** | ❌ 失败 | ? | ? | ? |

### **关键分析点**

1. **PSNR 趋势**
   - 3 → 6 → 9 views，DropGaussian 是否逐渐改善？
   - 在哪个视角数开始超越 baseline？

2. **改善图像占比**
   - 3 views：40%
   - 6 views：目标 > 50%
   - 9 views：目标 > 60%

3. **训练曲线对比**
   - 前期（0-5000 轮）：不 drop 阶段
   - 中期（5000-20000 轮）：drop 增长阶段
   - 后期（20000-30000 轮）：稳定 drop 阶段

---

## 🚀 实验流程

### **Phase 1：启动实验（并行）**

```bash
# 6 views 实验
nohup bash scripts/train_dropgaussian_6views.sh > logs/dropgaussian_6views.log 2>&1 &

# 9 views 实验
nohup bash scripts/train_dropgaussian_9views.sh > logs/dropgaussian_9views.log 2>&1 &
```

**预计训练时间：**
- 6 views：约 3-4 小时
- 9 views：约 4-5 小时

### **Phase 2：监控训练**

**关键检查点：**
| 迭代 | Drop Rate | 检查项 |
|------|-----------|-------|
| 5,000 | 0% | 前期 PSNR 应该领先 baseline |
| 10,000 | 2% | 开始 drop 后是否仍然领先 |
| 20,000 | 6% | 中期评估 |
| 30,000 | 10% | 最终结果 |

### **Phase 3：结果分析**

完成后生成完整对比报告：
- `cc-agent/experiments/dropgaussian_3_6_9_views_comparison.md`

---

## 📋 成功标准

### **最低要求（P0）**
- ✅ 6 views 或 9 views 至少一个成功（PSNR ≥ Baseline）
- ✅ 证明 DropGaussian 在足够视角下有效

### **目标要求（P1）**
- ✅ 6 views：PSNR ≥ Baseline（证明 6 views 是下限）
- ✅ 9 views：PSNR > Baseline（证明论文效果可复现）

### **理想要求（P2）**
- ✅ 明确找到 DropGaussian 的视角数阈值
- ✅ 为稀疏场景提供明确指导（需要多少视角）

---

## 🔄 如果失败怎么办？

### **场景：6 views 和 9 views 都失败**

**可能原因：**
1. CT 场景与 RGB 场景差异太大
2. DropGaussian 不适用于 X-ray 投影
3. 需要特定的 CT 正则化策略

**下一步：**
- ❌ 完全放弃 DropGaussian
- ✅ 转向 CT 专用正则化（Graph Laplacian, TV）
- ✅ 研究其他 CT 重建论文的正则化策略

---

## 📚 理论背景

### **DropGaussian 论文设置**

| 数据集 | 视角数 | 效果 |
|-------|-------|------|
| DTU | 9 | ✅ 提升 |
| NeRF Synthetic | 10-12 | ✅ 提升 |
| 最少测试 | **6** | ✅ 有效（论文下限） |

### **我们的场景**

| 场景 | 视角数 | 预期 |
|------|-------|------|
| Foot-3 | 3 | ❌ 失败（已验证） |
| Foot-6 | 6 | ⚠️  待验证（论文下限） |
| Foot-9 | 9 | ✅ 待验证（论文典型） |

---

## 🎯 核心问题

**这个实验将回答的核心问题：**

1. **DropGaussian 是否真的有效？**
   - 如果 6/9 views 成功 → ✅ 有效
   - 如果都失败 → ❌ 不适用于 CT

2. **多少视角才能使用 DropGaussian？**
   - 如果 6 views 成功 → 至少 6 views
   - 如果 6 views 失败但 9 views 成功 → 至少 7-9 views
   - 如果都失败 → 可能需要 > 10 views 或不适用

3. **3 views 失败的原因是否正确？**
   - 如果 6/9 views 成功 → ✅ 证明是视角数太少
   - 如果都失败 → ❌ 可能有其他原因

---

**实验准备时间：** 2025-11-19
**预计完成时间：** 4-5 小时后
**负责人：** Claude Code
**状态：** ✅ 准备就绪，等待启动
