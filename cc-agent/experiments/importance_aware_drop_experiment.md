# Importance-Aware Drop 实验方案

**实验日期**: 2025-11-19
**优先级**: P3（高级策略）
**目标**: 通过保护高 opacity Gaussians 提升 DropGaussian 性能，超越 baseline

---

## 一、实验动机

根据 DropGaussian 失败诊断，核心问题是：
- 平均 opacity 下降 44.47%
- 高 opacity (>0.5) Gaussians 减少 97.2%（106 → 3）
- **关键 Insight**: 高 opacity Gaussians 对渲染质量至关重要，应该被保护

**假设**: 如果保护 top 20% 高 opacity 的 Gaussians，可以：
1. 维持高质量 Gaussians 的数量
2. 保持正则化效果（仍然 drop 80% 的低质量 Gaussians）
3. 最终性能超越 baseline

---

## 二、Importance-Aware Drop 策略设计

### 2.1 核心思想

**不是均匀随机 drop**，而是基于 opacity 重要性自适应 drop：
- **Top 20% 高 opacity Gaussians**: drop rate × 0.2（大幅保护）
- **Bottom 80% 低 opacity Gaussians**: drop rate × 1.0（正常 drop）

### 2.2 实现逻辑

```python
# 1. 计算激活后的 opacity
opacity_activated = torch.sigmoid(density)

# 2. 确定 top 20% 阈值
k = int(len(opacity_activated) * 0.2)
threshold = torch.topk(opacity_activated, k)[0][-1]

# 3. 创建自适应 drop rate
adaptive_drop_rate = torch.where(
    opacity_activated >= threshold,
    drop_rate * 0.2,  # 保护高 opacity
    drop_rate         # 正常 drop 低 opacity
)

# 4. 应用自适应 dropout
random_mask = torch.rand(len(adaptive_drop_rate), device="cuda")
keep_mask = random_mask > adaptive_drop_rate
compensation = 1.0 / (1.0 - adaptive_drop_rate) * keep_mask
```

### 2.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_drop_gaussian` | True | 启用 DropGaussian |
| `drop_gamma` | 0.2 | 基础 drop rate（与原实验一致）|
| `use_importance_aware_drop` | True | 启用 Importance-Aware 策略 |
| `importance_protect_ratio` | 0.2 | 保护 top 20% 高 opacity |

---

## 三、实验配置

### 3.1 训练命令

```bash
conda activate r2_gaussian_new
python train.py \
  -s data/369/foot_50_3views.pickle \
  -m ./output/2025_11_19_foot_3views_importance_aware_drop \
  --iterations 30000 \
  --use_drop_gaussian \
  --drop_gamma 0.2 \
  --use_importance_aware_drop \
  --eval
```

### 3.2 成功标准

| 指标 | Baseline | DropGaussian (γ=0.2) | Importance-Aware 目标 |
|------|----------|----------------------|----------------------|
| **PSNR (dB)** | 28.55 | 28.12 | **> 28.98** ✅ |
| **SSIM** | 0.9008 | 0.9015 | **> 0.905** ✅ |
| **高 Opacity (>0.5)** | 106 | 3 | **> 80** ✅ |
| **平均 Opacity** | 0.046 | 0.025 | **> 0.040** ✅ |

**目标**: 超越 baseline，证明 Importance-Aware 策略有效

---

## 四、预期效果

### 4.1 定量预期

1. **Opacity 分布改善**:
   - 平均 opacity 应保持在 0.040 以上（vs DropGaussian 的 0.025）
   - 高 opacity Gaussians 应保持 > 80 个（vs DropGaussian 的 3 个）

2. **渲染质量提升**:
   - PSNR 应超过 28.98 dB（超越 baseline 的 28.55 dB）
   - SSIM 应超过 0.905（超越 baseline 的 0.9008）

3. **Good/Fail Cases 改善**:
   - Good Cases 应从 26% 提升到 > 50%
   - Fail Cases 中性能下降幅度应减小

### 4.2 定性预期

- 保护机制应让训练更稳定（后期不会性能下降）
- 高 opacity Gaussians 持续存在（不会在训练中消失）
- 仍然保持正则化效果（避免过拟合）

---

## 五、对比实验

| 实验组 | use_drop_gaussian | use_importance_aware_drop | drop_gamma | 预期 PSNR |
|--------|-------------------|---------------------------|------------|-----------|
| Baseline | False | False | - | 28.55 |
| DropGaussian (原版) | True | False | 0.2 | 28.12 |
| **Importance-Aware** | True | **True** | 0.2 | **> 28.98** |

---

## 六、风险与备选方案

### 6.1 潜在风险

1. **保护比例不足**:
   - 风险：20% 可能不够，导致高 opacity Gaussians 仍然减少
   - 备选：调整 `importance_protect_ratio` 到 0.3（30%）或 0.4（40%）

2. **保护因子不足**:
   - 风险：drop rate × 0.2 可能仍然过高
   - 备选：调整保护因子到 0.1（drop rate × 0.1）甚至 0.0（完全保护）

3. **动态阈值问题**:
   - 风险：训练初期 opacity 都很低，top 20% 阈值可能不合理
   - 备选：设置最小阈值（如 opacity > 0.1 才考虑保护）

### 6.2 后续实验（如果成功）

如果 Importance-Aware Drop 成功（PSNR > 28.98），可以尝试：
- 测试其他器官（Chest, Head, Abdomen, Pancreas）
- 测试 6 视角和 9 视角场景
- 调优保护比例和保护因子
- 与其他策略组合（如 Curriculum Drop）

---

## 七、实验时间线

| 阶段 | 时间 | 任务 |
|------|------|------|
| **代码实现** | 完成 | 修改 `arguments/__init__.py` 和 `render_query.py` |
| **启动训练** | T+0 min | 执行训练命令 |
| **中期检查** | T+15 min | 查看 TensorBoard 曲线（5000 iter）|
| **训练完成** | T+30 min | 30000 iterations 训练完成 |
| **结果分析** | T+35 min | 运行分析脚本，生成报告 |
| **决策点** | T+40 min | 决定是否继续其他实验 |

---

## 八、分析脚本（复用已有工具）

训练完成后，使用以下脚本分析：

```bash
# 1. Opacity 统计对比
~/anaconda3/envs/r2_gaussian_new/bin/python cc-agent/experiments/analyze_opacity.py

# 2. 逐图 PSNR/SSIM 对比
~/anaconda3/envs/r2_gaussian_new/bin/python cc-agent/experiments/analyze_test_cases.py

# 3. 对比可视化
~/anaconda3/envs/r2_gaussian_new/bin/python cc-agent/experiments/visualize_baseline_cases.py
```

---

**实验负责人**: AI Research Assistant
**审核状态**: ⏸️ 等待用户批准执行
**数据完整性**: ✅ 所有策略基于前期诊断数据
