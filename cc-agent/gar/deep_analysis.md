# GAR 深度分析报告 - 每投影粒度

## 分析日期: 2025-12-06

---

## 1. 总体统计 (15 配置, 750 测试投影)

| 指标 | 数量 | 百分比 |
|------|------|--------|
| GAR 胜出投影 (>+0.05dB) | 282 | 37.6% |
| GAR 落后投影 (<-0.05dB) | 313 | 41.7% |
| 持平投影 (±0.05dB) | 155 | 20.7% |

**结论**: GAR 整体表现略差于 Baseline，胜率不足 40%。

---

## 2. 按器官分析

| 器官 | 平均差异 | 最好情况 | 最差情况 |
|------|----------|----------|----------|
| Chest | +0.023 dB | 6views +0.06 | 3views -0.02 |
| Foot | -0.044 dB | 3views **+0.21** | 6views **-0.30** |
| Head | -0.125 dB | 3views -0.08 | 9views -0.18 |
| Abdomen | -0.035 dB | 3views -0.02 | 9views -0.05 |
| Pancreas | -0.039 dB | 3views -0.01 | 6views -0.10 |

**关键发现**:
- **Foot 是唯一有显著好/坏 case 的器官**
- 其他器官差异都在 ±0.1dB 以内，几乎无影响

---

## 3. Good Case 深度分析: Foot 3views (+0.21 dB)

### 3.1 统计

- **胜/负/平**: 31/15/4 (投影级别)
- **差异范围**: [-3.86, +2.48] dB
- **标准差**: 0.95 dB (非常高！)

### 3.2 按 PSNR 区间分析

| 区间 | 帧数 | GAR 平均差异 | 分析 |
|------|------|--------------|------|
| 低 PSNR (<32dB) | 43 | **+0.34 dB** | GAR 在困难区域有帮助 |
| 高 PSNR (>40dB) | 3 | **-1.11 dB** | GAR 在简单区域反而有害 |

### 3.3 异常帧分析

- **最差帧 #23**: GAR=41.79, Base=45.65, Δ=-3.86 dB
  - 这是高 PSNR 区域，Baseline 已经很好，GAR 反而引入噪声

- **最好帧 #1**: GAR=33.59, Base=31.10, Δ=+2.48 dB
  - 低 PSNR 区域，GAR 的虚拟视角约束帮助了重建

### 3.4 结论

Foot 3views 成功的原因:
1. **稀疏视角 (3 views)** 下，几何约束更有价值
2. **低 PSNR 区域多 (43/50)**，GAR 有更多发挥空间
3. 但**高方差**表明效果不稳定

---

## 4. Bad Case 深度分析: Foot 6views (-0.30 dB)

### 4.1 统计

- **胜/负/平**: 23/23/4 (完全打平!)
- **差异范围**: [-2.33, +1.30] dB
- **标准差**: 0.83 dB

### 4.2 按 PSNR 区间分析

| 区间 | 帧数 | GAR 平均差异 | 分析 |
|------|------|--------------|------|
| 低 PSNR (<32dB) | 24 | **-0.63 dB** | GAR 在困难区域反而更差！ |
| 高 PSNR (>40dB) | 4 | +0.35 dB | 少数高 PSNR 帧略有改善 |

### 4.3 对比 Good Case

| 特征 | Foot 3views (Good) | Foot 6views (Bad) |
|------|-------------------|-------------------|
| 低 PSNR 区域效果 | **+0.34** | **-0.63** |
| 高 PSNR 区域效果 | -1.11 | +0.35 |
| 低 PSNR 帧占比 | 86% (43/50) | 48% (24/50) |

### 4.4 结论

Foot 6views 失败的原因:
1. **6 views 信息已足够**，GAR 的几何约束成为干扰
2. 低 PSNR 区域**减少** (48% vs 86%)，GAR 优势场景减少
3. 在困难区域 GAR 反而**恶化**了重建质量

---

## 5. 根本原因总结

### 5.1 代码问题 (已修复)

**问题**: ProximityGuidedDensifier 完全没有被调用！

```
train.py 中只启用了:
✓ Binocular Consistency Loss (微弱效果)
✗ Proximity-Guided Densification (完全缺失!)
```

**修复**: 已在 train.py 中集成 ProximityGuidedDensifier

### 5.2 算法问题

即使只有 Binocular Loss，也暴露了 GAR 的局限:

1. **视角数敏感**: 3views 有效，6/9views 无效或有害
2. **PSNR 区间敏感**: 低 PSNR 有帮助，高 PSNR 有害
3. **高方差**: 同一配置下帧间差异 >3dB

### 5.3 建议

1. **Proximity Densification 集成后重跑实验**
2. **调整超参数**:
   - `proximity_threshold`: 当前 5.0 可能需要针对 CT 数据调整
   - `proximity_start_iter`: 当前 1000，可能太早
3. **视角自适应**: 考虑根据视角数量调整 GAR 策略
   - 3views: 激进 GAR
   - 6/9views: 保守或禁用 GAR

---

## 6. 代码修改清单

### 已完成

1. ✅ `train.py`: 集成 ProximityGuidedDensifier
2. ✅ `r2_gaussian/arguments/__init__.py`: 添加时间参数
3. ✅ `r2_gaussian/innovations/fsgs/__init__.py`: 移除 Medical Constraints
4. ✅ `r2_gaussian/innovations/fsgs/medical_constraints.py`: 删除
5. ✅ `cc-agent/scripts/run_spags_ablation.sh`: 移除 Medical Constraints 参数

### 待验证

- 运行新的 GAR 实验验证 Proximity Densification 效果

---

*报告更新时间: 2025-12-06*
