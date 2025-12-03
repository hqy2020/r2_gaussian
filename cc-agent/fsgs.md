# FSGS 超参数实验记录

> 实验日期: 2025-11-29
> 实验类型: 10k 迭代快速验证

## 最优超参配置

```bash
--enable_fsgs_depth true
--enable_fsgs_pseudo_views true
--enable_medical_constraints true
--depth_pseudo_weight 0.08        # 核心参数，两个场景都最优
--fsgs_depth_weight 0.05          # 保持默认
--proximity_threshold 5.0         # 保持默认
--proximity_k_neighbors 5         # 保持默认
--start_sample_pseudo 5000        # 保持默认
```

---

## 实验结果汇总

### Foot 3-views（7 个实验完成）

**R²-Gaussian Baseline**: PSNR **28.487**, SSIM **0.9005**

| 配置 | PSNR | SSIM | vs Baseline |
|------|------|------|-------------|
| **F_dpw_008** | **28.524** | 0.8981 | ✅ +0.037 / ❌ -0.0024 |
| F_fdw_008 | 28.508 | 0.8980 | ✅ +0.021 / ❌ -0.0025 |
| F_dpw_001 | 28.501 | 0.8984 | ✅ +0.014 / ❌ -0.0021 |
| F_baseline (dpw=0.03) | 28.434 | 0.8980 | ❌ -0.053 / ❌ -0.0025 |
| F_pt_3 | 28.426 | 0.8981 | ❌ -0.061 / ❌ -0.0024 |
| F_fdw_003 | 28.388 | 0.8973 | ❌ -0.099 / ❌ -0.0032 |
| F_dpw_005 | 28.223 | 0.8971 | ❌ -0.264 / ❌ -0.0034 |

**结论**: ⚠️ **不稳定** - PSNR 3/7 超过，SSIM 0/7 超过

---

### Abdomen 9-views（5 个实验完成）

**R²-Gaussian Baseline**: PSNR **29.29**, SSIM **0.9366**

| 配置 | PSNR | SSIM | vs Baseline |
|------|------|------|-------------|
| **F_dpw_008** | **36.501** | **0.9802** | ✅ **+7.21** / ✅ **+0.044** |
| F_dpw_005 | 36.387 | 0.9802 | ✅ +7.10 / ✅ +0.044 |
| F_fdw_003 | 36.367 | 0.9802 | ✅ +7.08 / ✅ +0.044 |
| F_baseline (dpw=0.03) | 36.276 | 0.9801 | ✅ +6.99 / ✅ +0.044 |
| F_dpw_001 | 36.201 | 0.9800 | ✅ +6.91 / ✅ +0.043 |

**结论**: ✅ **非常稳定** - 全部大幅超越 baseline

---

## 关键发现

### 1. 稳定性结论

| 场景 | 稳定性 | 说明 |
|------|--------|------|
| **Foot-3** | ❌ 不稳定 | 仅部分配置超过 baseline，SSIM 全部下降 |
| **Abdomen-9** | ✅ 稳定 | 所有配置都大幅超越 baseline (+7 dB) |

**总结**: ❌ **不能得出 FSGS "稳定超过 baseline" 的结论**

### 2. 超参数敏感性

- `depth_pseudo_weight=0.08` 在两个场景都是最优
- `depth_pseudo_weight=0.05` 在 Foot-3 场景有害 (-0.264 dB)
- `proximity_threshold` 调整效果不明显

### 3. 场景差异分析

| 因素 | Foot-3 (极稀疏) | Abdomen-9 (中等稀疏) |
|------|-----------------|---------------------|
| 视角数量 | 3 | 9 |
| 信息充足度 | 不足 | 相对充足 |
| 深度先验可靠性 | 低 | 高 |
| FSGS 效果 | 不稳定 | 显著 |

---

## 参数说明

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `depth_pseudo_weight` | 0.03 | **0.08** | 伪视角深度一致性损失权重 |
| `fsgs_depth_weight` | 0.05 | 0.05 | MiDaS 深度监督权重 |
| `proximity_threshold` | 5.0 | 5.0 | 邻近约束距离阈值 |
| `proximity_k_neighbors` | 5 | 5 | K近邻数量 |
| `start_sample_pseudo` | 5000 | 5000 | 伪视角采样起始迭代 |

---

## 实验目录

```
output/fsgs_search/
├── 2025_11_29_00_11_foot_3views_F_baseline/
├── 2025_11_29_00_25_foot_3views_F_dpw_001/
├── 2025_11_29_00_49_foot_3views_F_dpw_005/
├── 2025_11_29_01_00_foot_3views_F_dpw_008/      # Foot 最优
├── 2025_11_29_01_10_foot_3views_F_fdw_003/
├── 2025_11_29_01_20_foot_3views_F_fdw_008/
├── 2025_11_29_01_31_foot_3views_F_pt_3/
├── 2025_11_29_00_11_abdomen_9views_F_baseline/
├── 2025_11_29_00_34_abdomen_9views_F_dpw_001/
├── 2025_11_29_00_48_abdomen_9views_F_dpw_005/
├── 2025_11_29_00_59_abdomen_9views_F_dpw_008/   # Abdomen 最优
└── 2025_11_29_01_15_abdomen_9views_F_fdw_003/
```

---

## 后续建议

1. **3 views 场景**: FSGS 单独使用效果不稳定，建议结合其他技术（如 Init-PCD、Bino）
2. **9 views 场景**: FSGS 效果显著，可作为主要增强技术
3. **完整训练验证**: 当前为 10k 快速验证，关键配置需 30k 完整训练确认

---

*文档创建时间: 2025-12-03*
*数据来源: output/fsgs_search/*
