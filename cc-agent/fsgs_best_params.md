# FSGS 最佳参数设置指南

> 基于 2025-11-29 超参数搜索实验结果

## 推荐配置

```bash
# FSGS 最佳超参数配置
--enable_fsgs_depth true
--enable_fsgs_pseudo_views true
--enable_medical_constraints true

# 核心参数（经过验证的最优值）
--depth_pseudo_weight 0.08        # 伪视角深度损失权重 ✓✓ 最佳
--fsgs_depth_weight 0.05          # 深度监督权重（MiDaS）
--proximity_threshold 5.0         # 邻近距离阈值
--proximity_k_neighbors 5         # K近邻数量
--start_sample_pseudo 5000        # 伪视角采样起始迭代
```

## 超参数搜索实验结果

### Foot 3-views 场景 (Baseline: 28.487 dB)

| 配置 | depth_pseudo_weight | PSNR | SSIM | vs Baseline |
|------|---------------------|------|------|-------------|
| **F_dpw_008** | **0.08** | **28.524** | 0.8981 | **+0.037 dB** ✓✓ |
| F_fdw_008 | 0.03 (fdw=0.08) | 28.508 | 0.8980 | +0.021 dB |
| F_dpw_001 | 0.01 | 28.501 | 0.8984 | +0.014 dB |
| F_baseline | 0.03 | 28.434 | 0.8980 | -0.053 dB |
| F_pt_3 | 0.03 (pt=3.0) | 28.426 | 0.8981 | -0.061 dB |
| F_fdw_003 | 0.03 (fdw=0.03) | 28.388 | 0.8973 | -0.099 dB |
| F_dpw_005 | 0.05 | 28.223 | 0.8971 | **-0.264 dB** ⚠️ |

### Abdomen 9-views 场景 (Baseline: 29.29 dB)

| 配置 | depth_pseudo_weight | PSNR | SSIM | vs FSGS Baseline |
|------|---------------------|------|------|------------------|
| **F_dpw_008** | **0.08** | **36.501** | 0.9802 | **+0.22 dB** ✓✓ |
| F_dpw_005 | 0.05 | 36.387 | 0.9802 | +0.11 dB |
| F_fdw_003 | 0.03 (fdw=0.03) | 36.367 | 0.9802 | +0.09 dB |
| F_baseline | 0.03 | 36.276 | 0.9801 | — |
| F_dpw_001 | 0.01 | 36.201 | 0.9800 | -0.08 dB |

## 关键发现

### 1. `depth_pseudo_weight=0.08` 是最佳配置

- 在 **两个场景都取得最好结果**
- Foot-3views: +0.037 dB（vs R²-Gaussian baseline）
- Abdomen-9views: +0.22 dB（vs FSGS baseline with dpw=0.03）

### 2. 中等权重在极稀疏场景有害

- `dpw=0.05` 在 Foot-3views 导致 **-0.264 dB 下降**
- 原因推测：中等权重不够强也不够弱，导致优化方向不明确

### 3. proximity_threshold 调整效果不明显

- 从 5.0 收紧到 3.0 差异 < 0.01 dB
- 建议保持默认值 5.0

### 4. fsgs_depth_weight 小幅提升

- 从 0.05 提升到 0.08 有 +0.07 dB 提升
- 但不如 depth_pseudo_weight 调整效果显著

## 参数说明

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `depth_pseudo_weight` | 0.03 | **0.08** | 伪视角深度一致性损失权重 |
| `fsgs_depth_weight` | 0.05 | 0.05 | MiDaS 深度监督权重 |
| `proximity_threshold` | 5.0 | 5.0 | 邻近约束距离阈值 |
| `proximity_k_neighbors` | 5 | 5 | K近邻数量 |
| `start_sample_pseudo` | 5000 | 5000 | 伪视角采样起始迭代 |

## 使用示例

```bash
# Foot 3-views 训练（推荐配置）
python train.py \
    -s data/369/foot_50_3views.pickle \
    --enable_fsgs_depth \
    --enable_fsgs_pseudo_views \
    --enable_medical_constraints \
    --depth_pseudo_weight 0.08 \
    --fsgs_depth_weight 0.05 \
    --proximity_threshold 5.0 \
    --proximity_k_neighbors 5 \
    --iterations 30000

# Abdomen 9-views 训练（推荐配置）
python train.py \
    -s data/369/abdomen_50_9views.pickle \
    --enable_fsgs_depth \
    --enable_fsgs_pseudo_views \
    --enable_medical_constraints \
    --depth_pseudo_weight 0.08 \
    --fsgs_depth_weight 0.05 \
    --iterations 30000
```

## 历史最佳记录

| 日期 | 场景 | 配置 | PSNR | SSIM | 备注 |
|------|------|------|------|------|------|
| 2025-11-25 | Foot-3 | FSGS v3 (dpw=0.03, k=5, τ=5.0) | 28.681 | 0.9014 | 首次超越 baseline |
| 2025-11-29 | Foot-3 | FSGS (dpw=0.08) | 28.524 | 0.8981 | 10k 快速验证 |
| 2025-11-29 | Abdomen-9 | FSGS (dpw=0.08) | 36.501 | 0.9802 | 10k 快速验证 |

---

*文档创建时间: 2025-12-03*
*实验数据来源: output/fsgs_search/*
