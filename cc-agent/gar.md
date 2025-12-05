# GAR (Geometry-Aware Refinement) 超参数文档

## 概述

GAR 是 SPAGS 框架的 Stage 2 技术，包含两个子模块：
1. **双目一致性损失 (Binocular Consistency)**: 利用虚拟视角 warp 进行自监督
2. **邻近密化 (Proximity-Guided Densification)**: 在稀疏区域自适应增加高斯点

---

## 最优超参数配置

### 核心参数

| 参数 | 最优值 | 原版值 | 说明 |
|------|--------|--------|------|
| `proximity_threshold` | **5** | 10.0 | 邻近度阈值，越小越严格 |
| `proximity_k_neighbors` | **5** | 3 | K 最近邻数量 |
| `binocular_loss_weight` | **0.08** | 0.15 | 双目一致性损失权重 |
| `binocular_max_angle_offset` | **0.04** | 0.06 | 虚拟视角最大角度偏移 (rad) |
| `binocular_start_iter` | **5000** | 7000 | 开始应用双目损失的迭代 |
| `binocular_warmup_iters` | **3000** | - | 损失权重 warmup 迭代数 |
| `enable_medical_constraints` | **True** | - | 启用医学约束 |

### 命令行示例

```bash
python train.py \
    -s data/369/foot_50_3views.pickle \
    -m output/gar_experiment \
    --enable_binocular_consistency \
    --binocular_loss_weight 0.08 \
    --binocular_max_angle_offset 0.04 \
    --binocular_start_iter 5000 \
    --binocular_warmup_iters 3000 \
    --enable_fsgs_proximity \
    --proximity_threshold 5 \
    --proximity_k_neighbors 5 \
    --enable_medical_constraints
```

---

## proximity_threshold 消融实验

**实验日期**: 2025-12-04
**数据集**: Foot 3 views
**点云**: 3k SPS 密度加权采样

| proximity_threshold | PSNR (dB) | SSIM | 相比 Baseline |
|---------------------|-----------|------|---------------|
| 1 | 29.677 | 0.9052 | +1.19 dB |
| 3 | 29.470 | 0.9049 | +0.98 dB |
| **5** | **29.669** | **0.9056** | **+1.18 dB** |
| 7 | 29.569 | 0.9054 | +1.08 dB |
| 10 | 29.608 | 0.9053 | +1.12 dB |

**Baseline**: PSNR 28.4873, SSIM 0.9005

### 结论

- **选择 threshold=5** 作为最优配置
  - PSNR 29.669 dB，SSIM 0.9056
  - 相比 threshold=1 (PSNR 略高 0.008 dB)，threshold=5 的 SSIM 更高 (0.9056 vs 0.9052)
  - 综合 PSNR/SSIM 表现更均衡

- 所有阈值配置都大幅超越 baseline (+0.98~1.19 dB)
- 较小阈值 (1, 5) 整体优于较大阈值 (7, 10)

---

## 医学约束 (Medical Constraints)

当启用 `enable_medical_constraints=True` 时，系统根据 opacity 值自动分类组织类型，应用差异化的邻近密化参数：

| 组织类型 | Opacity 范围 | proximity_threshold | k_neighbors |
|----------|--------------|---------------------|-------------|
| Background Air | [0.0, 0.05) | 2.0 (最严格) | 6 |
| Tissue Transition | [0.05, 0.15) | 1.5 | 8 |
| Soft Tissue | [0.15, 0.40) | 1.0 | 6 |
| Dense Structures | [0.40, 1.0] | 0.8 (最宽松) | - |

**增益**: 约 +0.24 dB (28.50 vs 28.26 dB)

---

## 代码位置

- **双目一致性**: `r2_gaussian/utils/binocular_utils.py`
- **邻近密化**: `r2_gaussian/innovations/fsgs/proximity_densifier.py`
- **医学约束**: `r2_gaussian/innovations/fsgs/medical_constraints.py`
- **参数定义**: `r2_gaussian/arguments/__init__.py` (行 158-178, 75-91)
- **训练集成**: `train.py` (行 96-106, 229-261)

---

## 消融实验脚本

```bash
# 运行 proximity_threshold 消融实验
./scripts/ablation_gar_proximity_threshold.sh

# 运行指定阈值
./scripts/ablation_gar_proximity_threshold.sh 1 5 10
```
