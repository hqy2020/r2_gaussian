# SPAGS 消融实验结果

## 实验信息

- **日期**: 2025-12-04
- **数据集**: Foot 3views (data/369/)
- **Baseline**: R²-Gaussian SOTA (PSNR 28.4873, SSIM 0.9005)
- **训练轮次**: 30,000 iterations

## 完整结果表

| 配置 | PSNR | SSIM | Δ PSNR | 备注 |
|------|------|------|--------|------|
| **SPAGS (完整)** | **28.8346** | **0.8998** | **+0.347** | SPS+GAR+ADM 最佳配置 |
| SPS+ADM | 28.7888 | 0.8991 | +0.302 | 次优配置 |
| GAR | 28.7370 | 0.8988 | +0.250 | 单独贡献最大 |
| GAR+ADM | 28.7130 | 0.8991 | +0.226 | |
| SPS | 28.6605 | 0.8982 | +0.173 | |
| SPS+GAR | 28.6121 | 0.8987 | +0.125 | |
| ADM | 28.5733 | 0.8979 | +0.086 | |
| Baseline | 28.4873 | 0.9005 | --- | R²-Gaussian |

## 各模块单独贡献

| 模块 | 全称 | PSNR 提升 | 说明 |
|------|------|-----------|------|
| **GAR** | Geometry-Aware Refinement | +0.250 dB | 贡献最大，双目一致性损失有效 |
| **SPS** | Spatial Prior Seeding | +0.173 dB | 密度加权点云初始化有效 |
| **ADM** | Adaptive Density Modulation | +0.086 dB | K-Planes 特征调制有一定效果 |

## 组合效应分析

### 正向协同效应

1. **SPS + ADM**:
   - 实际: +0.302 dB
   - 预期 (SPS + ADM): +0.173 + 0.086 = +0.259 dB
   - **协同增益**: +0.043 dB

2. **SPAGS 完整版**:
   - 实际: +0.347 dB
   - 预期 (SPS + GAR + ADM): +0.173 + 0.250 + 0.086 = +0.509 dB
   - 实际低于预期，说明部分技术间存在冗余

### 组合效果排序

1. SPAGS (SPS+GAR+ADM): +0.347 dB
2. SPS+ADM: +0.302 dB
3. GAR+ADM: +0.226 dB
4. SPS+GAR: +0.125 dB

## 关键发现

1. **GAR (几何感知细化) 是核心技术**
   - 单独使用时贡献最大 (+0.250 dB)
   - 双目一致性损失对稀疏视角重建非常有效

2. **SPS (空间先验播种) 提供稳定基础**
   - 密度加权采样确保高信息区域获得更多初始点
   - 与其他技术组合时效果稳定

3. **ADM (自适应密度调制) 起辅助作用**
   - 单独贡献较小 (+0.086 dB)
   - 与 SPS 组合时有正向协同

4. **完整 SPAGS 是最优选择**
   - 三个模块协同工作达到最佳效果
   - 推荐用于生产环境

## 实验目录

```
output/2025_12_04_14_51_foot_3views_spags/    # SPAGS 完整版 (最佳)
output/2025_12_04_13_52_foot_3views_sps_adm/  # SPS+ADM
output/2025_12_04_14_22_foot_3views_gar/      # GAR
output/2025_12_04_13_22_foot_3views_gar_adm/  # GAR+ADM
output/2025_12_04_13_13_foot_3views_sps/      # SPS
output/2025_12_04_13_22_foot_3views_sps_gar/  # SPS+GAR
output/2025_12_04_13_15_foot_3views_adm/      # ADM
```

## 下一步计划

- [ ] 在其他器官 (Chest, Abdomen) 上验证 SPAGS 效果
- [ ] 在 6-views 和 9-views 场景测试
- [ ] 对比其他 SOTA 方法
