# Good/Fail Cases 详细分析

## Good Cases - DropGaussian 表现更好

| 图片编号 | PSNR 提升 | Baseline PSNR | DropGaussian PSNR | 分析 |
|---------|-----------|---------------|-------------------|------|
| #46 | +2.557 dB | - | - | - |
| #33 | +2.176 dB | - | - | - |
| #32 | +2.132 dB | - | - | - |
| #45 | +1.852 dB | - | - | - |
| #34 | +1.719 dB | - | - | - |

## Fail Cases - DropGaussian 表现更差

| 图片编号 | PSNR 下降 | Baseline PSNR | DropGaussian PSNR | 分析 |
|---------|-----------|---------------|-------------------|------|
| #26 | -2.226 dB | - | - | - |
| #17 | -1.785 dB | - | - | - |
| #18 | -1.764 dB | - | - | - |
| #3 | -1.760 dB | - | - | - |
| #2 | -1.725 dB | - | - | - |

## 关键观察

### Good Cases 特征
- 这些图片在 DropGaussian 下表现更好
- PSNR 提升范围：+1.719 dB 到 +2.557 dB
- 需要分析这些图片的共同特征（如密度、对比度、结构复杂度等）

### Fail Cases 特征
- 这些图片在 DropGaussian 下表现更差
- PSNR 下降范围：-1.725 dB 到 -2.226 dB
- 需要分析这些图片的共同特征

### 数据支持的结论

1. **Opacity 大幅下降**：
   - Baseline 平均 opacity: 0.046
   - DropGaussian 平均 opacity: 0.025
   - 下降幅度: **44.47%**

2. **高质量 Gaussian 急剧减少**：
   - Baseline 高 opacity (>0.5): 112 个 (0.18%)
   - DropGaussian 高 opacity (>0.5): 3 个 (0.00%)
   - 减少幅度: **97.3%**

3. **整体性能对比**：
   - Good Cases: 13/50 (26%)
   - Fail Cases: 37/50 (74%)
   - 平均 PSNR 下降: 0.426 dB

