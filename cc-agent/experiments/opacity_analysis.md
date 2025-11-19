# Opacity 分析报告

## Baseline 统计

- Gaussian 数量: 61,514
- Opacity 均值: 0.045556
- 高 Opacity (>0.5) 比例: 0.18%
- 超高 Opacity (>0.8) 比例: 0.00%

## DropGaussian 统计

- Gaussian 数量: 67,310
- Opacity 均值: 0.025298
- 高 Opacity (>0.5) 比例: 0.00%
- 超高 Opacity (>0.8) 比例: 0.00%

## 对比

| 指标 | Baseline | DropGaussian | 变化 |
|------|----------|--------------|------|
| Gaussian 数量 | 61,514 | 67,310 | +5,796 (+9.42%) |
| Opacity 均值 | 0.045556 | 0.025298 | -0.020258 (-44.47%) |
| 高 Opacity (>0.5) | 0.18% | 0.00% | -0.18 pp |
| 超高 Opacity (>0.8) | 0.00% | 0.00% | -0.00 pp |

## 结论

DropGaussian 的平均 opacity 比 Baseline 低 44.47%，确实存在 opacity 下降。
