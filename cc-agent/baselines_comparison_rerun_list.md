# 建议重跑列表（model + organ + view）

基于 `cc-agent/baselines_comparison_psnr2d_ssim2d.md` 的数值自洽性检查：

- **可用（无需重跑）**：`xgaussian/xgs`、`r2gaussian/r2gs`（随视角增加整体单调提升，数值区间正常）
- **建议重跑**：`tensorf`、`naf`、`saxnerf`（出现负 PSNR/极低 SSIM 或随视角趋势异常）

## 建议重跑（45 组）

| model | organ | view |
|---|---|---|
| tensorf | foot | 3 |
| tensorf | foot | 6 |
| tensorf | foot | 9 |
| tensorf | chest | 3 |
| tensorf | chest | 6 |
| tensorf | chest | 9 |
| tensorf | head | 3 |
| tensorf | head | 6 |
| tensorf | head | 9 |
| tensorf | abdomen | 3 |
| tensorf | abdomen | 6 |
| tensorf | abdomen | 9 |
| tensorf | pancreas | 3 |
| tensorf | pancreas | 6 |
| tensorf | pancreas | 9 |
| naf | foot | 3 |
| naf | foot | 6 |
| naf | foot | 9 |
| naf | chest | 3 |
| naf | chest | 6 |
| naf | chest | 9 |
| naf | head | 3 |
| naf | head | 6 |
| naf | head | 9 |
| naf | abdomen | 3 |
| naf | abdomen | 6 |
| naf | abdomen | 9 |
| naf | pancreas | 3 |
| naf | pancreas | 6 |
| naf | pancreas | 9 |
| saxnerf | foot | 3 |
| saxnerf | foot | 6 |
| saxnerf | foot | 9 |
| saxnerf | chest | 3 |
| saxnerf | chest | 6 |
| saxnerf | chest | 9 |
| saxnerf | head | 3 |
| saxnerf | head | 6 |
| saxnerf | head | 9 |
| saxnerf | abdomen | 3 |
| saxnerf | abdomen | 6 |
| saxnerf | abdomen | 9 |
| saxnerf | pancreas | 3 |
| saxnerf | pancreas | 6 |
| saxnerf | pancreas | 9 |

## 便于直接跑的命令模板

把 `<GPU>` 换成你的 GPU id：

```bash
for model in tensorf naf saxnerf; do
  for organ in foot chest head abdomen pancreas; do
    for view in 3 6 9; do
      echo ./cc-agent/scripts/run_spags_ablation.sh ${model} ${organ} ${view} <GPU>
    done
  done
done
```

