#!/usr/bin/env python3
"""
分析 DropGaussian vs Baseline 在 Foot-3 测试集上的逐图对比
找出 Good Cases 和 Fail Cases
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
with open('/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_19_foot_3views_dropgaussian/eval/iter_030000/eval2d_render_test.yml', 'r') as f:
    drop_data = yaml.safe_load(f)

with open('/home/qyhu/Documents/r2_ours/r2_gaussian/output/foot_3views_r2_baseline_1113/eval/iter_030000/eval2d_render_test.yml', 'r') as f:
    baseline_data = yaml.safe_load(f)

# 提取数据
drop_psnrs = np.array(drop_data['psnr_2d_projs'])
baseline_psnrs = np.array(baseline_data['psnr_2d_projs'])
drop_ssims = np.array(drop_data['ssim_2d_projs'])
baseline_ssims = np.array(baseline_data['ssim_2d_projs'])

# 计算差异
psnr_diff = drop_psnrs - baseline_psnrs
ssim_diff = drop_ssims - baseline_ssims

# 整体统计
print("="*80)
print("📊 整体指标对比")
print("="*80)
print(f"Baseline 平均 PSNR: {baseline_data['psnr_2d']:.4f}")
print(f"DropGaussian 平均 PSNR: {drop_data['psnr_2d']:.4f}")
print(f"PSNR 差异: {drop_data['psnr_2d'] - baseline_data['psnr_2d']:.4f} dB")
print()
print(f"Baseline 平均 SSIM: {baseline_data['ssim_2d']:.4f}")
print(f"DropGaussian 平均 SSIM: {drop_data['ssim_2d']:.4f}")
print(f"SSIM 差异: {drop_data['ssim_2d'] - baseline_data['ssim_2d']:.4f}")
print()

# Good Cases: DropGaussian 表现优于 Baseline
good_mask = psnr_diff > 0
good_indices = np.where(good_mask)[0]
good_improvements = psnr_diff[good_mask]

print("="*80)
print(f"✅ Good Cases ({len(good_indices)}/{len(psnr_diff)}): DropGaussian 优于 Baseline")
print("="*80)
for idx, improvement in zip(good_indices, good_improvements):
    print(f"图片 #{idx:2d}: PSNR Δ = +{improvement:+.3f} dB, "
          f"SSIM Δ = {ssim_diff[idx]:+.4f} "
          f"(Baseline: {baseline_psnrs[idx]:.2f} → Drop: {drop_psnrs[idx]:.2f})")
print()

# Fail Cases: DropGaussian 表现劣于 Baseline
fail_mask = psnr_diff < 0
fail_indices = np.where(fail_mask)[0]
fail_degradations = psnr_diff[fail_mask]

print("="*80)
print(f"❌ Fail Cases ({len(fail_indices)}/{len(psnr_diff)}): DropGaussian 劣于 Baseline")
print("="*80)
# 按性能下降排序（最差的在前）
sorted_fail_idx = fail_indices[np.argsort(fail_degradations)]
sorted_fail_deg = fail_degradations[np.argsort(fail_degradations)]

for idx, degradation in zip(sorted_fail_idx, sorted_fail_deg):
    print(f"图片 #{idx:2d}: PSNR Δ = {degradation:+.3f} dB, "
          f"SSIM Δ = {ssim_diff[idx]:+.4f} "
          f"(Baseline: {baseline_psnrs[idx]:.2f} → Drop: {drop_psnrs[idx]:.2f})")
print()

# 极端案例分析
print("="*80)
print("🔍 极端案例分析")
print("="*80)

# Top 5 最大提升
if len(good_indices) > 0:
    top_improvements_idx = good_indices[np.argsort(good_improvements)[::-1]][:5]
    print("📈 Top 5 最大提升:")
    for rank, idx in enumerate(top_improvements_idx, 1):
        print(f"  {rank}. 图片 #{idx}: +{psnr_diff[idx]:.3f} dB "
              f"(Baseline: {baseline_psnrs[idx]:.2f} → Drop: {drop_psnrs[idx]:.2f})")
    print()

# Top 5 最大下降
if len(fail_indices) > 0:
    top_degradations_idx = sorted_fail_idx[:5]
    print("📉 Top 5 最大下降:")
    for rank, idx in enumerate(top_degradations_idx, 1):
        print(f"  {rank}. 图片 #{idx}: {psnr_diff[idx]:.3f} dB "
              f"(Baseline: {baseline_psnrs[idx]:.2f} → Drop: {drop_psnrs[idx]:.2f})")
    print()

# 统计分析
print("="*80)
print("📐 统计分析")
print("="*80)
print(f"PSNR 差异均值: {psnr_diff.mean():.4f} dB")
print(f"PSNR 差异标准差: {psnr_diff.std():.4f} dB")
print(f"PSNR 差异中位数: {np.median(psnr_diff):.4f} dB")
print(f"PSNR 差异范围: [{psnr_diff.min():.3f}, {psnr_diff.max():.3f}] dB")
print()
print(f"SSIM 差异均值: {ssim_diff.mean():.4f}")
print(f"SSIM 差异标准差: {ssim_diff.std():.4f}")
print(f"SSIM 差异中位数: {np.median(ssim_diff):.4f}")
print()

# 保存结果到文件
output_report = f"""# DropGaussian vs Baseline 逐图对比分析报告

## 整体指标

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| Baseline | {baseline_data['psnr_2d']:.4f} | {baseline_data['ssim_2d']:.4f} |
| DropGaussian | {drop_data['psnr_2d']:.4f} | {drop_data['ssim_2d']:.4f} |
| **差异** | **{drop_data['psnr_2d'] - baseline_data['psnr_2d']:+.4f}** | **{drop_data['ssim_2d'] - baseline_data['ssim_2d']:+.4f}** |

## Good Cases 分析

**定义**: DropGaussian PSNR > Baseline PSNR

- 数量: {len(good_indices)}/{len(psnr_diff)} ({100*len(good_indices)/len(psnr_diff):.1f}%)
- 平均提升: {psnr_diff[good_mask].mean():.4f} dB
- 最大提升: {psnr_diff.max():.4f} dB (图片 #{np.argmax(psnr_diff)})

### Top 5 最佳案例

| 排名 | 图片编号 | Baseline PSNR | Drop PSNR | 提升 (dB) | Baseline SSIM | Drop SSIM | SSIM 差异 |
|------|----------|---------------|-----------|-----------|---------------|-----------|-----------|
"""
if len(good_indices) > 0:
    for rank, idx in enumerate(top_improvements_idx, 1):
        output_report += f"| {rank} | #{idx} | {baseline_psnrs[idx]:.2f} | {drop_psnrs[idx]:.2f} | +{psnr_diff[idx]:.3f} | {baseline_ssims[idx]:.4f} | {drop_ssims[idx]:.4f} | {ssim_diff[idx]:+.4f} |\n"

output_report += f"""
## Fail Cases 分析

**定义**: DropGaussian PSNR < Baseline PSNR

- 数量: {len(fail_indices)}/{len(psnr_diff)} ({100*len(fail_indices)/len(psnr_diff):.1f}%)
- 平均下降: {psnr_diff[fail_mask].mean():.4f} dB
- 最大下降: {psnr_diff.min():.4f} dB (图片 #{np.argmin(psnr_diff)})

### Top 5 最差案例

| 排名 | 图片编号 | Baseline PSNR | Drop PSNR | 下降 (dB) | Baseline SSIM | Drop SSIM | SSIM 差异 |
|------|----------|---------------|-----------|-----------|---------------|-----------|-----------|
"""
if len(fail_indices) > 0:
    for rank, idx in enumerate(top_degradations_idx, 1):
        output_report += f"| {rank} | #{idx} | {baseline_psnrs[idx]:.2f} | {drop_psnrs[idx]:.2f} | {psnr_diff[idx]:.3f} | {baseline_ssims[idx]:.4f} | {drop_ssims[idx]:.4f} | {ssim_diff[idx]:+.4f} |\n"

output_report += f"""
## 统计总结

- **PSNR 差异统计**:
  - 均值: {psnr_diff.mean():.4f} dB
  - 标准差: {psnr_diff.std():.4f} dB
  - 中位数: {np.median(psnr_diff):.4f} dB
  - 范围: [{psnr_diff.min():.3f}, {psnr_diff.max():.3f}] dB

- **SSIM 差异统计**:
  - 均值: {ssim_diff.mean():.4f}
  - 标准差: {ssim_diff.std():.4f}
  - 中位数: {np.median(ssim_diff):.4f}

## 观察结论

1. **整体表现**: DropGaussian PSNR 平均下降 {abs(drop_data['psnr_2d'] - baseline_data['psnr_2d']):.4f} dB
2. **案例分布**: {len(good_indices)} 个提升案例 vs {len(fail_indices)} 个下降案例
3. **性能差异**: 最大提升 {psnr_diff.max():.3f} dB，最大下降 {psnr_diff.min():.3f} dB
4. **方差**: PSNR 差异标准差为 {psnr_diff.std():.4f} dB，说明不同测试图片表现差异{'较大' if psnr_diff.std() > 1.0 else '适中'}
"""

with open('/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/test_cases_comparison.md', 'w', encoding='utf-8') as f:
    f.write(output_report)

print("✅ 分析报告已保存到: cc-agent/experiments/test_cases_comparison.md")
