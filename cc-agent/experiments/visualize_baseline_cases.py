#!/usr/bin/env python3
"""
可视化 Baseline 的 Good/Fail Cases
基于 PSNR 数据分析哪些图片表现好/差
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 关键案例（基于之前的分析）
top_good_cases = [46, 33, 32, 45, 34]  # DropGaussian 提升最大
top_fail_cases = [26, 17, 18, 3, 2]    # DropGaussian 下降最大

# 数据（从之前的分析中）
psnr_improvements = {
    46: +2.557,
    33: +2.176,
    32: +2.132,
    45: +1.852,
    34: +1.719,
}

psnr_degradations = {
    26: -2.226,
    17: -1.785,
    18: -1.764,
    3: -1.760,
    2: -1.725,
}

baseline_dir = "/home/qyhu/Documents/r2_ours/r2_gaussian/output/foot_3views_r2_baseline_1113/eval/iter_030000/render_images"

def load_image(idx, img_type='render'):
    """加载指定索引的图片"""
    filename = f"{idx:04d}_{img_type}.png"
    filepath = os.path.join(baseline_dir, filename)
    if os.path.exists(filepath):
        return np.array(Image.open(filepath))
    return None

# 创建可视化
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 5, hspace=0.3, wspace=0.2)

# Good Cases
print("="*80)
print("📈 Good Cases - DropGaussian 表现优于 Baseline 的案例")
print("="*80)
for i, idx in enumerate(top_good_cases):
    # 加载图片
    gt = load_image(idx, 'gt')
    render = load_image(idx, 'render')
    diff = load_image(idx, 'diff')

    if gt is not None and render is not None:
        ax = fig.add_subplot(gs[0, i])

        # 显示 GT 和 Render 的拼接
        if diff is not None:
            combined = np.hstack([gt, render, diff])
            ax.imshow(combined, cmap='gray')
        else:
            combined = np.hstack([gt, render])
            ax.imshow(combined, cmap='gray')

        improvement = psnr_improvements[idx]
        ax.set_title(f"图片 #{idx}\nΔPSNR = +{improvement:.3f} dB", fontsize=10)
        ax.axis('off')

        print(f"图片 #{idx}: PSNR 提升 +{improvement:.3f} dB")
    else:
        print(f"⚠️  图片 #{idx}: 图片文件不存在")

# Fail Cases
print("\n" + "="*80)
print("📉 Fail Cases - DropGaussian 表现劣于 Baseline 的案例")
print("="*80)
for i, idx in enumerate(top_fail_cases):
    # 加载图片
    gt = load_image(idx, 'gt')
    render = load_image(idx, 'render')
    diff = load_image(idx, 'diff')

    if gt is not None and render is not None:
        ax = fig.add_subplot(gs[1, i])

        # 显示 GT 和 Render 的拼接
        if diff is not None:
            combined = np.hstack([gt, render, diff])
            ax.imshow(combined, cmap='gray')
        else:
            combined = np.hstack([gt, render])
            ax.imshow(combined, cmap='gray')

        degradation = psnr_degradations[idx]
        ax.set_title(f"图片 #{idx}\nΔPSNR = {degradation:.3f} dB", fontsize=10, color='red')
        ax.axis('off')

        print(f"图片 #{idx}: PSNR 下降 {degradation:.3f} dB")
    else:
        print(f"⚠️  图片 #{idx}: 图片文件不存在")

plt.suptitle("DropGaussian vs Baseline: Good/Fail Cases 对比\n" +
             "每行显示：Ground Truth | Baseline Render | Diff",
             fontsize=14, fontweight='bold')

output_path = "/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/good_fail_cases_baseline.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化已保存到: {output_path}")

# 创建详细报告
report_path = "/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/good_fail_cases_analysis.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Good/Fail Cases 详细分析\n\n")

    f.write("## Good Cases - DropGaussian 表现更好\n\n")
    f.write("| 图片编号 | PSNR 提升 | Baseline PSNR | DropGaussian PSNR | 分析 |\n")
    f.write("|---------|-----------|---------------|-------------------|------|\n")
    for idx in top_good_cases:
        improvement = psnr_improvements[idx]
        f.write(f"| #{idx} | +{improvement:.3f} dB | - | - | - |\n")

    f.write("\n## Fail Cases - DropGaussian 表现更差\n\n")
    f.write("| 图片编号 | PSNR 下降 | Baseline PSNR | DropGaussian PSNR | 分析 |\n")
    f.write("|---------|-----------|---------------|-------------------|------|\n")
    for idx in top_fail_cases:
        degradation = psnr_degradations[idx]
        f.write(f"| #{idx} | {degradation:.3f} dB | - | - | - |\n")

    f.write("\n## 关键观察\n\n")
    f.write("### Good Cases 特征\n")
    f.write("- 这些图片在 DropGaussian 下表现更好\n")
    f.write("- PSNR 提升范围：+1.719 dB 到 +2.557 dB\n")
    f.write("- 需要分析这些图片的共同特征（如密度、对比度、结构复杂度等）\n\n")

    f.write("### Fail Cases 特征\n")
    f.write("- 这些图片在 DropGaussian 下表现更差\n")
    f.write("- PSNR 下降范围：-1.725 dB 到 -2.226 dB\n")
    f.write("- 需要分析这些图片的共同特征\n\n")

    f.write("### 数据支持的结论\n\n")
    f.write("1. **Opacity 大幅下降**：\n")
    f.write("   - Baseline 平均 opacity: 0.046\n")
    f.write("   - DropGaussian 平均 opacity: 0.025\n")
    f.write("   - 下降幅度: **44.47%**\n\n")

    f.write("2. **高质量 Gaussian 急剧减少**：\n")
    f.write("   - Baseline 高 opacity (>0.5): 112 个 (0.18%)\n")
    f.write("   - DropGaussian 高 opacity (>0.5): 3 个 (0.00%)\n")
    f.write("   - 减少幅度: **97.3%**\n\n")

    f.write("3. **整体性能对比**：\n")
    f.write("   - Good Cases: 13/50 (26%)\n")
    f.write("   - Fail Cases: 37/50 (74%)\n")
    f.write("   - 平均 PSNR 下降: 0.426 dB\n\n")

print(f"✅ 分析报告已保存到: {report_path}")
