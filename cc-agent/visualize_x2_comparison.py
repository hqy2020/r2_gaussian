#!/usr/bin/env python3
"""
x2-Gaussian vs Baseline 对比可视化脚本
生成日期: 2025-11-24
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 实验数据
organs = ['Chest', 'Foot', 'Abdomen', 'Head', 'Pancreas']
organs_zh = ['胸部', '足部', '腹部', '头部', '胰腺']

# PSNR 数据
psnr_baseline = [26.506, 28.4873, 29.2896, 26.6915, 28.7669]
psnr_x2 = [26.627, 28.683, 29.323, 26.732, 28.919]
psnr_improvement = [x2 - baseline for x2, baseline in zip(psnr_x2, psnr_baseline)]

# SSIM 数据
ssim_baseline = [0.8413, 0.9005, 0.9366, 0.9247, 0.9247]
ssim_x2 = [0.8451, 0.9007, 0.9373, 0.9254, 0.9265]
ssim_improvement = [x2 - baseline for x2, baseline in zip(ssim_x2, ssim_baseline)]

# 创建图表
fig = plt.figure(figsize=(16, 10))

# 1. PSNR 对比柱状图
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(organs))
width = 0.35

bars1 = ax1.bar(x - width/2, psnr_baseline, width, label='Baseline',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, psnr_x2, width, label='x2-Gaussian',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

ax1.set_xlabel('器官 (Organ)', fontsize=12, fontweight='bold')
ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
ax1.set_title('PSNR 对比 - Baseline vs x2-Gaussian', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{en}\n{zh}' for en, zh in zip(organs, organs_zh)])
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([25, 30])

# 在柱子上添加数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. SSIM 对比柱状图
ax2 = plt.subplot(2, 3, 2)
bars3 = ax2.bar(x - width/2, ssim_baseline, width, label='Baseline',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
bars4 = ax2.bar(x + width/2, ssim_x2, width, label='x2-Gaussian',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

ax2.set_xlabel('器官 (Organ)', fontsize=12, fontweight='bold')
ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
ax2.set_title('SSIM 对比 - Baseline vs x2-Gaussian', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{en}\n{zh}' for en, zh in zip(organs, organs_zh)])
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0.83, 0.94])

# 在柱子上添加数值
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# 3. PSNR 提升幅度
ax3 = plt.subplot(2, 3, 3)
colors = ['#2ecc71' if imp > 0 else '#e67e22' for imp in psnr_improvement]
bars5 = ax3.bar(x, psnr_improvement, color=colors, alpha=0.8,
                edgecolor='black', linewidth=1.2)

ax3.set_xlabel('器官 (Organ)', fontsize=12, fontweight='bold')
ax3.set_ylabel('PSNR 提升 (dB)', fontsize=12, fontweight='bold')
ax3.set_title('x2-Gaussian PSNR 提升幅度', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{en}\n{zh}' for en, zh in zip(organs, organs_zh)])
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 在柱子上添加数值
for bar in bars5:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'+{height:.3f}' if height > 0 else f'{height:.3f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

# 4. SSIM 提升幅度
ax4 = plt.subplot(2, 3, 4)
colors = ['#2ecc71' if imp > 0 else '#e67e22' for imp in ssim_improvement]
bars6 = ax4.bar(x, np.array(ssim_improvement) * 1000, color=colors, alpha=0.8,
                edgecolor='black', linewidth=1.2)

ax4.set_xlabel('器官 (Organ)', fontsize=12, fontweight='bold')
ax4.set_ylabel('SSIM 提升 (×10⁻³)', fontsize=12, fontweight='bold')
ax4.set_title('x2-Gaussian SSIM 提升幅度', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([f'{en}\n{zh}' for en, zh in zip(organs, organs_zh)])
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# 在柱子上添加数值
for i, bar in enumerate(bars6):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'+{height:.2f}' if height > 0 else f'{height:.2f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=10, fontweight='bold')

# 5. 综合性能雷达图
ax5 = plt.subplot(2, 3, 5, projection='polar')

# 归一化数据到 0-1 范围
psnr_norm_baseline = [(p - 26) / (30 - 26) for p in psnr_baseline]
psnr_norm_x2 = [(p - 26) / (30 - 26) for p in psnr_x2]
ssim_norm_baseline = [(s - 0.84) / (0.94 - 0.84) for s in ssim_baseline]
ssim_norm_x2 = [(s - 0.84) / (0.94 - 0.84) for s in ssim_x2]

# 合并 PSNR 和 SSIM（平均值）
radar_baseline = [(p + s) / 2 for p, s in zip(psnr_norm_baseline, ssim_norm_baseline)]
radar_x2 = [(p + s) / 2 for p, s in zip(psnr_norm_x2, ssim_norm_x2)]

angles = np.linspace(0, 2 * np.pi, len(organs), endpoint=False).tolist()
radar_baseline += radar_baseline[:1]
radar_x2 += radar_x2[:1]
angles += angles[:1]

ax5.plot(angles, radar_baseline, 'o-', linewidth=2, label='Baseline', color='#3498db')
ax5.fill(angles, radar_baseline, alpha=0.25, color='#3498db')
ax5.plot(angles, radar_x2, 'o-', linewidth=2, label='x2-Gaussian', color='#e74c3c')
ax5.fill(angles, radar_x2, alpha=0.25, color='#e74c3c')

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels([f'{en}\n{zh}' for en, zh in zip(organs, organs_zh)], fontsize=10)
ax5.set_ylim(0, 1)
ax5.set_title('综合性能雷达图\n(PSNR + SSIM 归一化)', fontsize=14, fontweight='bold', pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax5.grid(True, linestyle='--', alpha=0.5)

# 6. 相对提升百分比
ax6 = plt.subplot(2, 3, 6)
psnr_rel_improvement = [(x2 - baseline) / baseline * 100
                        for x2, baseline in zip(psnr_x2, psnr_baseline)]
ssim_rel_improvement = [(x2 - baseline) / baseline * 100
                        for x2, baseline in zip(ssim_x2, ssim_baseline)]

bars7 = ax6.bar(x - width/2, psnr_rel_improvement, width, label='PSNR',
                color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.2)
bars8 = ax6.bar(x + width/2, ssim_rel_improvement, width, label='SSIM',
                color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.2)

ax6.set_xlabel('器官 (Organ)', fontsize=12, fontweight='bold')
ax6.set_ylabel('相对提升 (%)', fontsize=12, fontweight='bold')
ax6.set_title('x2-Gaussian 相对提升百分比', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels([f'{en}\n{zh}' for en, zh in zip(organs, organs_zh)])
ax6.legend(fontsize=11)
ax6.grid(axis='y', alpha=0.3, linestyle='--')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)

# 在柱子上添加数值
for bars in [bars7, bars8]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'+{height:.2f}%' if height > 0 else f'{height:.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

plt.suptitle('R²-Gaussian vs x2-Gaussian 三视角实验对比分析',
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图表
output_path = 'cc-agent/x2-gaussian_3views_对比图表.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 图表已保存到: {output_path}")

# 显示图表
plt.show()

print("\n📊 可视化完成！")
print(f"- Markdown 报告: cc-agent/x2-gaussian_3views_实验对比报告.md")
print(f"- 可视化图表: {output_path}")
