#!/usr/bin/env python3
"""
分析 DropGaussian vs Baseline 的 Opacity 统计数据
验证 opacity 是否真的衰减
"""
import torch
import numpy as np
import os
import pickle

def load_checkpoint(ckpt_path):
    """加载模型 checkpoint"""
    if not os.path.exists(ckpt_path):
        return None

    if ckpt_path.endswith('.pickle'):
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)
    else:
        return torch.load(ckpt_path, map_location='cpu')

def analyze_opacity(ckpt, name="Model"):
    """分析 opacity 统计"""
    if ckpt is None:
        print(f"⚠️ {name}: Checkpoint 不存在")
        return None

    # 提取 opacity (density) 数据
    # R²-Gaussian 使用 'density' 存储 opacity
    if 'density' in ckpt:
        density = ckpt['density']
        if isinstance(density, torch.Tensor):
            density = density.numpy()
    elif '_density' in ckpt:
        density = ckpt['_density']
        if isinstance(density, torch.Tensor):
            density = density.numpy()
    elif 'opacity' in ckpt:
        density = ckpt['opacity']
        if isinstance(density, torch.Tensor):
            density = density.numpy()
    else:
        print(f"⚠️ {name}: 找不到 opacity/density 数据")
        print(f"可用的键: {list(ckpt.keys())}")
        return None

    # 计算激活后的 opacity (sigmoid)
    opacity = 1.0 / (1.0 + np.exp(-density))

    stats = {
        'name': name,
        'count': len(opacity),
        'mean': opacity.mean(),
        'std': opacity.std(),
        'median': np.median(opacity),
        'min': opacity.min(),
        'max': opacity.max(),
        'high_opacity_count': (opacity > 0.5).sum(),
        'high_opacity_ratio': (opacity > 0.5).sum() / len(opacity),
        'very_high_count': (opacity > 0.8).sum(),
        'very_high_ratio': (opacity > 0.8).sum() / len(opacity),
        'raw_density_mean': density.mean(),
        'raw_density_std': density.std(),
    }

    return stats

def print_stats(stats):
    """打印统计信息"""
    if stats is None:
        return

    print(f"\n{'='*80}")
    print(f"📊 {stats['name']} Opacity 统计")
    print(f"{'='*80}")
    print(f"Gaussian 数量: {stats['count']:,}")
    print(f"\nOpacity (激活后):")
    print(f"  均值: {stats['mean']:.6f}")
    print(f"  标准差: {stats['std']:.6f}")
    print(f"  中位数: {stats['median']:.6f}")
    print(f"  范围: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"\n高 Opacity (>0.5):")
    print(f"  数量: {stats['high_opacity_count']:,} / {stats['count']:,}")
    print(f"  比例: {stats['high_opacity_ratio']*100:.2f}%")
    print(f"\n超高 Opacity (>0.8):")
    print(f"  数量: {stats['very_high_count']:,} / {stats['count']:,}")
    print(f"  比例: {stats['very_high_ratio']*100:.2f}%")
    print(f"\nRaw Density (激活前):")
    print(f"  均值: {stats['raw_density_mean']:.6f}")
    print(f"  标准差: {stats['raw_density_std']:.6f}")

# 路径 - R²-Gaussian 使用 .pickle 格式保存模型
baseline_pt = "/home/qyhu/Documents/r2_ours/r2_gaussian/output/foot_3views_r2_baseline_1113/point_cloud/iteration_30000/point_cloud.pickle"
drop_pt = "/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_19_foot_3views_dropgaussian/point_cloud/iteration_30000/point_cloud.pickle"

print("="*80)
print("🔍 检查 Checkpoint 文件")
print("="*80)

for path in [baseline_pt, drop_pt]:
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"{exists} {path}")

print("\n" + "="*80)
print("📦 加载 Checkpoint")
print("="*80)

baseline_data = load_checkpoint(baseline_pt)
drop_data = load_checkpoint(drop_pt)

if baseline_data is not None:
    print("✅ Baseline checkpoint 加载成功")
else:
    print("❌ Baseline checkpoint 加载失败")

if drop_data is not None:
    print("✅ DropGaussian checkpoint 加载成功")
else:
    print("❌ DropGaussian checkpoint 加载失败")

# 分析 opacity
baseline_stats = analyze_opacity(baseline_data, "Baseline")
drop_stats = analyze_opacity(drop_data, "DropGaussian")

# 打印统计
print_stats(baseline_stats)
print_stats(drop_stats)

# 对比分析
if baseline_stats is not None and drop_stats is not None:
    print(f"\n{'='*80}")
    print("🔄 对比分析")
    print(f"{'='*80}")

    print(f"\nGaussian 数量变化:")
    print(f"  Baseline: {baseline_stats['count']:,}")
    print(f"  DropGaussian: {drop_stats['count']:,}")
    print(f"  变化: {drop_stats['count'] - baseline_stats['count']:+,} ({(drop_stats['count']/baseline_stats['count']-1)*100:+.2f}%)")

    print(f"\nOpacity 均值变化:")
    print(f"  Baseline: {baseline_stats['mean']:.6f}")
    print(f"  DropGaussian: {drop_stats['mean']:.6f}")
    print(f"  变化: {drop_stats['mean'] - baseline_stats['mean']:+.6f} ({(drop_stats['mean']/baseline_stats['mean']-1)*100:+.2f}%)")

    print(f"\n高 Opacity (>0.5) 比例变化:")
    print(f"  Baseline: {baseline_stats['high_opacity_ratio']*100:.2f}%")
    print(f"  DropGaussian: {drop_stats['high_opacity_ratio']*100:.2f}%")
    print(f"  变化: {(drop_stats['high_opacity_ratio'] - baseline_stats['high_opacity_ratio'])*100:+.2f} 个百分点")

    print(f"\n超高 Opacity (>0.8) 比例变化:")
    print(f"  Baseline: {baseline_stats['very_high_ratio']*100:.2f}%")
    print(f"  DropGaussian: {drop_stats['very_high_ratio']*100:.2f}%")
    print(f"  变化: {(drop_stats['very_high_ratio'] - baseline_stats['very_high_ratio'])*100:+.2f} 个百分点")

    # 保存报告
    report = f"""# Opacity 分析报告

## Baseline 统计

- Gaussian 数量: {baseline_stats['count']:,}
- Opacity 均值: {baseline_stats['mean']:.6f}
- 高 Opacity (>0.5) 比例: {baseline_stats['high_opacity_ratio']*100:.2f}%
- 超高 Opacity (>0.8) 比例: {baseline_stats['very_high_ratio']*100:.2f}%

## DropGaussian 统计

- Gaussian 数量: {drop_stats['count']:,}
- Opacity 均值: {drop_stats['mean']:.6f}
- 高 Opacity (>0.5) 比例: {drop_stats['high_opacity_ratio']*100:.2f}%
- 超高 Opacity (>0.8) 比例: {drop_stats['very_high_ratio']*100:.2f}%

## 对比

| 指标 | Baseline | DropGaussian | 变化 |
|------|----------|--------------|------|
| Gaussian 数量 | {baseline_stats['count']:,} | {drop_stats['count']:,} | {drop_stats['count'] - baseline_stats['count']:+,} ({(drop_stats['count']/baseline_stats['count']-1)*100:+.2f}%) |
| Opacity 均值 | {baseline_stats['mean']:.6f} | {drop_stats['mean']:.6f} | {drop_stats['mean'] - baseline_stats['mean']:+.6f} ({(drop_stats['mean']/baseline_stats['mean']-1)*100:+.2f}%) |
| 高 Opacity (>0.5) | {baseline_stats['high_opacity_ratio']*100:.2f}% | {drop_stats['high_opacity_ratio']*100:.2f}% | {(drop_stats['high_opacity_ratio'] - baseline_stats['high_opacity_ratio'])*100:+.2f} pp |
| 超高 Opacity (>0.8) | {baseline_stats['very_high_ratio']*100:.2f}% | {drop_stats['very_high_ratio']*100:.2f}% | {(drop_stats['very_high_ratio'] - baseline_stats['very_high_ratio'])*100:+.2f} pp |

## 结论

"""

    if drop_stats['mean'] < baseline_stats['mean']:
        report += f"DropGaussian 的平均 opacity 比 Baseline 低 {abs((drop_stats['mean']/baseline_stats['mean']-1)*100):.2f}%，确实存在 opacity 下降。\n"
    else:
        report += f"DropGaussian 的平均 opacity 比 Baseline 高 {(drop_stats['mean']/baseline_stats['mean']-1)*100:.2f}%，不存在 opacity 下降。\n"

    with open('/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/opacity_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n✅ 分析报告已保存到: cc-agent/experiments/opacity_analysis.md")
