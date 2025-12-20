#!/usr/bin/env python3
"""
图4-6: SPAGS 完整消融实验可视化

绘制所有 8 种配置的消融实验柱状图：
- Baseline, +SPS, +GAR, +ADM
- +SPS+GAR, +SPS+ADM, +GAR+ADM
- Full SPAGS (SPS+GAR+ADM)

使用方法：
    python cc-agent/scripts/plot_fig4_6_full_ablation.py [--output OUTPUT_PATH]
"""

import os
import json
import argparse
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

# 配置中文字体
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(FONT_PATH):
    CHINESE_FONT = fm.FontProperties(fname=FONT_PATH)
    # 设置全局字体（兼容旧版 matplotlib）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
else:
    CHINESE_FONT = None
    print("Warning: Chinese font not found, using default font")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # 使用 TrueType 字体
plt.rcParams['ps.fonttype'] = 42


def load_ablation_data_from_json():
    """从 JSON 文件加载消融实验数据"""

    # 加载消融实验结果
    ablation_file = 'cc-agent/experiment/ablation_results.json'
    all_exp_file = 'cc-agent/experiment/all_90_experiments.json'

    # 从 ablation_results.json 加载消融配置
    with open(ablation_file, 'r') as f:
        ablation_data = json.load(f)

    # 从 all_90_experiments.json 加载 baseline
    with open(all_exp_file, 'r') as f:
        all_data = json.load(f)

    # 提取 3 views 的数据
    configs = ['baseline', 'sps', 'gar', 'adm', 'sps_gar', 'sps_adm', 'gar_adm', 'spags']
    averages = {}

    # baseline 从 all_90_experiments 获取
    baseline_3v = [x['psnr'] for x in all_data if x['method'] == 'baseline' and x['views'] == 3]
    if baseline_3v:
        averages['baseline'] = sum(baseline_3v) / len(baseline_3v)

    # 其他配置从 ablation_results 获取
    for config in configs[1:]:  # 跳过 baseline
        psnrs = [x['psnr'] for x in ablation_data if x['config'] == config and x['views'] == 3 and x['psnr']]
        if psnrs:
            averages[config] = sum(psnrs) / len(psnrs)

    print("=== 3 views 平均 PSNR (从 JSON 加载) ===")
    for config in configs:
        if config in averages:
            print(f"{config}: {averages[config]:.2f}")

    return averages


def plot_full_ablation_chart(averages, output_path):
    """绘制完整消融实验柱状图"""

    # 配置标签和颜色（中文）
    config_labels = ['基线', '+SPS', '+GAR', '+ADM',
                     '+SPS\n+GAR', '+SPS\n+ADM', '+GAR\n+ADM',
                     '完整\nSPAGS']
    config_keys = ['baseline', 'sps', 'gar', 'adm',
                   'sps_gar', 'sps_adm', 'gar_adm', 'spags']

    # 颜色方案
    colors = [
        '#9E9E9E',  # 基线 - 灰色
        '#2196F3',  # SPS - 蓝色
        '#4CAF50',  # GAR - 绿色
        '#FF9800',  # ADM - 橙色
        '#00BCD4',  # SPS+GAR - 青色
        '#3F51B5',  # SPS+ADM - 靛蓝
        '#8BC34A',  # GAR+ADM - 浅绿
        '#9C27B0',  # 完整 SPAGS - 紫色
    ]

    # 从 averages 获取 PSNR 值
    psnr_values = [averages.get(k, 0) for k in config_keys]

    print("\n=== 绘图数据 ===")
    for label, key, val in zip(config_labels, config_keys, psnr_values):
        print(f"{label.replace(chr(10), '+')}: {val:.2f}")

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    x = np.arange(len(config_labels))
    bars = ax.bar(x, psnr_values, color=colors, edgecolor='black', linewidth=0.5, width=0.7)

    # 高亮完整 SPAGS
    bars[-1].set_edgecolor('#6A1B9A')
    bars[-1].set_linewidth(2)

    # 添加数值标注
    for bar, val in zip(bars, psnr_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='normal')

    # 基线参考线
    baseline_val = psnr_values[0]
    ax.axhline(y=baseline_val, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    # 添加相对提升标注
    spags_val = psnr_values[-1]
    delta = spags_val - baseline_val
    ax.annotate(f'+{delta:.2f} dB',
                xy=(7, spags_val),
                xytext=(7.6, spags_val + 0.03),
                fontsize=10, color='#6A1B9A', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.2))

    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_xlabel('配置', fontsize=12, fontproperties=CHINESE_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=10, fontproperties=CHINESE_FONT)

    # 动态设置 Y 轴范围
    valid_vals = [v for v in psnr_values if v > 0]
    if valid_vals:
        min_val = min(valid_vals) - 0.2
        max_val = max(valid_vals) + 0.25
        ax.set_ylim(min_val, max_val)

    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 添加分组说明
    ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(1.5, ax.get_ylim()[1] - 0.02, '单组件消融',
            ha='center', fontsize=9, color='gray', fontproperties=CHINESE_FONT)
    ax.text(5.5, ax.get_ylim()[1] - 0.02, '组件组合消融',
            ha='center', fontsize=9, color='gray', fontproperties=CHINESE_FONT)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {output_path}")

    # 保存 PDF（使用 SVG 作为中间格式避免字体问题）
    pdf_path = output_path.replace('.png', '.pdf')
    svg_path = output_path.replace('.png', '.svg')

    try:
        # 先保存 SVG
        plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
        print(f"SVG saved to: {svg_path}")

        # 尝试用 cairosvg 转 PDF
        try:
            import cairosvg
            cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
            print(f"PDF saved to: {pdf_path}")
        except ImportError:
            # 没有 cairosvg，尝试直接保存 PDF
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', format='pdf')
            print(f"PDF saved to: {pdf_path}")
    except Exception as e:
        print(f"PDF/SVG 保存失败: {e}")

    plt.close()


def save_data_json(averages, output_path):
    """保存数据到 JSON 文件"""
    configs = ['baseline', 'sps', 'gar', 'adm', 'sps_gar', 'sps_adm', 'gar_adm', 'spags']

    baseline = averages.get('baseline', 0)

    data = {
        'created_at': datetime.now().isoformat(),
        'description': 'SPAGS Full Ablation Study Results (3 Views, 5 Organs Average)',
        'configs': configs,
        'psnr_values': {k: round(averages.get(k, 0), 2) for k in configs},
        'relative_to_baseline': {k: round(averages.get(k, 0) - baseline, 2) for k in configs if k != 'baseline'},
        'improvement_spags_vs_baseline': round(averages.get('spags', 0) - baseline, 2)
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot SPAGS full ablation study results')
    parser.add_argument('--output', '-o', type=str,
                        default='cc-agent/figures/fig4_6_ablation_full_3views.png',
                        help='Output path for the chart')
    parser.add_argument('--data-output', '-d', type=str,
                        default='cc-agent/figures/fig4_6_ablation_full_3views_data.json',
                        help='Output path for the data JSON')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("SPAGS Full Ablation Study (8 configurations)")
    print("=" * 60)
    print("\nLoading ablation data from JSON...")

    averages = load_ablation_data_from_json()

    print("\n" + "=" * 60)
    print("Ablation Results Summary (5 organs average)")
    print("=" * 60)

    configs = ['baseline', 'sps', 'gar', 'adm', 'sps_gar', 'sps_adm', 'gar_adm', 'spags']
    print(f"{'Config':<12} {'PSNR (dB)':<12} {'vs Baseline':<12}")
    print("-" * 36)

    baseline = averages.get('baseline', 0)
    for config in configs:
        val = averages.get(config, 0)
        delta = val - baseline if val and baseline else 0
        sign = '+' if delta >= 0 else ''
        val_str = f"{val:.2f}" if val else "N/A"
        print(f"{config:<12} {val_str:<12} {sign}{delta:.2f} dB")

    print("\nGenerating chart...")
    plot_full_ablation_chart(averages, args.output)

    save_data_json(averages, args.data_output)

    print("\nDone!")


if __name__ == '__main__':
    main()
