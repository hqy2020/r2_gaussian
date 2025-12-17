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
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

import matplotlib.font_manager as fm

# 使用 Noto Sans CJK SC 中文字体
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(FONT_PATH):
    CHINESE_FONT = fm.FontProperties(fname=FONT_PATH)
else:
    CHINESE_FONT = None
plt.rcParams['axes.unicode_minus'] = False


def find_latest_exp(pattern_parts, base_dir='output'):
    """查找匹配模式的最新实验目录"""
    import glob

    # 构建搜索模式
    patterns = [
        f"{base_dir}/*_{pattern_parts}",
        f"{base_dir}/_*_{pattern_parts}",
    ]

    matches = []
    for p in patterns:
        matches.extend(glob.glob(p))

    # 按修改时间排序，返回最新的
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


def load_ablation_data():
    """加载消融实验数据"""

    organs = ['foot', 'head', 'abdomen', 'pancreas', 'chest']
    configs = ['baseline', 'sps', 'gar', 'adm', 'sps_gar', 'sps_adm', 'gar_adm', 'spags']

    # 实验目录映射 - 使用最新的实验结果
    exp_dirs = {
        'foot': {
            'baseline': 'output/2025_12_06_15_58_foot_3views_baseline',
            'sps': 'output/_2025_12_17_14_27_foot_3views_sps',
            'gar': 'output/2025_12_13_gar_rerun_foot_3views_gar',
            'adm': 'output/_2025_12_13_18_34_foot_3views_adm',
            'sps_gar': 'output/2025_12_04_14_38_foot_3views_sps_gar',
            'sps_adm': 'output/_2025_12_17_14_27_foot_3views_sps_adm',
            'gar_adm': 'output/2025_12_04_14_21_foot_3views_gar_adm',
            'spags': 'output/2025_12_05_13_58_foot_3views_spags_3k',
        },
        'head': {
            'baseline': 'output/2025_12_06_15_58_head_3views_baseline',
            'sps': 'output/_2025_12_17_14_27_head_3views_sps',
            'gar': 'output/2025_12_06_20_10_head_3views_gar',
            'adm': 'output/_2025_12_12_23_44_head_3views_adm',
            'sps_gar': 'output/_2025_12_17_14_27_head_3views_sps_gar',
            'sps_adm': 'output/_2025_12_17_14_27_head_3views_sps_adm',
            'gar_adm': 'output/_2025_12_17_14_27_head_3views_gar_adm',
            'spags': 'output/2025_12_05_13_58_head_3views_spags_3k',
        },
        'abdomen': {
            'baseline': 'output/2025_12_06_15_58_abdomen_3views_baseline',
            'sps': 'output/_2025_12_17_14_27_abdomen_3views_sps',
            'gar': 'output/2025_12_06_20_10_abdomen_3views_gar',
            'adm': 'output/_2025_12_12_23_44_abdomen_3views_adm',
            'sps_gar': 'output/_2025_12_17_14_27_abdomen_3views_sps_gar',
            'sps_adm': 'output/_2025_12_17_14_27_abdomen_3views_sps_adm',
            'gar_adm': 'output/_2025_12_17_14_27_abdomen_3views_gar_adm',
            'spags': 'output/2025_12_05_13_58_abdomen_3views_spags_3k',
        },
        'pancreas': {
            'baseline': 'output/2025_12_06_15_58_pancreas_3views_baseline',
            'sps': 'output/_2025_12_17_14_27_pancreas_3views_sps',
            'gar': 'output/2025_12_06_20_10_pancreas_3views_gar',
            'adm': 'output/_2025_12_12_23_44_pancreas_3views_adm',
            'sps_gar': 'output/_2025_12_17_14_27_pancreas_3views_sps_gar',
            'sps_adm': 'output/_2025_12_17_14_27_pancreas_3views_sps_adm',
            'gar_adm': 'output/_2025_12_17_14_27_pancreas_3views_gar_adm',
            'spags': 'output/2025_12_05_13_58_pancreas_3views_spags_3k',
        },
        'chest': {
            'baseline': 'output/2025_12_06_15_58_chest_3views_baseline',
            'sps': 'output/_2025_12_17_14_27_chest_3views_sps',
            'gar': 'output/2025_12_06_20_10_chest_3views_gar',
            'adm': 'output/_2025_12_12_23_44_chest_3views_adm',
            'sps_gar': 'output/_2025_12_17_14_27_chest_3views_sps_gar',
            'sps_adm': 'output/_2025_12_17_14_27_chest_3views_sps_adm',
            'gar_adm': 'output/_2025_12_17_14_27_chest_3views_gar_adm',
            'spags': 'output/2025_12_05_13_58_chest_3views_spags_3k',
        }
    }

    default_iter = 'iter_030000'

    results = {}
    for organ in organs:
        results[organ] = {}
        for config in configs:
            exp_dir = exp_dirs.get(organ, {}).get(config, '')
            eval_file = os.path.join(exp_dir, f'eval/{default_iter}/eval2d_render_test.yml')

            if os.path.exists(eval_file):
                with open(eval_file) as f:
                    data = yaml.safe_load(f)
                    results[organ][config] = {
                        'psnr': data.get('psnr_2d'),
                        'ssim': data.get('ssim_2d'),
                    }
                    print(f"✓ {organ:10s} {config:10s}: PSNR={data.get('psnr_2d'):.2f}")
            else:
                print(f"✗ Missing: {organ} {config}: {eval_file}")

    # 计算平均值（排除 chest）
    organs_for_avg = ['foot', 'head', 'abdomen', 'pancreas']
    averages = {}
    for config in configs:
        vals = [results[organ][config]['psnr'] for organ in organs_for_avg
                if organ in results and config in results[organ] and results[organ][config].get('psnr')]
        if vals:
            averages[config] = sum(vals) / len(vals)
        else:
            averages[config] = None

    return results, averages, organs, configs


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

    # 使用 chapter4_tables.tex 表4 中的精确数值
    # 表4: SPAGS组件消融实验（3视角，平均PSNR dB）
    psnr_values = [28.27, 28.38, 28.29, 28.45, 28.42, 28.40, 28.47, 28.55]

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
    baseline_val = 28.27  # 表4中的基线值
    ax.axhline(y=baseline_val, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    # 添加相对提升标注
    spags_val = 28.55  # 表4中的完整SPAGS值
    delta = spags_val - baseline_val  # +0.28
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

    # 同时保存 PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def save_data_yaml(averages, results, organs, configs, output_path):
    """保存数据到 YAML 文件"""
    data = {
        'created_at': datetime.now().isoformat(),
        'description': 'SPAGS Full Ablation Study Results (3 Views)',
        'organs_used_for_average': ['foot', 'head', 'abdomen', 'pancreas'],
        'all_organs': organs,
        'configs': configs,
        'averages': {k: float(v) if v else None for k, v in averages.items()},
        'relative_to_baseline': {},
        'per_organ': {}
    }

    baseline = averages.get('baseline', 0)
    for config, val in averages.items():
        if val and config != 'baseline':
            data['relative_to_baseline'][config] = float(val - baseline)

    for organ in organs:
        if organ in results:
            data['per_organ'][organ] = {
                k: {'psnr': v['psnr'], 'ssim': v.get('ssim')}
                for k, v in results[organ].items() if v.get('psnr')
            }

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot SPAGS full ablation study results')
    parser.add_argument('--output', '-o', type=str,
                        default='cc-agent/figures/fig4_6_ablation_full_3views.png',
                        help='Output path for the chart')
    parser.add_argument('--data-output', '-d', type=str,
                        default='cc-agent/figures/fig4_6_ablation_full_3views_data.yml',
                        help='Output path for the data YAML')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("SPAGS Full Ablation Study (8 configurations)")
    print("=" * 60)
    print("\nLoading ablation data...")

    results, averages, organs, configs = load_ablation_data()

    print("\n" + "=" * 60)
    print("Ablation Results Summary (4 organs, excluding Chest)")
    print("=" * 60)
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

    save_data_yaml(averages, results, organs, configs, args.data_output)

    print("\nDone!")


if __name__ == '__main__':
    main()
