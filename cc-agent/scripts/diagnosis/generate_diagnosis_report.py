#!/usr/bin/env python3
"""
综合诊断报告生成器

整合 GAR、ADM、训练曲线分析的结果，生成完整诊断报告

使用方法：
    python generate_diagnosis_report.py \
        --gar_report diagnosis/gar/gar_diagnosis_report.json \
        --adm_report diagnosis/adm/adm_diagnosis_report.json \
        --training_report diagnosis/comparison/training_analysis_report.json \
        --output diagnosis/full_report.md
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_json_report(path: str) -> dict:
    """加载JSON报告"""
    if path and Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def generate_markdown_report(
    gar_report: dict,
    adm_report: dict,
    training_report: dict,
    output_path: str
):
    """
    生成Markdown格式的综合诊断报告
    """
    lines = []

    # 标题
    lines.append('# SPAGS 综合诊断报告')
    lines.append(f'\n生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 执行摘要
    lines.append('## 执行摘要\n')

    all_issues = []

    # 收集所有问题
    if gar_report and 'diagnosis' in gar_report:
        for d in gar_report['diagnosis']:
            d['source'] = 'GAR'
            all_issues.append(d)

    if adm_report and 'diagnosis' in adm_report:
        for d in adm_report['diagnosis']:
            d['source'] = 'ADM'
            all_issues.append(d)

    if training_report and 'diagnosis' in training_report:
        for d in training_report['diagnosis']:
            d['source'] = 'Training'
            all_issues.append(d)

    # 按严重程度排序
    severity_order = {'ERROR': 0, 'WARNING': 1, 'INFO': 2, 'OK': 3}
    all_issues.sort(key=lambda x: severity_order.get(x.get('level', 'INFO'), 2))

    # 统计
    error_count = sum(1 for i in all_issues if i.get('level') == 'ERROR')
    warning_count = sum(1 for i in all_issues if i.get('level') == 'WARNING')
    ok_count = sum(1 for i in all_issues if i.get('level') == 'OK')

    if error_count > 0:
        lines.append(f'**状态**: ❌ 发现 {error_count} 个错误, {warning_count} 个警告\n')
    elif warning_count > 0:
        lines.append(f'**状态**: ⚠️ 发现 {warning_count} 个警告\n')
    else:
        lines.append(f'**状态**: ✅ 一切正常\n')

    # GAR 部分
    lines.append('---\n')
    lines.append('## GAR (几何感知细化) 诊断\n')

    if gar_report:
        stats = gar_report.get('statistics', {})
        lines.append('### 邻近分数统计\n')
        lines.append('| 指标 | 值 |')
        lines.append('|------|-----|')
        lines.append(f"| 高斯点数 | {stats.get('count', 'N/A')} |")
        lines.append(f"| 均值 | {stats.get('mean', 'N/A'):.4f} |" if isinstance(stats.get('mean'), float) else f"| 均值 | {stats.get('mean', 'N/A')} |")
        lines.append(f"| 标准差 | {stats.get('std', 'N/A'):.4f} |" if isinstance(stats.get('std'), float) else f"| 标准差 | {stats.get('std', 'N/A')} |")
        lines.append(f"| 范围 | [{stats.get('min', 'N/A'):.4f}, {stats.get('max', 'N/A'):.4f}] |" if isinstance(stats.get('min'), float) else f"| 范围 | N/A |")

        # 阈值分析
        threshold_analysis = gar_report.get('threshold_analysis', {})
        if threshold_analysis:
            lines.append('\n### 阈值敏感性分析\n')
            lines.append('| 阈值 | 候选点数 | 比例 |')
            lines.append('|------|---------|------|')
            for t_key, t_data in sorted(threshold_analysis.items()):
                t_val = t_key.replace('threshold_', '')
                lines.append(f"| {t_val} | {t_data.get('count', 'N/A')} | {t_data.get('ratio', 0)*100:.1f}% |")

        # GAR诊断结果
        lines.append('\n### GAR 诊断结果\n')
        for d in gar_report.get('diagnosis', []):
            level = d.get('level', 'INFO')
            icon = {'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️', 'OK': '✅'}.get(level, '•')
            lines.append(f"- {icon} **{d.get('issue', 'N/A')}**")
            lines.append(f"  - {d.get('detail', '')}")
            if d.get('suggestion'):
                lines.append(f"  - 建议: {d.get('suggestion')}")
    else:
        lines.append('*GAR诊断报告未提供*\n')

    # ADM 部分
    lines.append('---\n')
    lines.append('## ADM (自适应密度调制) 诊断\n')

    if adm_report:
        kplanes_stats = adm_report.get('kplanes_statistics', {})
        summary = adm_report.get('summary', {})

        lines.append('### K-Planes 统计\n')

        if summary.get('kplanes_total_params'):
            lines.append(f"**总参数量**: {summary['kplanes_total_params']}\n")

        if kplanes_stats:
            lines.append('| 平面 | 形状 | 均值 | 标准差 | 范围 |')
            lines.append('|------|------|------|--------|------|')
            for plane_name, stats in kplanes_stats.items():
                shape = str(stats.get('shape', 'N/A'))
                mean = f"{stats.get('mean', 'N/A'):.4f}" if isinstance(stats.get('mean'), float) else 'N/A'
                std = f"{stats.get('std', 'N/A'):.4f}" if isinstance(stats.get('std'), float) else 'N/A'
                range_str = f"[{stats.get('min', 'N/A'):.4f}, {stats.get('max', 'N/A'):.4f}]" if isinstance(stats.get('min'), float) else 'N/A'
                lines.append(f"| {plane_name} | {shape} | {mean} | {std} | {range_str} |")

        # ADM诊断结果
        lines.append('\n### ADM 诊断结果\n')
        for d in adm_report.get('diagnosis', []):
            level = d.get('level', 'INFO')
            icon = {'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️', 'OK': '✅'}.get(level, '•')
            lines.append(f"- {icon} **{d.get('issue', 'N/A')}**")
            lines.append(f"  - {d.get('detail', '')}")
            if d.get('suggestion'):
                lines.append(f"  - 建议: {d.get('suggestion')}")
    else:
        lines.append('*ADM诊断报告未提供*\n')

    # 训练对比部分
    lines.append('---\n')
    lines.append('## 训练结果对比\n')

    if training_report:
        final_results = training_report.get('final_results', {})

        if final_results:
            lines.append('### 最终评估结果\n')
            lines.append('| 实验 | 迭代 | PSNR_3D | SSIM_3D |')
            lines.append('|------|------|---------|---------|')

            for exp_name, results in final_results.items():
                iter_num = results.get('iteration', 'N/A')
                psnr = f"{results.get('psnr_3d', 'N/A'):.4f}" if isinstance(results.get('psnr_3d'), float) else 'N/A'
                ssim = f"{results.get('ssim_3d', 'N/A'):.4f}" if isinstance(results.get('ssim_3d'), float) else 'N/A'
                lines.append(f"| {exp_name} | {iter_num} | {psnr} | {ssim} |")

        # 训练诊断结果
        lines.append('\n### 训练诊断结果\n')
        for d in training_report.get('diagnosis', []):
            level = d.get('level', 'INFO')
            icon = {'ERROR': '❌', 'WARNING': '⚠️', 'INFO': 'ℹ️', 'OK': '✅'}.get(level, '•')
            lines.append(f"- {icon} **{d.get('issue', 'N/A')}**")
            lines.append(f"  - {d.get('detail', '')}")
            if d.get('suggestion'):
                lines.append(f"  - 建议: {d.get('suggestion')}")
    else:
        lines.append('*训练分析报告未提供*\n')

    # 综合建议
    lines.append('---\n')
    lines.append('## 综合建议\n')

    if error_count > 0:
        lines.append('### 需要立即关注的问题\n')
        for issue in all_issues:
            if issue.get('level') == 'ERROR':
                lines.append(f"1. **[{issue['source']}] {issue.get('issue')}**")
                lines.append(f"   - {issue.get('detail')}")
                if issue.get('suggestion'):
                    lines.append(f"   - 建议: {issue.get('suggestion')}")

    if warning_count > 0:
        lines.append('\n### 需要注意的警告\n')
        for issue in all_issues:
            if issue.get('level') == 'WARNING':
                lines.append(f"1. **[{issue['source']}] {issue.get('issue')}**")
                lines.append(f"   - {issue.get('detail')}")
                if issue.get('suggestion'):
                    lines.append(f"   - 建议: {issue.get('suggestion')}")

    # 下一步行动
    lines.append('\n### 建议的下一步行动\n')

    if error_count > 0 or warning_count > 0:
        lines.append('1. 根据上述诊断结果，优先解决ERROR级别问题')
        lines.append('2. 运行控制实验验证假设')
        lines.append('3. 调整参数后重新训练并对比')
    else:
        lines.append('1. 当前配置看起来正常')
        lines.append('2. 如果性能仍不理想，考虑调整超参数')
        lines.append('3. 或者探索新的改进方向')

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ 保存综合诊断报告: {output_path}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='综合诊断报告生成器')
    parser.add_argument('--gar_report', type=str, default=None,
                        help='GAR诊断报告路径')
    parser.add_argument('--adm_report', type=str, default=None,
                        help='ADM诊断报告路径')
    parser.add_argument('--training_report', type=str, default=None,
                        help='训练分析报告路径')
    parser.add_argument('--output', type=str, default='diagnosis/full_report.md',
                        help='输出报告路径')

    args = parser.parse_args()

    print("综合诊断报告生成器")
    print()

    # 加载各报告
    gar_report = load_json_report(args.gar_report)
    if gar_report:
        print(f"✓ 加载GAR报告: {args.gar_report}")

    adm_report = load_json_report(args.adm_report)
    if adm_report:
        print(f"✓ 加载ADM报告: {args.adm_report}")

    training_report = load_json_report(args.training_report)
    if training_report:
        print(f"✓ 加载训练报告: {args.training_report}")

    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 生成报告
    print("\n生成综合报告...")
    report = generate_markdown_report(
        gar_report,
        adm_report,
        training_report,
        args.output
    )

    print("\n诊断完成！")


if __name__ == '__main__':
    main()
