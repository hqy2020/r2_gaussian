#!/usr/bin/env python3
"""
训练曲线分析工具

功能：
1. 解析TensorBoard事件文件
2. 提取关键指标曲线（PSNR、loss、高斯数）
3. 对比多个实验
4. 生成诊断报告

使用方法：
    python analyze_training.py --baseline_dir output/baseline_experiment/ --spags_dir output/spags_experiment/ --output_dir diagnosis/comparison/
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_eval_results(experiment_dir: Path) -> Dict[str, dict]:
    """
    从eval目录加载评估结果

    返回:
        dict: {iteration: {metric: value, ...}, ...}
    """
    results = {}
    eval_dir = experiment_dir / 'eval'

    if not eval_dir.exists():
        print(f"  警告: {eval_dir} 不存在")
        return results

    for iter_dir in sorted(eval_dir.iterdir()):
        if not iter_dir.is_dir():
            continue

        iter_name = iter_dir.name  # e.g., "iter_010000"
        try:
            iteration = int(iter_name.replace('iter_', ''))
        except ValueError:
            continue

        iter_results = {}

        # 读取3D评估
        eval3d_path = iter_dir / 'eval3d.yml'
        if eval3d_path.exists():
            with open(eval3d_path, 'r') as f:
                eval3d = yaml.safe_load(f)
            iter_results.update(eval3d)

        # 读取2D评估（test）
        eval2d_test_path = iter_dir / 'eval2d_render_test.yml'
        if eval2d_test_path.exists():
            with open(eval2d_test_path, 'r') as f:
                eval2d_test = yaml.safe_load(f)
            for k, v in eval2d_test.items():
                iter_results[f'2d_test_{k}'] = v

        # 读取2D评估（train）
        eval2d_train_path = iter_dir / 'eval2d_render_train.yml'
        if eval2d_train_path.exists():
            with open(eval2d_train_path, 'r') as f:
                eval2d_train = yaml.safe_load(f)
            for k, v in eval2d_train.items():
                iter_results[f'2d_train_{k}'] = v

        results[iteration] = iter_results

    return results


def load_tensorboard_events(experiment_dir: Path) -> Dict[str, List]:
    """
    从TensorBoard事件文件加载标量数据

    返回:
        dict: {tag: [(step, value), ...], ...}
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("  警告: tensorboard未安装，跳过事件文件解析")
        return {}

    events = defaultdict(list)

    # 查找事件文件
    event_files = list(experiment_dir.glob('events.out.tfevents.*'))
    if not event_files:
        print(f"  警告: {experiment_dir} 中未找到TensorBoard事件文件")
        return events

    for event_file in event_files:
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()

            # 获取所有标量标签
            scalar_tags = ea.Tags().get('scalars', [])

            for tag in scalar_tags:
                for scalar in ea.Scalars(tag):
                    events[tag].append((scalar.step, scalar.value))
        except Exception as e:
            print(f"  警告: 解析 {event_file} 失败: {e}")

    return events


def load_config(experiment_dir: Path) -> dict:
    """
    加载实验配置
    """
    config_path = experiment_dir / 'cfg_args.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    config_path = experiment_dir / 'cfg_args'
    if config_path.exists():
        # 旧格式，尝试解析
        config = {}
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key.strip()] = value.strip()
        return config

    return {}


def compare_experiments(experiments: Dict[str, Path]) -> dict:
    """
    对比多个实验

    Args:
        experiments: {name: dir_path, ...}

    返回:
        dict: 对比结果
    """
    comparison = {
        'experiments': {},
        'eval_results': {},
        'configs': {},
    }

    for name, exp_dir in experiments.items():
        exp_dir = Path(exp_dir)

        if not exp_dir.exists():
            print(f"  警告: {exp_dir} 不存在，跳过")
            continue

        print(f"\n加载实验: {name}")
        print(f"  路径: {exp_dir}")

        # 加载配置
        config = load_config(exp_dir)
        comparison['configs'][name] = config

        # 关键配置提取
        key_config = {
            'enable_fsgs_proximity': config.get('enable_fsgs_proximity', False),
            'enable_kplanes': config.get('enable_kplanes', False),
            'ply_path': config.get('ply_path', ''),
            'iterations': config.get('iterations', 30000),
        }
        comparison['experiments'][name] = key_config

        # 加载评估结果
        eval_results = load_eval_results(exp_dir)
        comparison['eval_results'][name] = eval_results
        print(f"  找到 {len(eval_results)} 个评估点")

    return comparison


def plot_comparison_curves(comparison: dict, output_dir: Path):
    """
    生成对比曲线图
    """
    eval_results = comparison['eval_results']

    if not eval_results:
        print("  没有评估结果可绘制")
        return

    # 收集所有指标
    all_metrics = set()
    for exp_results in eval_results.values():
        for iter_results in exp_results.values():
            all_metrics.update(iter_results.keys())

    # 关键指标
    key_metrics = ['psnr_3d', 'ssim_3d']

    # 绘制关键指标对比
    fig, axes = plt.subplots(1, len(key_metrics), figsize=(7*len(key_metrics), 5))
    if len(key_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, key_metrics):
        for exp_name, exp_results in eval_results.items():
            iterations = sorted(exp_results.keys())
            values = [exp_results[it].get(metric, None) for it in iterations]

            # 过滤None值
            valid_points = [(it, v) for it, v in zip(iterations, values) if v is not None]
            if valid_points:
                its, vals = zip(*valid_points)
                ax.plot(its, vals, marker='o', label=exp_name, linewidth=2, markersize=6)

        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'{metric.upper()} 对比', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存训练曲线: {output_path}")


def generate_comparison_table(comparison: dict, output_dir: Path):
    """
    生成对比表格（Markdown格式）
    """
    eval_results = comparison['eval_results']

    if not eval_results:
        return

    # 找到最终迭代的结果
    final_results = {}
    for exp_name, exp_results in eval_results.items():
        if exp_results:
            final_iter = max(exp_results.keys())
            final_results[exp_name] = {
                'iteration': final_iter,
                **exp_results[final_iter]
            }

    # 生成Markdown表格
    lines = ['# 实验对比表\n']
    lines.append('## 配置对比\n')
    lines.append('| 实验 | GAR启用 | ADM启用 | 初始化点云 | 迭代次数 |')
    lines.append('|------|---------|---------|------------|----------|')

    for exp_name, exp_config in comparison['experiments'].items():
        gar = '✓' if exp_config.get('enable_fsgs_proximity') else '✗'
        adm = '✓' if exp_config.get('enable_kplanes') else '✗'
        ply = exp_config.get('ply_path', 'N/A')
        if ply:
            ply = Path(ply).name if ply else 'N/A'
        iters = exp_config.get('iterations', 'N/A')
        lines.append(f'| {exp_name} | {gar} | {adm} | {ply} | {iters} |')

    lines.append('\n## 评估结果对比\n')
    lines.append('| 实验 | 迭代 | PSNR_3D | SSIM_3D | PSNR变化 |')
    lines.append('|------|------|---------|---------|----------|')

    # 计算baseline
    baseline_psnr = None
    for exp_name in ['baseline', 'Baseline']:
        if exp_name in final_results:
            baseline_psnr = final_results[exp_name].get('psnr_3d')
            break

    for exp_name, results in final_results.items():
        iter_num = results.get('iteration', 'N/A')
        psnr = results.get('psnr_3d', 'N/A')
        ssim = results.get('ssim_3d', 'N/A')

        if isinstance(psnr, float):
            psnr_str = f'{psnr:.4f}'
            if baseline_psnr is not None and exp_name.lower() != 'baseline':
                delta = psnr - baseline_psnr
                delta_str = f'{delta:+.4f} dB'
            else:
                delta_str = '-'
        else:
            psnr_str = str(psnr)
            delta_str = '-'

        if isinstance(ssim, float):
            ssim_str = f'{ssim:.4f}'
        else:
            ssim_str = str(ssim)

        lines.append(f'| {exp_name} | {iter_num} | {psnr_str} | {ssim_str} | {delta_str} |')

    # 保存
    output_path = output_dir / 'comparison_table.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✓ 保存对比表格: {output_path}")


def generate_analysis_report(comparison: dict, output_dir: Path):
    """
    生成分析报告
    """
    report = {
        'experiments': comparison['experiments'],
        'final_results': {},
        'diagnosis': []
    }

    eval_results = comparison['eval_results']

    # 提取最终结果
    for exp_name, exp_results in eval_results.items():
        if exp_results:
            final_iter = max(exp_results.keys())
            report['final_results'][exp_name] = {
                'iteration': final_iter,
                **exp_results[final_iter]
            }

    # 自动诊断
    diagnosis = report['diagnosis']

    # 找baseline
    baseline_psnr = None
    baseline_name = None
    for name in ['baseline', 'Baseline']:
        if name in report['final_results']:
            baseline_psnr = report['final_results'][name].get('psnr_3d')
            baseline_name = name
            break

    # 诊断：对比SPAGS和baseline
    for exp_name, results in report['final_results'].items():
        if exp_name.lower() == 'baseline':
            continue

        exp_config = comparison['experiments'].get(exp_name, {})
        psnr = results.get('psnr_3d')

        if psnr is None:
            continue

        if baseline_psnr is not None:
            delta = psnr - baseline_psnr

            if delta < -0.1:
                diagnosis.append({
                    'level': 'ERROR',
                    'experiment': exp_name,
                    'issue': f'{exp_name} PSNR 低于 Baseline',
                    'detail': f'PSNR = {psnr:.4f}, Baseline = {baseline_psnr:.4f}, 差值 = {delta:+.4f} dB',
                    'config': {
                        'GAR': exp_config.get('enable_fsgs_proximity', False),
                        'ADM': exp_config.get('enable_kplanes', False),
                    },
                    'suggestion': 'GAR/ADM可能在伤害性能，检查参数设置或禁用这些模块'
                })
            elif delta < 0.1:
                diagnosis.append({
                    'level': 'WARNING',
                    'experiment': exp_name,
                    'issue': f'{exp_name} PSNR 与 Baseline 接近',
                    'detail': f'PSNR = {psnr:.4f}, Baseline = {baseline_psnr:.4f}, 差值 = {delta:+.4f} dB',
                    'suggestion': 'GAR/ADM可能没有产生预期效果，需要进一步诊断'
                })
            else:
                diagnosis.append({
                    'level': 'OK',
                    'experiment': exp_name,
                    'issue': f'{exp_name} PSNR 优于 Baseline',
                    'detail': f'PSNR = {psnr:.4f}, Baseline = {baseline_psnr:.4f}, 提升 = {delta:+.4f} dB',
                    'suggestion': '继续优化以获得更大提升'
                })

    # 保存报告
    output_path = output_dir / 'training_analysis_report.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ 保存分析报告: {output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("训练分析报告摘要")
    print("="*60)

    print("\n实验配置:")
    for exp_name, exp_config in comparison['experiments'].items():
        gar = '✓' if exp_config.get('enable_fsgs_proximity') else '✗'
        adm = '✓' if exp_config.get('enable_kplanes') else '✗'
        print(f"  {exp_name}: GAR={gar}, ADM={adm}")

    print("\n最终结果:")
    for exp_name, results in report['final_results'].items():
        psnr = results.get('psnr_3d', 'N/A')
        ssim = results.get('ssim_3d', 'N/A')
        if isinstance(psnr, float):
            psnr_str = f'{psnr:.4f}'
        else:
            psnr_str = str(psnr)
        if isinstance(ssim, float):
            ssim_str = f'{ssim:.4f}'
        else:
            ssim_str = str(ssim)
        print(f"  {exp_name}: PSNR={psnr_str}, SSIM={ssim_str}")

    print("\n诊断结果:")
    for d in diagnosis:
        print(f"  [{d['level']}] {d['issue']}")
        print(f"         {d['detail']}")
        if d.get('suggestion'):
            print(f"         建议: {d['suggestion']}")

    print("="*60)

    return report


def main():
    parser = argparse.ArgumentParser(description='训练曲线分析工具')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='Baseline实验目录')
    parser.add_argument('--spags_dir', type=str, default=None,
                        help='SPAGS实验目录')
    parser.add_argument('--experiment_dirs', type=str, nargs='+', default=None,
                        help='多个实验目录（格式: name:path）')
    parser.add_argument('--output_dir', type=str, default='diagnosis/comparison',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"训练曲线分析工具")
    print(f"  输出目录: {output_dir}")
    print()

    # 构建实验字典
    experiments = {}

    if args.baseline_dir:
        experiments['Baseline'] = args.baseline_dir

    if args.spags_dir:
        experiments['SPAGS'] = args.spags_dir

    if args.experiment_dirs:
        for exp in args.experiment_dirs:
            if ':' in exp:
                name, path = exp.split(':', 1)
            else:
                name = Path(exp).name
                path = exp
            experiments[name] = path

    if not experiments:
        print("错误: 至少需要指定一个实验目录")
        parser.print_help()
        return

    # 对比实验
    print("加载实验数据...")
    comparison = compare_experiments(experiments)

    # 生成可视化
    print("\n生成可视化...")
    plot_comparison_curves(comparison, output_dir)

    # 生成对比表格
    print("\n生成对比表格...")
    generate_comparison_table(comparison, output_dir)

    # 生成分析报告
    print("\n生成分析报告...")
    generate_analysis_report(comparison, output_dir)

    print("\n分析完成！")


if __name__ == '__main__':
    main()
