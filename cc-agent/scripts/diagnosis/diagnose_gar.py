#!/usr/bin/env python3
"""
GAR (Geometry-Aware Refinement) 诊断工具

功能：
1. 从checkpoint加载高斯点云
2. 计算邻近分数分布
3. 生成可视化图表
4. 输出诊断报告

使用方法：
    python diagnose_gar.py --checkpoint output/xxx/point_cloud/iteration_30000/point_cloud.pickle --output_dir diagnosis/gar/
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from r2_gaussian.innovations.fsgs.proximity_densifier import ProximityGuidedDensifier


def load_gaussian_positions(checkpoint_path: str) -> np.ndarray:
    """
    从checkpoint加载高斯点位置

    支持格式:
    - .pickle: GaussianModel 序列化
    - .ply: 点云文件
    - .npy: numpy 数组
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == '.pickle':
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            # 检查常见的键名
            for key in ['xyz', '_xyz', 'positions', 'points']:
                if key in data:
                    return np.array(data[key])
            # 如果是完整的模型数据，尝试解析
            if 'active_sh_degree' in data:
                return np.array(data.get('xyz', data.get('_xyz')))
        raise ValueError(f"无法从 {checkpoint_path} 解析位置数据")

    elif checkpoint_path.suffix == '.npy':
        data = np.load(checkpoint_path)
        if data.ndim == 2 and data.shape[1] >= 3:
            return data[:, :3]
        raise ValueError(f"无法从 {checkpoint_path} 解析位置数据，shape={data.shape}")

    elif checkpoint_path.suffix == '.ply':
        # 尝试用 plyfile 加载
        try:
            from plyfile import PlyData
            plydata = PlyData.read(checkpoint_path)
            vertex = plydata['vertex']
            positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
            return positions
        except ImportError:
            raise ImportError("需要安装 plyfile: pip install plyfile")

    else:
        raise ValueError(f"不支持的文件格式: {checkpoint_path.suffix}")


def compute_proximity_statistics(positions: np.ndarray, k: int = 5, device: str = 'cuda') -> dict:
    """
    计算邻近分数统计

    返回:
        dict: 包含 scores, statistics, histogram_data
    """
    # 转换为 tensor
    positions_tensor = torch.from_numpy(positions).float().to(device)

    # 创建 densifier
    densifier = ProximityGuidedDensifier(
        k_neighbors=k,
        proximity_threshold=0.05,
        use_faiss=True
    )

    # 计算邻近分数
    scores, neighbor_indices, neighbor_distances = densifier.compute_proximity_scores(
        positions_tensor,
        return_neighbors=True
    )

    scores_np = scores.cpu().numpy()

    # 计算统计量
    statistics = {
        'count': len(scores_np),
        'mean': float(np.mean(scores_np)),
        'std': float(np.std(scores_np)),
        'min': float(np.min(scores_np)),
        'max': float(np.max(scores_np)),
        'median': float(np.median(scores_np)),
        'percentile_25': float(np.percentile(scores_np, 25)),
        'percentile_50': float(np.percentile(scores_np, 50)),
        'percentile_75': float(np.percentile(scores_np, 75)),
        'percentile_90': float(np.percentile(scores_np, 90)),
        'percentile_95': float(np.percentile(scores_np, 95)),
        'percentile_99': float(np.percentile(scores_np, 99)),
    }

    # 检查不同阈值下的候选点比例
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    threshold_analysis = {}
    for t in thresholds:
        count = np.sum(scores_np > t)
        ratio = count / len(scores_np)
        threshold_analysis[f'threshold_{t}'] = {
            'count': int(count),
            'ratio': float(ratio)
        }

    return {
        'scores': scores_np,
        'statistics': statistics,
        'threshold_analysis': threshold_analysis,
        'neighbor_indices': neighbor_indices.cpu().numpy() if neighbor_indices is not None else None,
        'neighbor_distances': neighbor_distances.cpu().numpy() if neighbor_distances is not None else None,
    }


def plot_proximity_histogram(scores: np.ndarray, output_path: str, threshold: float = 0.05):
    """
    生成邻近分数直方图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：全范围直方图
    ax1 = axes[0]
    ax1.hist(scores, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'阈值 = {threshold}')
    ax1.axvline(x=np.mean(scores), color='green', linestyle=':', linewidth=2, label=f'均值 = {np.mean(scores):.4f}')
    ax1.set_xlabel('邻近分数 (Proximity Score)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('邻近分数分布 (全范围)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加统计信息文本
    stats_text = f'样本数: {len(scores)}\n'
    stats_text += f'均值: {np.mean(scores):.4f}\n'
    stats_text += f'标准差: {np.std(scores):.4f}\n'
    stats_text += f'中位数: {np.median(scores):.4f}\n'
    stats_text += f'范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 右图：累积分布函数 (CDF)
    ax2 = axes[1]
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax2.plot(sorted_scores, cdf, color='steelblue', linewidth=2)
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax2.axhline(y=np.mean(scores > threshold), color='red', linestyle=':', alpha=0.5)
    ax2.set_xlabel('邻近分数 (Proximity Score)', fontsize=12)
    ax2.set_ylabel('累积概率', fontsize=12)
    ax2.set_title('累积分布函数 (CDF)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 标注阈值对应的百分位
    above_threshold = np.mean(scores > threshold)
    ax2.annotate(f'{above_threshold*100:.1f}% > {threshold}',
                 xy=(threshold, 1 - above_threshold),
                 xytext=(threshold + 0.02, 1 - above_threshold + 0.1),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 保存直方图: {output_path}")


def plot_3d_scatter(positions: np.ndarray, scores: np.ndarray, output_path: str, threshold: float = 0.05):
    """
    生成3D散点图 (使用 plotly 交互式可视化)
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("⚠ 未安装 plotly，跳过3D可视化。安装方法: pip install plotly")
        return

    # 采样点（如果太多会很慢）
    max_points = 50000
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        scores = scores[indices]
        print(f"  采样 {max_points} 个点用于可视化")

    # 创建散点图
    fig = go.Figure(data=[
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=scores,
                colorscale='Viridis',
                colorbar=dict(title='邻近分数'),
                opacity=0.8
            ),
            text=[f'Score: {s:.4f}' for s in scores],
            hoverinfo='text'
        )
    ])

    fig.update_layout(
        title='高斯点云邻近分数可视化',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=1000,
        height=800
    )

    fig.write_html(output_path)
    print(f"✓ 保存3D可视化: {output_path}")


def generate_diagnosis_report(results: dict, config: dict, output_path: str):
    """
    生成诊断报告
    """
    stats = results['statistics']
    threshold_analysis = results['threshold_analysis']

    report = {
        'summary': {
            'total_gaussians': stats['count'],
            'proximity_score_mean': stats['mean'],
            'proximity_score_std': stats['std'],
            'proximity_score_range': [stats['min'], stats['max']],
        },
        'statistics': stats,
        'threshold_analysis': threshold_analysis,
        'config': config,
        'diagnosis': []
    }

    # 自动诊断
    diagnosis = report['diagnosis']

    # 诊断1：分数范围
    if stats['max'] - stats['min'] < 0.01:
        diagnosis.append({
            'level': 'WARNING',
            'issue': '邻近分数区分度过低',
            'detail': f"分数范围仅 [{stats['min']:.4f}, {stats['max']:.4f}]，区分度不足",
            'suggestion': '检查点云初始化是否过于均匀，或K值设置是否合理'
        })

    # 诊断2：阈值设置
    threshold = config.get('threshold', 0.05)
    if threshold_analysis.get(f'threshold_{threshold}', {}).get('ratio', 0) < 0.01:
        diagnosis.append({
            'level': 'WARNING',
            'issue': '阈值设置过高',
            'detail': f"阈值 {threshold} 下只有 {threshold_analysis.get(f'threshold_{threshold}', {}).get('ratio', 0)*100:.2f}% 的点会被密化",
            'suggestion': '降低阈值或使用自适应阈值'
        })
    elif threshold_analysis.get(f'threshold_{threshold}', {}).get('ratio', 0) > 0.5:
        diagnosis.append({
            'level': 'WARNING',
            'issue': '阈值设置过低',
            'detail': f"阈值 {threshold} 下有 {threshold_analysis.get(f'threshold_{threshold}', {}).get('ratio', 0)*100:.2f}% 的点会被密化，可能过度密化",
            'suggestion': '提高阈值或使用自适应阈值'
        })

    # 诊断3：分布偏斜
    if stats['mean'] < stats['median']:
        diagnosis.append({
            'level': 'INFO',
            'issue': '分数分布左偏',
            'detail': f"均值 ({stats['mean']:.4f}) < 中位数 ({stats['median']:.4f})，存在少量高分异常点",
            'suggestion': '考虑使用百分位数作为自适应阈值'
        })

    # 如果没有发现问题
    if not diagnosis:
        diagnosis.append({
            'level': 'OK',
            'issue': '邻近分数分布正常',
            'detail': f"分数范围合理，阈值设置适当",
            'suggestion': '无需调整'
        })

    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ 保存诊断报告: {output_path}")

    # 打印摘要
    print("\n" + "="*60)
    print("GAR 诊断报告摘要")
    print("="*60)
    print(f"总高斯数: {stats['count']}")
    print(f"邻近分数: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
    print(f"邻近分数范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"各百分位: P25={stats['percentile_25']:.4f}, P50={stats['percentile_50']:.4f}, P75={stats['percentile_75']:.4f}, P90={stats['percentile_90']:.4f}")
    print("\n阈值分析:")
    for t_key, t_data in threshold_analysis.items():
        t_val = float(t_key.replace('threshold_', ''))
        print(f"  阈值 {t_val}: {t_data['count']} 点 ({t_data['ratio']*100:.2f}%)")
    print("\n诊断结果:")
    for d in diagnosis:
        print(f"  [{d['level']}] {d['issue']}")
        print(f"         {d['detail']}")
        if d['suggestion']:
            print(f"         建议: {d['suggestion']}")
    print("="*60)

    return report


def main():
    parser = argparse.ArgumentParser(description='GAR 诊断工具')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint路径 (.pickle, .npy, 或 .ply)')
    parser.add_argument('--output_dir', type=str, default='diagnosis/gar',
                        help='输出目录')
    parser.add_argument('--k', type=int, default=5,
                        help='K近邻数 (默认: 5)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='邻近阈值 (默认: 0.05)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GAR 诊断工具")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  K近邻: {args.k}")
    print(f"  阈值: {args.threshold}")
    print(f"  输出目录: {output_dir}")
    print()

    # 加载点云
    print("加载点云数据...")
    positions = load_gaussian_positions(args.checkpoint)
    print(f"  加载 {len(positions)} 个高斯点")

    # 计算邻近分数
    print("\n计算邻近分数...")
    results = compute_proximity_statistics(positions, k=args.k, device=args.device)

    # 配置信息
    config = {
        'checkpoint': str(args.checkpoint),
        'k': args.k,
        'threshold': args.threshold,
    }

    # 生成可视化
    print("\n生成可视化...")
    plot_proximity_histogram(
        results['scores'],
        str(output_dir / 'gar_proximity_histogram.png'),
        threshold=args.threshold
    )

    plot_3d_scatter(
        positions,
        results['scores'],
        str(output_dir / 'gar_3d_visualization.html'),
        threshold=args.threshold
    )

    # 生成诊断报告
    print("\n生成诊断报告...")
    generate_diagnosis_report(results, config, str(output_dir / 'gar_diagnosis_report.json'))

    print("\n诊断完成！")


if __name__ == '__main__':
    main()
