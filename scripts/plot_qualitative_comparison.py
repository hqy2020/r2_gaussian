#!/usr/bin/env python3
"""
Fig 4-5: 定性对比实验可视化

生成 5行 × 7列 的网格布局图，展示 SPAGS 与其他方法在 CT 重建上的视觉质量对比。

用法:
    python scripts/plot_qualitative_comparison.py --views 3 --output figures/fig4_qualitative_3views.pdf
"""
import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, ConnectionPatch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 设置中文字体（如果需要）
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 器官和方法定义
ORGANS = ['chest', 'foot', 'head', 'abdomen', 'pancreas']
ORGAN_LABELS = ['Chest', 'Foot', 'Head', 'Abdomen', 'Pancreas']

# 方法顺序：GT, TensoRF, NAF, SAX-NeRF, X-Gaussian, R²-Gaussian, SPAGS
METHODS = ['gt', 'tensorf', 'naf', 'saxnerf', 'xgaussian', 'r2_gaussian', 'spags']
METHOD_LABELS = ['Ground Truth', 'TensoRF', 'NAF', 'SAX-NeRF', 'X-Gaussian', 'R²-Gaussian', 'SPAGS']

# 用于放大的 ROI 区域（每个器官的感兴趣区域）- [y_start, y_end, x_start, x_end]
# 这些值会根据实际 volume 尺寸进行调整
ROI_CONFIGS = {
    'chest': {'roi': [80, 130, 80, 130], 'slice_idx': 64},  # 肺部区域
    'foot': {'roi': [60, 110, 60, 110], 'slice_idx': 64},   # 骨骼区域
    'head': {'roi': [70, 120, 70, 120], 'slice_idx': 64},   # 脑部区域
    'abdomen': {'roi': [80, 130, 60, 110], 'slice_idx': 64}, # 腹部区域
    'pancreas': {'roi': [70, 120, 60, 110], 'slice_idx': 64}, # 胰腺区域
}


def find_experiment_path(output_dir: Path, organ: str, views: int, method: str) -> Optional[Path]:
    """查找指定器官、视角、方法的实验目录"""

    if method == 'gt':
        # GT 从任意实验获取
        return find_experiment_path(output_dir, organ, views, 'spags')

    # 方法名称映射
    method_suffix = {
        'tensorf': 'tensorf',
        'naf': 'naf',
        'saxnerf': 'saxnerf',
        'xgaussian': 'xgaussian',
        'r2_gaussian': 'baseline',
        'spags': 'spags',
    }.get(method, method)

    # 搜索模式（优先使用最新的实验）
    patterns = [
        f"_*_{organ}_{views}views_{method_suffix}",  # 新格式：_2025_12_15_...
        f"_*_{organ}_{views}views_{method_suffix}_*",
        f"*_{organ}_{views}views_{method_suffix}",
        f"*_{organ}_{views}views_{method_suffix}_*",
        f"*{organ}_{views}views_{method_suffix}*",
        f"spags_50k_*/{organ}_{views}views",  # 特殊目录结构
        f"baselines_comparison/*_{organ}_{views}views_{method_suffix}",
    ]

    all_matches = []
    for pattern in patterns:
        matches = list(output_dir.glob(pattern))
        for match in matches:
            if match.is_dir():
                all_matches.append(match)

    # 按修改时间倒序排序，优先使用最新的
    all_matches = sorted(all_matches, key=lambda x: x.stat().st_mtime, reverse=True)

    # 过滤掉没有 volume 数据的实验
    for match in all_matches:
        # 检查是否有 volume 文件
        has_volume = (
            (match / "volume" / "vol_pred.npy").exists() or
            (match / "point_cloud" / "iteration_30000" / "vol_pred.npy").exists()
        )
        if has_volume:
            return match

    # 如果没有带 volume 的，返回第一个匹配
    return all_matches[0] if all_matches else None


def load_volume(exp_path: Path, method: str, is_gt: bool = False) -> Optional[np.ndarray]:
    """加载 volume 数据"""
    if exp_path is None:
        return None

    # 尝试不同的路径
    possible_paths = [
        exp_path / "volume" / ("vol_gt.npy" if is_gt else "vol_pred.npy"),
        exp_path / "point_cloud" / "iteration_30000" / ("vol_gt.npy" if is_gt else "vol_pred.npy"),
    ]

    # 对于非 30k 的情况，尝试查找其他迭代
    if not is_gt:
        for iter_dir in sorted((exp_path / "point_cloud").glob("iteration_*"), reverse=True):
            possible_paths.append(iter_dir / "vol_pred.npy")

    for path in possible_paths:
        if path.exists():
            try:
                return np.load(path)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

    return None


def load_metrics(exp_path: Path, method: str) -> Dict[str, float]:
    """加载评估指标"""
    if exp_path is None:
        return {}

    # 尝试不同的路径
    possible_paths = [
        exp_path / "eval" / "iter_030000" / "eval3d.yml",
        exp_path / "eval" / "iter_030000" / f"eval3d_{method}.yml",
        exp_path / "eval" / "iter_002000" / "eval3d.yml",
    ]

    for iter_dir in sorted((exp_path / "eval").glob("iter_*"), reverse=True) if (exp_path / "eval").exists() else []:
        possible_paths.insert(0, iter_dir / "eval3d.yml")

    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                return {
                    'psnr': data.get('psnr_3d', 0),
                    'ssim': data.get('ssim_3d', 0),
                }
            except Exception as e:
                print(f"Warning: Failed to load metrics from {path}: {e}")

    return {}


def get_slice(volume: np.ndarray, slice_idx: int, axis: int = 2) -> np.ndarray:
    """获取指定切片"""
    if axis == 0:
        return volume[slice_idx, :, :]
    elif axis == 1:
        return volume[:, slice_idx, :]
    else:
        return volume[:, :, slice_idx]


def create_comparison_figure(
    output_dir: Path,
    views: int,
    output_path: Path,
    show_zoom: bool = True,
    show_metrics: bool = True,
    dpi: int = 300,
):
    """创建定性对比图"""

    n_rows = len(ORGANS)
    n_cols = len(METHODS)

    # 图像尺寸
    if show_zoom:
        fig_width = n_cols * 2.5 + 1.5
        fig_height = n_rows * 2.5 + 1.0
    else:
        fig_width = n_cols * 2.0 + 1.0
        fig_height = n_rows * 2.0 + 0.5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # 收集所有数据
    data = {}
    for organ in ORGANS:
        data[organ] = {}
        vol_gt = None

        for method in METHODS:
            if method == 'gt':
                continue

            exp_path = find_experiment_path(output_dir, organ, views, method)

            if exp_path is not None:
                vol_pred = load_volume(exp_path, method, is_gt=False)
                metrics = load_metrics(exp_path, method)

                if vol_gt is None:
                    vol_gt = load_volume(exp_path, method, is_gt=True)

                data[organ][method] = {
                    'volume': vol_pred,
                    'metrics': metrics,
                    'path': exp_path,
                }
            else:
                data[organ][method] = {
                    'volume': None,
                    'metrics': {},
                    'path': None,
                }
                print(f"Warning: No experiment found for {organ}_{views}views_{method}")

        data[organ]['gt'] = {
            'volume': vol_gt,
            'metrics': {'psnr': float('inf'), 'ssim': 1.0},
            'path': None,
        }

    # 找到全局最佳 PSNR/SSIM 用于标记
    best_metrics = {}
    for organ in ORGANS:
        best_psnr = max(
            (data[organ][m]['metrics'].get('psnr', 0) for m in METHODS if m != 'gt'),
            default=0
        )
        best_ssim = max(
            (data[organ][m]['metrics'].get('ssim', 0) for m in METHODS if m != 'gt'),
            default=0
        )
        best_metrics[organ] = {'psnr': best_psnr, 'ssim': best_ssim}

    # 绘制每个单元格
    for row_idx, organ in enumerate(ORGANS):
        roi_cfg = ROI_CONFIGS[organ]
        slice_idx = roi_cfg['slice_idx']
        roi = roi_cfg['roi']

        for col_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]

            vol = data[organ][method]['volume']
            metrics = data[organ][method]['metrics']

            if vol is not None:
                # 获取切片
                if slice_idx >= vol.shape[2]:
                    slice_idx = vol.shape[2] // 2

                img = get_slice(vol, slice_idx)

                # 归一化到 [0, 1]
                vmin, vmax = img.min(), img.max()
                if vmax > vmin:
                    img_norm = (img - vmin) / (vmax - vmin)
                else:
                    img_norm = np.zeros_like(img)

                ax.imshow(img_norm, cmap='gray', vmin=0, vmax=1)

                # 添加 ROI 框
                if show_zoom and method != 'gt':
                    # 调整 ROI 到实际尺寸
                    h, w = img.shape
                    roi_adj = [
                        int(roi[0] * h / 128),
                        int(roi[1] * h / 128),
                        int(roi[2] * w / 128),
                        int(roi[3] * w / 128),
                    ]

                    rect = Rectangle(
                        (roi_adj[2], roi_adj[0]),
                        roi_adj[3] - roi_adj[2],
                        roi_adj[1] - roi_adj[0],
                        linewidth=1,
                        edgecolor='red',
                        facecolor='none',
                    )
                    ax.add_patch(rect)

                # 添加指标标注
                if show_metrics and method != 'gt' and metrics:
                    psnr = metrics.get('psnr', 0)
                    ssim = metrics.get('ssim', 0)

                    # 高亮最佳结果
                    is_best_psnr = abs(psnr - best_metrics[organ]['psnr']) < 0.01
                    is_best_ssim = abs(ssim - best_metrics[organ]['ssim']) < 0.001

                    psnr_color = 'lime' if is_best_psnr else 'white'
                    ssim_color = 'lime' if is_best_ssim else 'white'

                    ax.text(
                        0.02, 0.98,
                        f'{psnr:.2f}',
                        transform=ax.transAxes,
                        fontsize=7,
                        color=psnr_color,
                        fontweight='bold' if is_best_psnr else 'normal',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5),
                    )
                    ax.text(
                        0.02, 0.82,
                        f'{ssim:.3f}',
                        transform=ax.transAxes,
                        fontsize=7,
                        color=ssim_color,
                        fontweight='bold' if is_best_ssim else 'normal',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5),
                    )
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('#f0f0f0')

            ax.axis('off')

            # 添加列标题（方法名）
            if row_idx == 0:
                ax.set_title(METHOD_LABELS[col_idx], fontsize=10, fontweight='bold')

            # 添加行标签（器官名）
            if col_idx == 0:
                ax.text(
                    -0.15, 0.5,
                    ORGAN_LABELS[row_idx],
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight='bold',
                    verticalalignment='center',
                    horizontalalignment='right',
                    rotation=90,
                )

    # 添加总标题
    fig.suptitle(
        f'{views}-View Sparse CT Reconstruction',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    # 添加图例说明
    legend_text = 'PSNR (dB) / SSIM shown in corners. Best results highlighted in green.'
    fig.text(0.5, 0.01, legend_text, ha='center', fontsize=8, style='italic')

    plt.tight_layout(rect=[0.05, 0.03, 1, 0.96])

    # 保存图像
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

    # 同时保存 PNG 格式
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved PNG to {png_path}")

    plt.close(fig)


def create_zoom_comparison(
    output_dir: Path,
    views: int,
    organ: str,
    output_path: Path,
    dpi: int = 300,
):
    """创建单器官的放大对比图（用于补充材料）"""

    n_methods = len(METHODS) - 1  # 不包括 GT

    fig, axes = plt.subplots(2, n_methods, figsize=(n_methods * 2, 4))

    roi_cfg = ROI_CONFIGS[organ]
    slice_idx = roi_cfg['slice_idx']
    roi = roi_cfg['roi']

    # 加载数据
    vol_gt = None
    for method_idx, method in enumerate([m for m in METHODS if m != 'gt']):
        exp_path = find_experiment_path(output_dir, organ, views, method)

        if exp_path is not None:
            vol = load_volume(exp_path, method, is_gt=False)
            metrics = load_metrics(exp_path, method)

            if vol_gt is None:
                vol_gt = load_volume(exp_path, method, is_gt=True)
        else:
            vol = None
            metrics = {}

        # 上排：完整切片
        ax_full = axes[0, method_idx]
        # 下排：放大区域
        ax_zoom = axes[1, method_idx]

        if vol is not None:
            if slice_idx >= vol.shape[2]:
                slice_idx = vol.shape[2] // 2

            img = get_slice(vol, slice_idx)

            # 归一化
            vmin, vmax = img.min(), img.max()
            if vmax > vmin:
                img_norm = (img - vmin) / (vmax - vmin)
            else:
                img_norm = np.zeros_like(img)

            # 完整图像
            ax_full.imshow(img_norm, cmap='gray')

            # 调整 ROI
            h, w = img.shape
            roi_adj = [
                int(roi[0] * h / 128),
                int(roi[1] * h / 128),
                int(roi[2] * w / 128),
                int(roi[3] * w / 128),
            ]

            # 画 ROI 框
            rect = Rectangle(
                (roi_adj[2], roi_adj[0]),
                roi_adj[3] - roi_adj[2],
                roi_adj[1] - roi_adj[0],
                linewidth=2,
                edgecolor='red',
                facecolor='none',
            )
            ax_full.add_patch(rect)

            # 放大区域
            zoom_img = img_norm[roi_adj[0]:roi_adj[1], roi_adj[2]:roi_adj[3]]
            ax_zoom.imshow(zoom_img, cmap='gray')
            ax_zoom.set_title(
                f"PSNR: {metrics.get('psnr', 0):.2f}\nSSIM: {metrics.get('ssim', 0):.3f}",
                fontsize=8
            )
        else:
            ax_full.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_full.transAxes)
            ax_zoom.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_zoom.transAxes)

        ax_full.axis('off')
        ax_zoom.axis('off')
        ax_full.set_title(METHOD_LABELS[METHODS.index(method)], fontsize=10)

    fig.suptitle(f'{ORGAN_LABELS[ORGANS.index(organ)]} - {views} Views', fontsize=12, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved zoom comparison to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate qualitative comparison figures')
    parser.add_argument('--views', type=int, default=3, choices=[3, 6, 9], help='Number of views')
    parser.add_argument('--output', type=str, default='figures/fig4_qualitative.pdf', help='Output path')
    parser.add_argument('--output_dir', type=str, default='output', help='Experiment output directory')
    parser.add_argument('--no_zoom', action='store_true', help='Disable zoom regions')
    parser.add_argument('--no_metrics', action='store_true', help='Disable metric annotations')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--zoom_only', type=str, help='Generate zoom comparison for specific organ')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = Path(args.output)

    if args.zoom_only:
        # 生成单个器官的放大对比图
        zoom_output = output_path.parent / f"zoom_{args.zoom_only}_{args.views}views.pdf"
        create_zoom_comparison(output_dir, args.views, args.zoom_only, zoom_output, args.dpi)
    else:
        # 生成完整对比图
        create_comparison_figure(
            output_dir,
            args.views,
            output_path,
            show_zoom=not args.no_zoom,
            show_metrics=not args.no_metrics,
            dpi=args.dpi,
        )


if __name__ == '__main__':
    main()
