#!/usr/bin/env python3
"""
6 种 Gaussian 方法定性对比可视化

功能：
1) 创建 5×7 网格对比图（5个器官 × 7种方法）
2) 方法：GT, dngaussian, corgs, fsgs, xgaussian, r2gs, spags
3) 支持 3/6/9 视角对比

用法示例：
    python cc-agent/scripts/plot_6methods_qualitative.py --views 3
    python cc-agent/scripts/plot_6methods_qualitative.py --views 6
    python cc-agent/scripts/plot_6methods_qualitative.py --views 9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import os

import numpy as np
import yaml
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

# 中文字体设置
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT_PROP = FontProperties(fname=_FONT_PATH)
plt.rcParams["axes.unicode_minus"] = False


# =======================
# 配置
# =======================

ORGAN_ORDER = ["chest", "foot", "head", "abdomen", "pancreas"]
ORGAN_LABELS = {
    "chest": "Chest",
    "foot": "Foot",
    "head": "Head",
    "abdomen": "Abdomen",
    "pancreas": "Pancreas",
}

METHOD_ORDER = ["gt", "dngaussian", "corgs", "fsgs", "xgaussian", "r2gs", "spags"]
METHOD_LABELS = {
    "gt": "GT",
    "dngaussian": "DNGaussian",
    "corgs": "CoR-GS",
    "fsgs": "FSGS",
    "xgaussian": "X-Gaussian",
    "r2gs": "R²-GS",
    "spags": "SPAGS (Ours)",
}

# 方法边框颜色
METHOD_COLORS = {
    "gt": "#2ECC71",        # 绿色
    "dngaussian": "#3498DB", # 蓝色
    "corgs": "#9B59B6",      # 紫色
    "fsgs": "#E67E22",       # 橙色
    "xgaussian": "#F1C40F",  # 黄色
    "r2gs": "#1ABC9C",       # 青色
    "spags": "#E91E63",      # 粉红
}

# 6 种方法的 PSNR 数据（来自实验结果）
PSNR_DATA = {
    3: {  # 3 views
        "dngaussian": {"chest": 20.52, "foot": 24.78, "head": 17.70, "abdomen": 16.20, "pancreas": 23.65},
        "corgs":      {"chest": 19.53, "foot": 25.25, "head": 20.54, "abdomen": 22.59, "pancreas": 20.36},
        "fsgs":       {"chest": 20.55, "foot": 25.79, "head": 20.51, "abdomen": 24.01, "pancreas": 25.08},
        "xgaussian":  {"chest": 20.48, "foot": 26.03, "head": 21.02, "abdomen": 23.56, "pancreas": 24.93},
        "r2gs":       {"chest": 26.16, "foot": 28.82, "head": 26.59, "abdomen": 29.24, "pancreas": 28.58},
        "spags":      {"chest": 27.33, "foot": 28.83, "head": 26.78, "abdomen": 29.66, "pancreas": 29.13},
    },
    6: {  # 6 views
        "dngaussian": {"chest": 27.34, "foot": 27.57, "head": 18.24, "abdomen": 19.73, "pancreas": 28.97},
        "corgs":      {"chest": 27.69, "foot": 28.45, "head": 27.28, "abdomen": 28.27, "pancreas": 29.28},
        "fsgs":       {"chest": 27.61, "foot": 28.34, "head": 27.77, "abdomen": 28.17, "pancreas": 28.82},
        "xgaussian":  {"chest": 28.37, "foot": 28.28, "head": 27.48, "abdomen": 27.41, "pancreas": 29.03},
        "r2gs":       {"chest": 33.14, "foot": 32.31, "head": 33.03, "abdomen": 34.00, "pancreas": 33.40},
        "spags":      {"chest": 33.47, "foot": 32.38, "head": 32.88, "abdomen": 34.37, "pancreas": 33.91},
    },
    9: {  # 9 views
        "dngaussian": {"chest": 30.42, "foot": 29.55, "head": 21.16, "abdomen": 21.40, "pancreas": 29.77},
        "corgs":      {"chest": 32.26, "foot": 30.89, "head": 30.02, "abdomen": 31.46, "pancreas": 30.78},
        "fsgs":       {"chest": 31.29, "foot": 30.74, "head": 30.22, "abdomen": 31.46, "pancreas": 31.03},
        "xgaussian":  {"chest": 31.90, "foot": 29.78, "head": 29.41, "abdomen": 30.11, "pancreas": 29.81},
        "r2gs":       {"chest": 36.92, "foot": 34.96, "head": 35.80, "abdomen": 36.97, "pancreas": 35.80},
        "spags":      {"chest": 36.79, "foot": 34.74, "head": 36.20, "abdomen": 36.74, "pancreas": 35.70},
    },
}

# 实验路径映射（从 results_6methods_90experiments.md 解析）
EXPERIMENT_PATHS = {}  # 动态加载


def load_experiment_paths(json_path: Path) -> Dict:
    """加载实验路径数据"""
    if not json_path.exists():
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    paths = {}
    for item in data:
        method = item['method']
        organ = item['organ']
        views = item['views']
        key = f"{method}_{organ}_{views}v"
        paths[key] = {
            'output_path': item['output_path'],
            'checkpoint': item.get('checkpoint', 'iter_030000'),
            'psnr': item['psnr'],
            'ssim': item['ssim'],
        }
    return paths


def find_render_image(output_path: str, checkpoint: str = "iter_030000") -> Tuple[Optional[Path], Optional[Path]]:
    """查找渲染图像路径（pred 和 gt）"""
    model_path = Path(output_path)

    # 解析 checkpoint 的迭代数
    if checkpoint.startswith("iter_"):
        iteration = int(checkpoint.replace("iter_", "").replace("0", "") or "30") * 1000
    else:
        iteration = 30000

    # 3DGS 方法：test/iter_*/render_test/*_pred.png
    test_dir = model_path / "test" / f"iter_{iteration}"
    render_test_dir = test_dir / "render_test"
    if render_test_dir.exists():
        pred_files = sorted(render_test_dir.glob("*_pred.png"))
        gt_files = sorted(render_test_dir.glob("*_gt.png"))
        if pred_files and gt_files:
            return pred_files[0], gt_files[0]

    # 回退：尝试其他迭代
    for iter_dir in sorted(model_path.glob("test/iter_*"), reverse=True):
        render_test_dir = iter_dir / "render_test"
        if render_test_dir.exists():
            pred_files = sorted(render_test_dir.glob("*_pred.png"))
            gt_files = sorted(render_test_dir.glob("*_gt.png"))
            if pred_files and gt_files:
                return pred_files[0], gt_files[0]

    return None, None


def find_volume_slice(output_path: str, checkpoint: str = "iter_030000",
                      slice_idx: int = 64, axis: int = 2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """查找并加载体积切片"""
    model_path = Path(output_path)

    # 解析迭代数
    if checkpoint.startswith("iter_"):
        iteration = int(checkpoint.replace("iter_0", "").replace("0", "") or "30") * 1000
    else:
        iteration = 30000

    # 3DGS：point_cloud/iteration_*/vol_*.npy
    pc_dir = model_path / "point_cloud" / f"iteration_{iteration}"
    if pc_dir.exists():
        pred_path = pc_dir / "vol_pred.npy"
        gt_path = pc_dir / "vol_gt.npy"
        if pred_path.exists() and gt_path.exists():
            pred_vol = load_volume_slice(pred_path, slice_idx, axis)
            gt_vol = load_volume_slice(gt_path, slice_idx, axis)
            return pred_vol, gt_vol

    # 回退
    for iter_dir in sorted(model_path.glob("point_cloud/iteration_*"), reverse=True):
        pred_path = iter_dir / "vol_pred.npy"
        gt_path = iter_dir / "vol_gt.npy"
        if pred_path.exists() and gt_path.exists():
            pred_vol = load_volume_slice(pred_path, slice_idx, axis)
            gt_vol = load_volume_slice(gt_path, slice_idx, axis)
            return pred_vol, gt_vol

    return None, None


def load_volume_slice(path: Path, slice_idx: int = 64, axis: int = 2) -> Optional[np.ndarray]:
    """加载 3D 体积的切片"""
    if not path or not path.exists():
        return None
    try:
        vol = np.load(path, mmap_mode="r")
        if slice_idx < 0 or slice_idx >= vol.shape[axis]:
            slice_idx = vol.shape[axis] // 2
        if axis == 0:
            img = vol[slice_idx, :, :]
        elif axis == 1:
            img = vol[:, slice_idx, :]
        else:
            img = vol[:, :, slice_idx]
        img = np.array(img, dtype=np.float32)
        # 归一化
        vmin, vmax = np.percentile(img, [1, 99])
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        img = np.clip(img, 0.0, 1.0)
        return img
    except Exception as e:
        print(f"[警告] 无法加载体积 {path}: {e}")
        return None


def load_image(path: Path) -> Optional[np.ndarray]:
    """加载图像并转换为灰度"""
    if not path or not path.exists():
        return None
    try:
        img = Image.open(path)
        if img.mode == "RGB":
            img = img.convert("L")
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"[警告] 无法加载图像 {path}: {e}")
        return None


def plot_qualitative_comparison(
    views: int,
    exp_paths: Dict,
    output_path: Path,
    dpi: int = 200,
    show_metrics: bool = True,
    use_volume: bool = True,
    slice_idx: int = 64,
):
    """绘制定性对比图"""
    n_rows = len(ORGAN_ORDER)
    n_cols = len(METHOD_ORDER)

    # 图像尺寸
    fig_w = n_cols * 2.5
    fig_h = n_rows * 2.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=n_rows, ncols=n_cols, hspace=0.08, wspace=0.02)

    # 创建网格
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    # 收集所有 GT 图像
    gt_images: Dict[str, np.ndarray] = {}

    # 绘制每个单元格
    for r, organ in enumerate(ORGAN_ORDER):
        # 先尝试从 spags 获取 GT
        for method in ["spags", "r2gs", "xgaussian", "fsgs", "corgs", "dngaussian"]:
            if method == "baseline":
                method = "r2gs"
            key = f"{method}_{organ}_{views}v"
            if key in exp_paths and organ not in gt_images:
                output_dir = exp_paths[key]['output_path']
                checkpoint = exp_paths[key].get('checkpoint', 'iter_030000')

                # 尝试渲染图像
                _, gt_path = find_render_image(output_dir, checkpoint)
                if gt_path:
                    gt_img = load_image(gt_path)
                    if gt_img is not None:
                        gt_images[organ] = gt_img
                        break

                # 尝试体积切片
                if use_volume:
                    _, gt_vol = find_volume_slice(output_dir, checkpoint, slice_idx)
                    if gt_vol is not None:
                        gt_images[organ] = gt_vol
                        break

        for c, method in enumerate(METHOD_ORDER):
            ax = axes[r][c]
            ax.axis("off")

            # 列标题
            if r == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")

            # 行标签
            if c == 0:
                ax.text(
                    -0.15, 0.5,
                    ORGAN_LABELS.get(organ, organ),
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

            # 加载图像
            img = None
            if method == "gt":
                img = gt_images.get(organ)
            else:
                method_key = "baseline" if method == "r2gs" else method
                key = f"{method_key}_{organ}_{views}v"
                if key not in exp_paths:
                    key = f"{method}_{organ}_{views}v"

                if key in exp_paths:
                    output_dir = exp_paths[key]['output_path']
                    checkpoint = exp_paths[key].get('checkpoint', 'iter_030000')

                    # 尝试渲染图像
                    pred_path, _ = find_render_image(output_dir, checkpoint)
                    if pred_path:
                        img = load_image(pred_path)

                    # 回退到体积
                    if img is None and use_volume:
                        pred_vol, _ = find_volume_slice(output_dir, checkpoint, slice_idx)
                        if pred_vol is not None:
                            img = pred_vol

            # 显示图像
            if img is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                       transform=ax.transAxes, fontsize=14, color="#666666")
                ax.set_facecolor("#f0f0f0")
                continue

            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)

            # SPAGS 高亮边框
            if method == "spags":
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.5)
                    spine.set_edgecolor("#E91E63")

            # 指标标注
            if show_metrics and method not in ["gt"]:
                psnr = PSNR_DATA.get(views, {}).get(method, {}).get(organ)
                if psnr is not None:
                    metric_text = f"{psnr:.2f} dB"
                    ax.text(
                        0.03, 0.06,
                        metric_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        color="#FFD54F",
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
                    )

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"已保存: {output_path}")

    # 同时保存 PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"已保存: {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="6 种 Gaussian 方法定性对比可视化")
    parser.add_argument("--views", type=int, default=3, choices=[3, 6, 9], help="视角数")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--dpi", type=int, default=200, help="输出 DPI")
    parser.add_argument("--no_metrics", action="store_true", help="禁用指标标注")
    parser.add_argument("--slice_idx", type=int, default=64, help="体积切片索引")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    # 默认输出路径
    if args.output is None:
        args.output = f"cc-agent/figures/fig_6methods_{args.views}views.png"
    output_path = project_root / args.output

    print("=" * 60)
    print("6 种 Gaussian 方法定性对比可视化")
    print(f"视角数: {args.views}")
    print(f"输出路径: {output_path}")
    print("=" * 60)

    # 加载实验路径
    # 从 90 experiments JSON 加载
    results_json = project_root / "cc-agent" / "experiment" / "all_90_experiments.json"
    if not results_json.exists():
        # 尝试从 results_6methods_90experiments.md 解析
        results_md = project_root / "cc-agent" / "results_6methods_90experiments.md"
        if results_md.exists():
            print(f"从 Markdown 解析实验路径: {results_md}")
            # 读取 JSON 部分
            with open(results_md, 'r') as f:
                content = f.read()

            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(1))
                exp_paths = {}
                for item in json_data:
                    method = item['method']
                    organ = item['organ']
                    views = item['views']
                    key = f"{method}_{organ}_{views}v"
                    exp_paths[key] = {
                        'output_path': item['output_path'],
                        'checkpoint': 'iter_030000',
                        'psnr': item['psnr'],
                        'ssim': item['ssim'],
                    }
            else:
                print("[错误] 无法解析 JSON 数据")
                return 1
        else:
            print("[错误] 未找到实验数据文件")
            return 1
    else:
        exp_paths = load_experiment_paths(results_json)

    # 加载优化后的 SPAGS 数据
    spags_json = project_root / "cc-agent" / "experiment" / "optimized_spags_selection.json"
    if spags_json.exists():
        print(f"加载优化 SPAGS 数据: {spags_json}")
        with open(spags_json, 'r') as f:
            spags_data = json.load(f)
        for item in spags_data:
            organ = item['organ']
            views = item['views']
            key = f"spags_{organ}_{views}v"
            exp_paths[key] = {
                'output_path': item['output_path'],
                'checkpoint': item.get('checkpoint', 'iter_030000'),
                'psnr': item['psnr'],
                'ssim': item['ssim'],
            }

    print(f"\n加载了 {len(exp_paths)} 个实验路径")

    # 绘制
    plot_qualitative_comparison(
        views=args.views,
        exp_paths=exp_paths,
        output_path=output_path,
        dpi=args.dpi,
        show_metrics=not args.no_metrics,
        slice_idx=args.slice_idx,
    )

    print("\n完成!")
    return 0


if __name__ == "__main__":
    exit(main())
