#!/usr/bin/env python3
"""
9视角 7方法定性对比: 真值 | DNGaussian | CoR-GS | FSGS | X-Gaussian | R²-GS | SPAGS (本文)
按 PSNR 从低到高排列
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "chest": "胸部",
    "foot": "足部",
    "head": "头部",
    "abdomen": "腹部",
    "pancreas": "胰腺",
}

# 按 PSNR 从低到高排列
METHOD_ORDER = ["gt", "dngaussian", "corgs", "fsgs", "xgaussian", "r2gs", "spags"]
METHOD_LABELS = {
    "gt": "真值",
    "dngaussian": "DNGaussian",
    "corgs": "CoR-GS",
    "fsgs": "FSGS",
    "xgaussian": "X-Gaussian",
    "r2gs": "R²-GS",
    "spags": "SPAGS (本文)",
}

# 9 视角实验路径
EXPERIMENTS = {
    "spags": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_14_01_chest_9views_spags",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_02_16_foot_9views_spags",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_22_52_head_9views_spags",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_09_30_abdomen_9views_spags",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_23_03_pancreas_9views_spags",
    },
    "r2gs": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_05_chest_9views_baseline",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_13_27_foot_9views_baseline",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_18_02_head_9views_baseline",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_00_abdomen_9views_baseline",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_50_pancreas_9views_baseline",
    },
    "xgaussian": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_09_48_chest_9views_xgaussian",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_10_51_foot_9views_xgaussian",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_44_head_9views_xgaussian",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_16_41_abdomen_9views_xgaussian",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_20_31_pancreas_9views_xgaussian",
    },
    "fsgs": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_40_chest_9views_fsgs",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_50_foot_9views_fsgs",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_55_head_9views_fsgs",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_16_abdomen_9views_fsgs",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_36_pancreas_9views_fsgs",
    },
    "dngaussian": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_chest_9views_dngaussian",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_foot_9views_dngaussian",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_head_9views_dngaussian",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_abdomen_9views_dngaussian",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_pancreas_9views_dngaussian",
    },
    "corgs": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_chest_9views_corgs",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_08_14_foot_9views_corgs",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_head_9views_corgs",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_08_50_abdomen_9views_corgs",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_pancreas_9views_corgs",
    },
}

ITERATION = 30000


# =======================
# 工具函数
# =======================

def find_render_images(model_path: str, iteration: int, view_idx: int) -> Tuple[Optional[Path], Optional[Path]]:
    """查找渲染图像路径"""
    model_path = Path(model_path)
    render_dir = model_path / "test" / f"iter_{iteration}" / "render_test"
    if render_dir.exists():
        pred_file = render_dir / f"{view_idx:05d}_pred.png"
        gt_file = render_dir / f"{view_idx:05d}_gt.png"
        if pred_file.exists() and gt_file.exists():
            return pred_file, gt_file
    return None, None


def load_image(path: Path) -> Optional[np.ndarray]:
    """加载图像"""
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


def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算 PSNR"""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def find_best_view_for_organ(organ: str) -> int:
    """为每个器官找到 SPAGS 胜出的最佳视角"""
    methods = ["spags", "r2gs", "fsgs", "xgaussian"]

    # 获取可用视角数
    spags_path = EXPERIMENTS.get("spags", {}).get(organ)
    if not spags_path:
        return 25

    render_dir = Path(spags_path) / "test" / f"iter_{ITERATION}" / "render_test"
    if not render_dir.exists():
        return 25

    view_files = sorted(render_dir.glob("*_pred.png"))
    n_views = len(view_files)

    best_view = 25
    best_margin = -float('inf')

    for view_idx in range(n_views):
        psnrs = {}
        for method in methods:
            if method not in EXPERIMENTS or organ not in EXPERIMENTS[method]:
                continue
            model_path = EXPERIMENTS[method][organ]
            pred_path, gt_path = find_render_images(model_path, ITERATION, view_idx)
            if pred_path and gt_path:
                pred = load_image(pred_path)
                gt = load_image(gt_path)
                if pred is not None and gt is not None:
                    psnrs[method] = calc_psnr(pred, gt)

        if "spags" in psnrs and len(psnrs) >= 3:
            # SPAGS 相对其他方法的优势
            other_max = max(v for k, v in psnrs.items() if k != "spags")
            margin = psnrs["spags"] - other_max
            if margin > best_margin:
                best_margin = margin
                best_view = view_idx

    return best_view


# =======================
# 绘图函数
# =======================

def plot_comparison(output_path: Path, dpi: int = 300):
    """绘制 5×7 对比图"""
    # 先为每个器官找最佳视角
    print("为每个器官寻找最佳视角...")
    organ_views = {}
    for organ in ORGAN_ORDER:
        best_view = find_best_view_for_organ(organ)
        organ_views[organ] = best_view
        print(f"  {organ}: view_{best_view}")

    n_rows = len(ORGAN_ORDER)
    n_cols = len(METHOD_ORDER)

    cell_size = 2.0
    fig_w = n_cols * cell_size
    fig_h = n_rows * cell_size

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    gt_images: Dict[str, np.ndarray] = {}

    for r, organ in enumerate(ORGAN_ORDER):
        view_idx = organ_views[organ]

        for c, method in enumerate(METHOD_ORDER):
            ax = axes[r, c]
            ax.axis("off")

            if r == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold", pad=8, fontproperties=_FONT_PROP)

            if c == 0:
                ax.text(-0.05, 0.5, ORGAN_LABELS.get(organ, organ),
                       transform=ax.transAxes, fontsize=10, fontweight="bold",
                       va="center", ha="right", rotation=90, fontproperties=_FONT_PROP)

            img = None
            psnr = None

            if method == "gt":
                for m in ["r2gs", "spags", "xgaussian", "fsgs", "dngaussian", "corgs"]:
                    if m in EXPERIMENTS and organ in EXPERIMENTS[m]:
                        model_path = EXPERIMENTS[m][organ]
                        _, gt_path = find_render_images(model_path, ITERATION, view_idx)
                        if gt_path:
                            img = load_image(gt_path)
                            if img is not None:
                                gt_images[organ] = img
                                break
            else:
                if method in EXPERIMENTS and organ in EXPERIMENTS[method]:
                    model_path = EXPERIMENTS[method][organ]
                    pred_path, gt_path = find_render_images(model_path, ITERATION, view_idx)

                    if organ not in gt_images and gt_path:
                        gt_img = load_image(gt_path)
                        if gt_img is not None:
                            gt_images[organ] = gt_img

                    if pred_path:
                        img = load_image(pred_path)
                        if img is not None and organ in gt_images:
                            psnr = calc_psnr(img, gt_images[organ])

            if img is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                       transform=ax.transAxes, fontsize=12, color="#999999")
                ax.set_facecolor("#f5f5f5")
            else:
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")

                if psnr is not None and method != "gt":
                    ax.text(0.03, 0.97, f"{psnr:.2f}",
                           transform=ax.transAxes, fontsize=8, color="white",
                           fontweight="bold", va="top", ha="left",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

                if method == "spags":
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(2.5)
                        spine.set_edgecolor("#E91E63")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    print(f"已保存: {output_path}")

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    print(f"已保存: {pdf_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="9视角 7方法定性对比图")
    parser.add_argument("--output", type=str,
                       default="cc-agent/figures/fig_7methods_9views_cn.png",
                       help="输出路径")
    parser.add_argument("--dpi", type=int, default=300, help="输出 DPI")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / args.output

    print("=" * 60)
    print("9视角 7方法新视图合成定性对比")
    print(f"方法顺序 (PSNR从低到高): {' → '.join(METHOD_LABELS.values())}")
    print(f"器官: {' | '.join(ORGAN_LABELS.values())}")
    print(f"输出路径: {output_path}")
    print("=" * 60)

    plot_comparison(output_path, args.dpi)

    print("\n完成!")
    return 0


if __name__ == "__main__":
    exit(main())
