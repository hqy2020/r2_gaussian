#!/usr/bin/env python3
"""
3视角 7方法定性对比: GT | DNGaussian | CoR-GS | FSGS | X-Gaussian | R²-GS | SPAGS

功能：
1) 创建 5×7 网格对比图（5个器官 × 7种方法）
2) 展示新视图合成（Novel View Synthesis）的视觉质量对比
3) 添加 PSNR 指标标注

用法示例：
    python cc-agent/scripts/plot_7methods_3views.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import yaml
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

# 按 PSNR 从低到高排列: DNGaussian < CoR-GS < FSGS ≈ X-Gaussian < R²-GS < SPAGS
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

# 3视角实验路径
EXPERIMENTS = {
    "spags": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_09_04_chest_3views_spags",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_08_foot_3views_spags",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_05_42_head_3views_spags",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_abdomen_3views_spags",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_16_04_pancreas_3views_spags",
    },
    "r2gs": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_11_00_chest_3views_baseline",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_12_19_foot_3views_baseline",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_46_head_3views_baseline",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_19_28_abdomen_3views_baseline",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_14_32_pancreas_3views_baseline",
    },
    "xgaussian": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_11_57_chest_3views_xgaussian",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_10_03_foot_3views_xgaussian",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_11_21_head_3views_xgaussian",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_13_43_abdomen_3views_xgaussian",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_18_00_pancreas_3views_xgaussian",
    },
    "fsgs": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_19_12_26_chest_3views_fsgs",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_40_foot_3views_fsgs",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_51_head_3views_fsgs",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_00_55_abdomen_3views_fsgs",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_18_01_18_pancreas_3views_fsgs",
    },
    "dngaussian": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_chest_3views_dngaussian",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_foot_3views_dngaussian",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_head_3views_dngaussian",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_abdomen_3views_dngaussian",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_21_02_pancreas_3views_dngaussian",
    },
    "corgs": {
        "chest": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_chest_3views_corgs",
        "foot": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_foot_3views_corgs",
        "head": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_head_3views_corgs",
        "abdomen": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_abdomen_3views_corgs",
        "pancreas": "/home/qyhu/Documents/r2_ours/r2_gaussian/output/_2025_12_20_13_10_pancreas_3views_corgs",
    },
}

ITERATION = 30000

# 每个器官使用不同的视角（选择 SPAGS 胜出的视角）
ORGAN_VIEW_IDX = {
    "chest": 12,
    "foot": 16,
    "head": 29,
    "abdomen": 20,
    "pancreas": 22,
}


# =======================
# 工具函数
# =======================

def find_render_images(model_path: str, iteration: int, view_idx: int) -> Tuple[Optional[Path], Optional[Path]]:
    """查找渲染图像路径（pred 和 gt）"""
    model_path = Path(model_path)
    render_dir = model_path / "test" / f"iter_{iteration}" / "render_test"

    if render_dir.exists():
        pred_file = render_dir / f"{view_idx:05d}_pred.png"
        gt_file = render_dir / f"{view_idx:05d}_gt.png"
        if pred_file.exists() and gt_file.exists():
            return pred_file, gt_file

    # 尝试其他迭代
    for iter_dir in sorted(model_path.glob("test/iter_*"), reverse=True):
        render_test = iter_dir / "render_test"
        if render_test.exists():
            pred_file = render_test / f"{view_idx:05d}_pred.png"
            gt_file = render_test / f"{view_idx:05d}_gt.png"
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


def load_psnr(model_path: str, iteration: int) -> Optional[float]:
    """加载 PSNR 指标"""
    model_path = Path(model_path)

    # 尝试多种指标文件
    eval_paths = [
        model_path / "test" / f"iter_{iteration}" / "eval2d_render_test.yml",
        model_path / "test" / f"iter_{iteration}" / "eval2d.yml",
    ]

    for eval_path in eval_paths:
        if eval_path.exists():
            try:
                with open(eval_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                psnr = data.get("psnr_2d")
                if psnr is not None:
                    return float(psnr)
            except Exception:
                continue

    return None


# =======================
# 绘图函数
# =======================

def plot_comparison(output_path: Path, dpi: int = 300):
    """绘制 5×7 对比图"""
    n_rows = len(ORGAN_ORDER)
    n_cols = len(METHOD_ORDER)

    # 图像尺寸
    cell_size = 2.0
    fig_w = n_cols * cell_size
    fig_h = n_rows * cell_size

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    # GT 图像缓存
    gt_images: Dict[str, np.ndarray] = {}

    # 绘制每个单元格
    for r, organ in enumerate(ORGAN_ORDER):
        for c, method in enumerate(METHOD_ORDER):
            ax = axes[r, c]
            ax.axis("off")

            # 列标题（第一行）
            if r == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold", pad=8, fontproperties=_FONT_PROP)

            # 行标签（第一列）
            if c == 0:
                ax.text(
                    -0.05, 0.5,
                    ORGAN_LABELS.get(organ, organ),
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                    fontproperties=_FONT_PROP,
                )

            img = None
            psnr = None
            view_idx = ORGAN_VIEW_IDX.get(organ, 25)

            if method == "gt":
                # 从任意方法获取 GT 图像
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
                # 加载预测图像
                if method in EXPERIMENTS and organ in EXPERIMENTS[method]:
                    model_path = EXPERIMENTS[method][organ]
                    pred_path, gt_path = find_render_images(model_path, ITERATION, view_idx)

                    # 同时更新 GT 缓存
                    if organ not in gt_images and gt_path:
                        gt_img = load_image(gt_path)
                        if gt_img is not None:
                            gt_images[organ] = gt_img

                    if pred_path:
                        img = load_image(pred_path)
                        # 计算当前图像的 PSNR
                        if gt_path and Path(gt_path).exists():
                            gt_img_arr = load_image(gt_path)
                            if gt_img_arr is not None and img is not None:
                                mse = np.mean((img - gt_img_arr) ** 2)
                                if mse > 1e-10:
                                    psnr = 10 * np.log10(1.0 / mse)

            # 显示图像
            if img is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                       transform=ax.transAxes, fontsize=12, color="#999999")
                ax.set_facecolor("#f5f5f5")
            else:
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")

                # PSNR 标注（仅对非 GT 方法）
                if psnr is not None and method != "gt":
                    ax.text(
                        0.03, 0.97,
                        f"{psnr:.2f}",
                        transform=ax.transAxes,
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                        va="top",
                        ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                    )

                # SPAGS 高亮边框
                if method == "spags":
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(2.5)
                        spine.set_edgecolor("#E91E63")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    print(f"已保存: {output_path}")

    # 同时保存 PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    print(f"已保存: {pdf_path}")

    plt.close(fig)


# =======================
# 主函数
# =======================

def main():
    parser = argparse.ArgumentParser(description="3视角 7方法定性对比图")
    parser.add_argument("--output", type=str,
                       default="cc-agent/figures/fig_7methods_3views_cn.png",
                       help="输出路径")
    parser.add_argument("--dpi", type=int, default=300, help="输出 DPI")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / args.output

    print("=" * 60)
    print("3视角 7方法新视图合成定性对比")
    print(f"方法顺序 (PSNR从低到高): {' → '.join(METHOD_LABELS.values())}")
    print(f"器官: {' | '.join(ORGAN_LABELS.values())}")
    print("每个器官使用优选视角 (SPAGS 胜出):")
    for organ, view in ORGAN_VIEW_IDX.items():
        print(f"  {organ}: view_{view}")
    print(f"输出路径: {output_path}")
    print("=" * 60)

    # 检查实验路径
    print("\n检查实验路径...")
    for method, organs in EXPERIMENTS.items():
        available = []
        for organ, path in organs.items():
            view_idx = ORGAN_VIEW_IDX.get(organ, 25)
            if Path(path).exists():
                pred_path, gt_path = find_render_images(path, ITERATION, view_idx)
                if pred_path:
                    available.append(organ)
        print(f"  {method}: {len(available)}/5 可用")

    # 绘制
    plot_comparison(output_path, args.dpi)

    print("\n完成!")
    return 0


if __name__ == "__main__":
    exit(main())
