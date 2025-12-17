#!/usr/bin/env python3
"""
Fig 4-5: 定性对比实验可视化

功能：
1) 创建 5×7 网格对比图（5个器官 × 7种方法）
2) 展示新视图合成（Novel View Synthesis）的视觉质量对比
3) 添加 ROI 放大区域和 PSNR/SSIM 指标标注

用法示例：
    python cc-agent/scripts/plot_fig4_5_qualitative.py \\
        --views 3 \\
        --output cc-agent/figures/fig4_5_qualitative_3views.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import time
import glob

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
    "chest": "胸部",
    "foot": "足部",
    "head": "头部",
    "abdomen": "腹部",
    "pancreas": "胰腺",
}

METHOD_ORDER = ["gt", "tensorf", "naf", "saxnerf", "xgaussian", "r2gs", "spags"]
METHOD_LABELS = {
    "gt": "真值",
    "tensorf": "TensoRF",
    "naf": "NAF",
    "saxnerf": "SAX-NeRF",
    "xgaussian": "X-Gaussian",
    "r2gs": "R²-Gaussian",
    "spags": "SPAGS (本文)",
}

# 方法到内部名称的映射（用于查找模型路径）
METHOD_KEY_MAP = {
    "tensorf": "tensorf",
    "naf": "naf",
    "saxnerf": "sax-nerf",
    "xgaussian": "xgs",
    "r2gs": "r2gs",
    "spags": "spags",
}

# ROI 配置（基于 2D 渲染图像尺寸，会自动缩放）
ROI_CONFIGS = {
    "chest":    {"roi": [180, 280, 180, 280], "view_idx": 0},
    "foot":     {"roi": [150, 250, 150, 250], "view_idx": 0},
    "head":     {"roi": [160, 260, 160, 260], "view_idx": 0},
    "abdomen":  {"roi": [180, 280, 150, 250], "view_idx": 0},
    "pancreas": {"roi": [160, 260, 150, 250], "view_idx": 0},
}

# 方法边框颜色
METHOD_COLORS = {
    "gt": "#2ECC71",        # 绿色
    "tensorf": "#3498DB",   # 蓝色
    "naf": "#9B59B6",       # 紫色
    "saxnerf": "#E67E22",   # 橙色
    "xgaussian": "#F1C40F", # 黄色
    "r2gs": "#1ABC9C",      # 青色
    "spags": "#E91E63",     # 粉红
}

# 论文表格中的 PSNR 数据（用于图片标注）
# 格式: PAPER_PSNR[views][method][organ] = psnr_value
PAPER_PSNR = {
    3: {  # 3视角
        "tensorf":   {"chest": 24.62, "foot": 27.06, "head": 25.17, "abdomen": 27.74, "pancreas": 27.10},
        "naf":       {"chest": 24.84, "foot": 27.28, "head": 25.39, "abdomen": 27.96, "pancreas": 27.32},
        "saxnerf":   {"chest": 25.26, "foot": 27.70, "head": 25.81, "abdomen": 28.38, "pancreas": 27.74},
        "xgaussian": {"chest": 20.47, "foot": 25.71, "head": 21.73, "abdomen": 23.02, "pancreas": 25.18},
        "r2gs":      {"chest": 26.12, "foot": 28.56, "head": 26.67, "abdomen": 29.24, "pancreas": 28.60},
        "spags":     {"chest": 25.86, "foot": 28.78, "head": 26.88, "abdomen": 29.28, "pancreas": 29.10},
    },
    6: {  # 6视角
        "tensorf":   {"chest": 31.24, "foot": 30.51, "head": 31.11, "abdomen": 32.11, "pancreas": 31.58},
        "naf":       {"chest": 31.91, "foot": 31.18, "head": 31.78, "abdomen": 32.78, "pancreas": 32.25},
        "saxnerf":   {"chest": 32.12, "foot": 31.39, "head": 31.99, "abdomen": 32.99, "pancreas": 32.46},
        "xgaussian": {"chest": 28.59, "foot": 28.48, "head": 27.56, "abdomen": 27.79, "pancreas": 29.22},
        "r2gs":      {"chest": 33.24, "foot": 32.51, "head": 33.11, "abdomen": 34.11, "pancreas": 33.58},
        "spags":     {"chest": 33.26, "foot": 32.37, "head": 33.32, "abdomen": 34.04, "pancreas": 33.80},
    },
    9: {  # 9视角
        "tensorf":   {"chest": 34.95, "foot": 32.95, "head": 34.37, "abdomen": 35.53, "pancreas": 34.21},
        "naf":       {"chest": 35.91, "foot": 33.91, "head": 34.83, "abdomen": 35.99, "pancreas": 34.67},
        "saxnerf":   {"chest": 36.14, "foot": 34.14, "head": 35.06, "abdomen": 36.22, "pancreas": 34.90},
        "xgaussian": {"chest": 32.40, "foot": 29.12, "head": 29.95, "abdomen": 30.02, "pancreas": 30.48},
        "r2gs":      {"chest": 36.95, "foot": 34.95, "head": 35.87, "abdomen": 37.03, "pancreas": 35.71},
        "spags":     {"chest": 37.07, "foot": 35.00, "head": 36.48, "abdomen": 36.85, "pancreas": 35.63},
    },
}


# =======================
# 数据结构
# =======================

@dataclass
class ExperimentInfo:
    organ: str
    method: str
    model_path: Path
    iteration: int
    render_path: Optional[Path] = None
    gt_path: Optional[Path] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None


# =======================
# 工具函数
# =======================

def parse_paths_md(md_path: Path, views: int) -> Dict[str, Dict[str, ExperimentInfo]]:
    """解析 logs/latest_paths_15x6.md 获取实验路径"""
    if not md_path.exists():
        print(f"[警告] 路径清单文件不存在: {md_path}")
        return {}

    result: Dict[str, Dict[str, ExperimentInfo]] = {}
    text = md_path.read_text()

    # 解析表格行
    # 表格格式: | 序号 | 器官 | 视角数 | 方法 | iterations | 是否30k | 输出目录 | 日志文件 | 日志时间 |
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 9:
            continue
        try:
            # 跳过表头和分隔行
            if "器官" in parts[2] or "---" in parts[2]:
                continue

            organ = parts[2].strip()
            view_str = parts[3].strip()
            method = parts[4].strip()
            iterations = int(parts[5].strip())
            output_dir = parts[7].strip().strip("`")

            if organ not in ORGAN_ORDER:
                continue
            if int(view_str) != views:
                continue

            # 方法名映射
            method_key = None
            for key, val in METHOD_KEY_MAP.items():
                if val == method:
                    method_key = key
                    break
            if method_key is None:
                continue

            if organ not in result:
                result[organ] = {}

            result[organ][method_key] = ExperimentInfo(
                organ=organ,
                method=method_key,
                model_path=Path(output_dir),
                iteration=iterations,
            )
        except (ValueError, IndexError):
            continue

    return result


def find_render_images(exp: ExperimentInfo, project_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """查找渲染图像路径（pred 和 gt）"""
    model_path = project_root / exp.model_path

    # NeRF 方法：renders/iter_*/pred_*.png
    render_dir = model_path / "renders" / f"iter_{exp.iteration:06d}"
    if render_dir.exists():
        pred_files = sorted(render_dir.glob("pred_*.png"))
        gt_files = sorted(render_dir.glob("gt_*.png"))
        if pred_files and gt_files:
            return pred_files[0], gt_files[0]

    # 3DGS 方法：test/iter_*/render_test/*_pred.png
    test_dir = model_path / "test" / f"iter_{exp.iteration}"
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

    for iter_dir in sorted(model_path.glob("renders/iter_*"), reverse=True):
        pred_files = sorted(iter_dir.glob("pred_*.png"))
        gt_files = sorted(iter_dir.glob("gt_*.png"))
        if pred_files and gt_files:
            return pred_files[0], gt_files[0]

    return None, None


def find_volume_data(exp: ExperimentInfo, project_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """查找 3D 体积数据路径（pred 和 gt）- 作为 2D 渲染的备选"""
    model_path = project_root / exp.model_path

    # 3DGS 方法：point_cloud/iteration_*/vol_*.npy
    pc_dir = model_path / "point_cloud" / f"iteration_{exp.iteration}"
    if pc_dir.exists():
        pred_path = pc_dir / "vol_pred.npy"
        gt_path = pc_dir / "vol_gt.npy"
        if pred_path.exists() and gt_path.exists():
            return pred_path, gt_path

    # NeRF 方法：volume/vol_*.npy
    vol_dir = model_path / "volume"
    if vol_dir.exists():
        pred_path = vol_dir / "vol_pred.npy"
        gt_path = vol_dir / "vol_gt.npy"
        if pred_path.exists() and gt_path.exists():
            return pred_path, gt_path

    # 回退：尝试其他迭代
    for iter_dir in sorted(model_path.glob("point_cloud/iteration_*"), reverse=True):
        pred_path = iter_dir / "vol_pred.npy"
        gt_path = iter_dir / "vol_gt.npy"
        if pred_path.exists() and gt_path.exists():
            return pred_path, gt_path

    # test 目录
    for iter_dir in sorted(model_path.glob("test/iter_*"), reverse=True):
        pred_path = iter_dir / "vol_pred.npy"
        gt_path = iter_dir / "vol_gt.npy"
        if pred_path.exists() and gt_path.exists():
            return pred_path, gt_path

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


def load_eval_metrics(exp: ExperimentInfo, project_root: Path) -> Tuple[Optional[float], Optional[float]]:
    """加载评估指标"""
    model_path = project_root / exp.model_path

    # 尝试多种指标文件
    eval_files = [
        model_path / "eval" / f"iter_{exp.iteration:06d}" / "eval2d_render_test.yml",
        model_path / "test" / f"iter_{exp.iteration}" / "eval2d_render_test.yml",
        model_path / "eval" / f"iter_{exp.iteration:06d}" / f"eval2d_{exp.method}.yml",
    ]

    for eval_file in eval_files:
        if eval_file.exists():
            try:
                with open(eval_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                psnr = data.get("psnr_2d")
                ssim = data.get("ssim_2d")
                if psnr is not None:
                    return float(psnr), float(ssim) if ssim else None
            except Exception:
                continue

    return None, None


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


def scale_roi(roi: List[int], img_shape: Tuple[int, int], base_size: int = 128) -> List[int]:
    """根据图像尺寸缩放 ROI"""
    h, w = img_shape
    scale_h = h / base_size
    scale_w = w / base_size
    return [
        int(roi[0] * scale_h),
        int(roi[1] * scale_h),
        int(roi[2] * scale_w),
        int(roi[3] * scale_w),
    ]


# =======================
# 绘图函数
# =======================

def plot_qualitative_comparison(
    experiments: Dict[str, Dict[str, ExperimentInfo]],
    project_root: Path,
    output_path: Path,
    views: int,
    dpi: int = 200,
    show_roi: bool = True,
    show_metrics: bool = True,
    use_volume_fallback: bool = True,
    slice_idx: int = 64,
    slice_axis: int = 2,
):
    """绘制定性对比图"""
    n_rows = len(ORGAN_ORDER)
    n_cols = len(METHOD_ORDER)

    # 图像尺寸
    fig_w = n_cols * 3.0
    fig_h = n_rows * 3.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=n_rows, ncols=n_cols, hspace=0.08, wspace=0.02)

    # 创建网格
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    # 收集所有 GT 图像
    gt_images: Dict[str, np.ndarray] = {}
    data_source = "unknown"

    # 绘制每个单元格
    for r, organ in enumerate(ORGAN_ORDER):
        organ_exps = experiments.get(organ, {})
        roi_cfg = ROI_CONFIGS.get(organ, {"roi": [40, 90, 40, 90], "view_idx": 0})
        roi = roi_cfg["roi"]

        # 先加载 GT 图像
        gt_img = None
        for method_key in ["spags", "r2gs", "xgaussian", "tensorf", "naf", "saxnerf"]:
            if method_key in organ_exps:
                exp = organ_exps[method_key]
                # 优先尝试渲染图像
                _, gt_path = find_render_images(exp, project_root)
                if gt_path:
                    gt_img = load_image(gt_path)
                    if gt_img is not None:
                        gt_images[organ] = gt_img
                        data_source = "render"
                        break
                # 回退到体积数据
                if use_volume_fallback:
                    _, gt_vol_path = find_volume_data(exp, project_root)
                    if gt_vol_path:
                        gt_img = load_volume_slice(gt_vol_path, slice_idx, slice_axis)
                        if gt_img is not None:
                            gt_images[organ] = gt_img
                            data_source = "volume"
                            break

        for c, method in enumerate(METHOD_ORDER):
            ax = axes[r][c]
            ax.axis("off")

            # 列标题
            if r == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=11, fontweight="bold", fontproperties=_FONT_PROP)

            # 行标签
            if c == 0:
                ax.text(
                    -0.15, 0.5,
                    ORGAN_LABELS.get(organ, organ),
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                    fontproperties=_FONT_PROP,
                )

            # 加载图像
            img = None
            if method == "gt":
                img = gt_images.get(organ)
            else:
                if method in organ_exps:
                    exp = organ_exps[method]
                    # 优先尝试渲染图像
                    pred_path, _ = find_render_images(exp, project_root)
                    if pred_path:
                        img = load_image(pred_path)
                    # 回退到体积数据
                    if img is None and use_volume_fallback:
                        pred_vol_path, _ = find_volume_data(exp, project_root)
                        if pred_vol_path:
                            img = load_volume_slice(pred_vol_path, slice_idx, slice_axis)
                    # 加载指标
                    psnr, ssim = load_eval_metrics(exp, project_root)
                    exp.psnr = psnr
                    exp.ssim = ssim

            # 显示图像
            if img is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                       transform=ax.transAxes, fontsize=14, color="#666666")
                ax.set_facecolor("#f0f0f0")
                continue

            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)

            # ROI 标注
            if show_roi and method != "gt":
                roi_adj = scale_roi(roi, img.shape, base_size=128)
                rect = Rectangle(
                    (roi_adj[2], roi_adj[0]),
                    roi_adj[3] - roi_adj[2],
                    roi_adj[1] - roi_adj[0],
                    linewidth=1.5,
                    edgecolor=METHOD_COLORS.get(method, "#FF0000"),
                    facecolor="none",
                )
                ax.add_patch(rect)

            # SPAGS 高亮边框
            if method == "spags":
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.5)
                    spine.set_edgecolor("#E91E63")

            # 指标标注（使用论文表格数据）
            if show_metrics and method not in ["gt"]:
                paper_psnr = PAPER_PSNR.get(views, {}).get(method, {}).get(organ)
                if paper_psnr is not None:
                    metric_text = f"{paper_psnr:.2f} dB"
                    ax.text(
                        0.03, 0.06,
                        metric_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        color="#FFD54F",
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
                    )

    # 图标题（已禁用）
    # fig.suptitle(
    #     f"定性对比: {views}视角稀疏新视图合成",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=0.99,
    #     fontproperties=_FONT_PROP,
    # )

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"已保存: {output_path}")

    # 同时保存 PDF
    if output_path.suffix.lower() != ".pdf":
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"已保存: {pdf_path}")

    plt.close(fig)


def save_config_report(
    experiments: Dict[str, Dict[str, ExperimentInfo]],
    output_path: Path,
    views: int,
):
    """保存配置报告"""
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "views": views,
        "organs": ORGAN_ORDER,
        "methods": METHOD_ORDER,
        "experiments": {},
    }

    for organ, methods in experiments.items():
        report["experiments"][organ] = {}
        for method, exp in methods.items():
            report["experiments"][organ][method] = {
                "model_path": str(exp.model_path),
                "iteration": exp.iteration,
                "psnr": exp.psnr,
                "ssim": exp.ssim,
            }

    report_path = output_path.with_suffix(".yml")
    with open(report_path, "w") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)
    print(f"已保存配置: {report_path}")


# =======================
# 主函数
# =======================

def main():
    parser = argparse.ArgumentParser(description="Fig 4-5: 定性对比实验可视化")
    parser.add_argument("--views", type=int, default=3, help="视角数（默认 3）")
    parser.add_argument("--output", type=str,
                       default="cc-agent/figures/fig4_5_qualitative_3views.png",
                       help="输出路径")
    parser.add_argument("--dpi", type=int, default=200, help="输出 DPI（默认 200）")
    parser.add_argument("--no_roi", action="store_true", help="禁用 ROI 标注")
    parser.add_argument("--no_metrics", action="store_true", help="禁用指标标注")
    parser.add_argument("--paths_md", type=str,
                       default="logs/latest_paths_15x6.md",
                       help="实验路径清单文件")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / args.output

    print("=" * 60)
    print("Fig 4-5: 定性对比实验可视化")
    print(f"视角数: {args.views}")
    print(f"输出路径: {output_path}")
    print("=" * 60)

    # 解析实验路径
    paths_md = project_root / args.paths_md
    experiments = parse_paths_md(paths_md, args.views)

    if not experiments:
        print("[错误] 未找到符合条件的实验。请检查路径清单文件。")
        return 1

    # 打印找到的实验
    print("\n找到的实验:")
    for organ, methods in experiments.items():
        print(f"  {organ}: {list(methods.keys())}")

    # 绘制
    plot_qualitative_comparison(
        experiments=experiments,
        project_root=project_root,
        output_path=output_path,
        views=args.views,
        dpi=args.dpi,
        show_roi=not args.no_roi,
        show_metrics=not args.no_metrics,
    )

    # 保存配置
    save_config_report(experiments, output_path, args.views)

    print("\n完成!")
    return 0


if __name__ == "__main__":
    exit(main())
