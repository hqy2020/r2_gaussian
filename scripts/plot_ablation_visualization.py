#!/usr/bin/env python3
"""
Fig 4-6: SPAGS 消融实验可视化（真实实验输出）

功能：
1) 上半部分：2 行 × 7 列网格对比（GT / Baseline / +SPS / +GAR / +ADM / +SPS+ADM / Full）
2) 下半部分：柱状图展示相对 Baseline 的 PSNR 提升，并标注协同增益（Synergy）

说明：
- 图像来自 `output/` 中实验目录保存的 `point_cloud/iteration_*/vol_{gt,pred}.npy`
- 默认指标使用 `eval/iter_*/eval2d_render_test.yml` 的 `psnr_2d`（与论文 NVS 指标一致）
- 如需让柱状图严格对齐论文表 4-4，可传入 `--bar_table_tex cc-agent/论文/4-method2-data.tex`
- 若某个配置缺失，将在图中显示 N/A，并在终端给出提示

用法示例：
    python3 scripts/plot_ablation_visualization.py \\
        --views 3 \\
        --organs foot abdomen \\
        --dataset_root data/369 \\
        --output figures/fig4_6_ablation_3views.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


ORGAN_LABELS = {
    "foot": "Foot",
    "abdomen": "Abdomen",
    "chest": "Chest",
    "head": "Head",
    "pancreas": "Pancreas",
}

COLUMN_ORDER = ["gt", "baseline", "sps", "gar", "adm", "sps_adm", "spags"]
COLUMN_LABELS = {
    "gt": "Ground Truth",
    "baseline": "Baseline",
    "sps": "+SPS",
    "gar": "+GAR",
    "adm": "+ADM",
    "sps_adm": "+SPS+ADM",
    "spags": "Full SPAGS",
}

ROI_CONFIGS = {
    # 基于 128×128 的经验 ROI，会按实际图像尺寸缩放
    "foot": {"roi": [60, 110, 60, 110], "slice_idx": 64},
    "abdomen": {"roi": [80, 130, 60, 110], "slice_idx": 64},
    "chest": {"roi": [80, 130, 80, 130], "slice_idx": 64},
    "head": {"roi": [70, 120, 70, 120], "slice_idx": 64},
    "pancreas": {"roi": [70, 120, 60, 110], "slice_idx": 64},
}

BOX_COLORS = {
    "sps": "#2E7DFF",  # 蓝
    "gar": "#2ECC71",  # 绿
    "adm": "#FF9F1C",  # 橙
    "gt": "#2ECC71",  # GT 默认绿框（示例）
}


@dataclass(frozen=True)
class ExperimentRecord:
    exp_dir: Path
    organ: str
    views: int
    config: str
    iterations: int
    vol_iter: Optional[int]
    vol_pred_path: Optional[Path]
    vol_gt_path: Optional[Path]
    psnr: Optional[float]
    ssim: Optional[float]
    metric_iter: Optional[int]


def _safe_load_yaml(path: Path) -> Dict:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        return {}


def _parse_source_path(source_path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    从 source_path（如 data/369/foot_50_3views.pickle）解析 organ/views。
    """
    if not source_path:
        return None, None

    name = Path(source_path).name
    parts = name.split("_")
    if not parts:
        return None, None

    organ = parts[0]
    views = None
    for p in parts:
        if "views" in p:
            digits = "".join(ch for ch in p if ch.isdigit())
            if digits:
                try:
                    views = int(digits)
                    break
                except Exception:
                    pass
    return organ, views


def _classify_config(cfg: Dict) -> str:
    """
    基于 cfg_args.yml 中的开关判断配置类别。

    注意：
    - SPS 在本仓库实验中通常表现为 `ply_path` 非空（使用密度加权/先验播种 init）
    - GAR：`enable_fsgs_proximity: true`
    - ADM：`enable_kplanes: true`
    """
    ply_path = str(cfg.get("ply_path", "") or "")
    sps = bool(ply_path.strip())
    gar = bool(cfg.get("enable_fsgs_proximity", False))
    adm = bool(cfg.get("enable_kplanes", False))

    if sps and gar and adm:
        return "spags"
    if sps and adm and not gar:
        return "sps_adm"
    if sps and gar and not adm:
        return "sps_gar"
    if gar and adm and not sps:
        return "gar_adm"
    if sps and not gar and not adm:
        return "sps"
    if gar and not sps and not adm:
        return "gar"
    if adm and not sps and not gar:
        return "adm"
    return "baseline"


def _find_latest_iter_dir(parent: Path, prefix: str = "iteration_") -> Optional[Path]:
    if not parent.exists():
        return None
    best = None
    best_iter = -1
    for p in parent.glob(f"{prefix}*"):
        if not p.is_dir():
            continue
        try:
            it = int(p.name.replace(prefix, ""))
        except Exception:
            continue
        if it > best_iter:
            best_iter = it
            best = p
    return best


def _find_volume_paths(exp_dir: Path, prefer_iter: int) -> Tuple[Optional[int], Optional[Path], Optional[Path]]:
    """
    返回 (vol_iter, vol_pred_path, vol_gt_path)。
    """
    # 优先 point_cloud/iteration_XXXXXX
    pc_dir = exp_dir / "point_cloud"
    if pc_dir.exists():
        prefer = pc_dir / f"iteration_{prefer_iter}"
        if (prefer / "vol_pred.npy").exists() and (prefer / "vol_gt.npy").exists():
            return prefer_iter, prefer / "vol_pred.npy", prefer / "vol_gt.npy"

        latest = _find_latest_iter_dir(pc_dir, prefix="iteration_")
        if latest and (latest / "vol_pred.npy").exists() and (latest / "vol_gt.npy").exists():
            try:
                it = int(latest.name.replace("iteration_", ""))
            except Exception:
                it = None
            return it, latest / "vol_pred.npy", latest / "vol_gt.npy"

    # 兼容 volume/ 结构（少数脚本产物）
    vol_dir = exp_dir / "volume"
    if (vol_dir / "vol_pred.npy").exists() and (vol_dir / "vol_gt.npy").exists():
        return None, vol_dir / "vol_pred.npy", vol_dir / "vol_gt.npy"

    return None, None, None


def _find_eval3d(exp_dir: Path, prefer_iter: int) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    返回 (eval_iter, psnr_3d, ssim_3d)。
    """
    eval_dir = exp_dir / "eval"
    if not eval_dir.exists():
        return None, None, None

    prefer = eval_dir / f"iter_{prefer_iter:06d}" / "eval3d.yml"
    if prefer.exists():
        data = _safe_load_yaml(prefer)
        return prefer_iter, data.get("psnr_3d"), data.get("ssim_3d")

    # fallback：找最新迭代
    latest_dir = _find_latest_iter_dir(eval_dir, prefix="iter_")
    if latest_dir:
        eval3d = latest_dir / "eval3d.yml"
        if eval3d.exists():
            data = _safe_load_yaml(eval3d)
            try:
                it = int(latest_dir.name.replace("iter_", ""))
            except Exception:
                it = None
            return it, data.get("psnr_3d"), data.get("ssim_3d")

    return None, None, None


def _find_eval2d(exp_dir: Path, prefer_iter: int, split: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    返回 (eval_iter, psnr_2d, ssim_2d)。

    split:
        - "test": eval2d_render_test.yml
        - "train": eval2d_render_train.yml
    """
    eval_dir = exp_dir / "eval"
    if not eval_dir.exists():
        return None, None, None

    filename = f"eval2d_render_{split}.yml"
    prefer = eval_dir / f"iter_{prefer_iter:06d}" / filename
    if prefer.exists():
        data = _safe_load_yaml(prefer)
        return prefer_iter, data.get("psnr_2d"), data.get("ssim_2d")

    latest_dir = _find_latest_iter_dir(eval_dir, prefix="iter_")
    if latest_dir:
        eval2d = latest_dir / filename
        if eval2d.exists():
            data = _safe_load_yaml(eval2d)
            try:
                it = int(latest_dir.name.replace("iter_", ""))
            except Exception:
                it = None
            return it, data.get("psnr_2d"), data.get("ssim_2d")

    return None, None, None


def _match_dataset_root(source_path: str, dataset_root: str) -> bool:
    if not dataset_root:
        return True
    root = dataset_root.rstrip("/") + "/"
    return source_path.startswith(root)


def _path_allowed(path: Path, include_regex: Optional[str], exclude_regex: Optional[str]) -> bool:
    s = str(path)
    if include_regex:
        if re.search(include_regex, s) is None:
            return False
    if exclude_regex:
        if re.search(exclude_regex, s) is not None:
            return False
    return True


def load_ablation_table_from_tex(tex_path: Path) -> Dict[str, Dict[str, float]]:
    """
    解析 `cc-agent/论文/4-method2-data.tex` 中的 “=== 消融实验数据 ===” CSV 注释段。

    返回:
        dict: {config_name: {"avg": float, "delta": float}, ...}

    config_name 使用内部 key：
        baseline / sps / gar / adm / sps_adm / spags
    """
    if not tex_path.exists():
        return {}

    text = tex_path.read_text(errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    start = None
    for i, ln in enumerate(lines):
        if "=== 消融实验数据" in ln:
            start = i
            break
    if start is None:
        return {}

    mapping = {
        "Baseline": "baseline",
        "+SPS": "sps",
        "+GAR": "gar",
        "+ADM": "adm",
        "+SPS+ADM": "sps_adm",
        "Full": "spags",
        "完整": "spags",
    }

    table: Dict[str, Dict[str, float]] = {}
    for ln in lines[start : start + 50]:
        if not ln.startswith("%"):
            continue
        content = ln.lstrip("%").strip()
        if not content or content.startswith("Config,"):
            continue

        parts = [p.strip() for p in content.split(",")]
        if len(parts) < 8:
            continue

        key = mapping.get(parts[0])
        if key is None:
            continue

        try:
            avg = float(parts[6])
            delta = float(parts[7])
        except Exception:
            continue

        table[key] = {"avg": avg, "delta": delta}

    return table


def build_experiment_index(
    output_dir: Path,
    dataset_root: str,
    views_filter: Optional[int],
    prefer_iter: int,
    metric: str,
    metric_split: str,
    include_path_regex: Optional[str] = None,
    exclude_path_regex: Optional[str] = None,
) -> List[ExperimentRecord]:
    records: List[ExperimentRecord] = []

    for cfg_path in output_dir.rglob("cfg_args.yml"):
        exp_dir = cfg_path.parent
        if not _path_allowed(exp_dir, include_path_regex, exclude_path_regex):
            continue
        cfg = _safe_load_yaml(cfg_path)

        method = cfg.get("method")
        if method not in (None, "r2_gaussian"):
            continue

        source_path = str(cfg.get("source_path", "") or "")
        if dataset_root and not _match_dataset_root(source_path, dataset_root):
            continue

        organ, views = _parse_source_path(source_path)
        if organ is None or views is None:
            continue
        if views_filter is not None and views != views_filter:
            continue

        config = _classify_config(cfg)
        iterations = int(cfg.get("iterations") or 0)

        vol_iter, vol_pred_path, vol_gt_path = _find_volume_paths(exp_dir, prefer_iter=prefer_iter)
        if metric == "2d":
            metric_iter, psnr, ssim = _find_eval2d(exp_dir, prefer_iter=prefer_iter, split=metric_split)
        else:
            metric_iter, psnr, ssim = _find_eval3d(exp_dir, prefer_iter=prefer_iter)

        records.append(
            ExperimentRecord(
                exp_dir=exp_dir,
                organ=organ,
                views=views,
                config=config,
                iterations=iterations,
                vol_iter=vol_iter,
                vol_pred_path=vol_pred_path,
                vol_gt_path=vol_gt_path,
                psnr=float(psnr) if isinstance(psnr, (int, float)) else None,
                ssim=float(ssim) if isinstance(ssim, (int, float)) else None,
                metric_iter=metric_iter if isinstance(metric_iter, int) else None,
            )
        )

    return records


def _pick_best(records: Iterable[ExperimentRecord]) -> Optional[ExperimentRecord]:
    best = None
    best_score = None
    for r in records:
        has_vol = int(r.vol_pred_path is not None and r.vol_gt_path is not None)
        vol_it = r.vol_iter or 0
        psnr = r.psnr if r.psnr is not None else -1e9
        mtime = r.exp_dir.stat().st_mtime
        score = (has_vol, vol_it, r.iterations, psnr, mtime)
        if best_score is None or score > best_score:
            best_score = score
            best = r
    return best


def select_record(
    index: List[ExperimentRecord],
    organ: str,
    views: int,
    config: str,
    required_iter: Optional[int] = None,
    strict_iter: bool = True,
    require_volume: bool = False,
    require_metric: bool = False,
) -> Optional[ExperimentRecord]:
    matched = [r for r in index if r.organ == organ and r.views == views and r.config == config]
    if strict_iter and required_iter is not None:
        matched = [r for r in matched if r.iterations >= required_iter]
        if require_volume:
            matched = [
                r
                for r in matched
                if r.vol_iter == required_iter and r.vol_pred_path is not None and r.vol_gt_path is not None
            ]
        if require_metric:
            matched = [
                r
                for r in matched
                if r.metric_iter == required_iter and r.psnr is not None
            ]
    return _pick_best(matched)


def _scale_roi(roi_128: List[int], img_shape: Tuple[int, int], base: int = 128) -> List[int]:
    h, w = img_shape
    return [
        int(roi_128[0] * h / base),
        int(roi_128[1] * h / base),
        int(roi_128[2] * w / base),
        int(roi_128[3] * w / base),
    ]


def _get_slice(vol: np.ndarray, slice_idx: int, axis: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(vol[slice_idx, :, :])
    if axis == 1:
        return np.asarray(vol[:, slice_idx, :])
    return np.asarray(vol[:, :, slice_idx])


def load_normalized_slice(
    npy_path: Path,
    slice_idx: int,
    axis: int,
    norm_vmin: float,
    norm_vmax: float,
) -> np.ndarray:
    vol = np.load(npy_path, mmap_mode="r")
    if slice_idx < 0 or slice_idx >= vol.shape[axis]:
        slice_idx = vol.shape[axis] // 2
    img = _get_slice(vol, slice_idx=slice_idx, axis=axis).astype(np.float32, copy=False)
    if norm_vmax > norm_vmin:
        img = (img - norm_vmin) / (norm_vmax - norm_vmin)
    else:
        img = img * 0.0
    return np.clip(img, 0.0, 1.0)


def _compute_norm_range(gt_slice: np.ndarray) -> Tuple[float, float]:
    # 用分位数增强鲁棒性，避免少数异常值导致整体发黑/发白
    vmin, vmax = np.percentile(gt_slice.astype(np.float32, copy=False), [1.0, 99.0])
    if vmax <= vmin:
        vmin = float(gt_slice.min())
        vmax = float(gt_slice.max())
    return float(vmin), float(vmax)


def plot_ablation_figure(
    index: List[ExperimentRecord],
    organs: List[str],
    bar_organs: List[str],
    views: int,
    axis: int,
    prefer_iter: int,
    strict_iter: bool,
    output_path: Path,
    dpi: int,
    report_path: Optional[Path] = None,
    bar_table: Optional[Dict[str, Dict[str, float]]] = None,
):
    n_rows = len(organs)
    n_cols = len(COLUMN_ORDER)

    fig_w = n_cols * 2.4
    fig_h = n_rows * 2.4 + 3.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows=n_rows + 1, ncols=n_cols, height_ratios=[1] * n_rows + [0.9], hspace=0.12, wspace=0.02)

    grid_axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    ax_bar = fig.add_subplot(gs[n_rows, :])

    # 预先拿 baseline / full，用于行标签与 delta
    baseline_psnr: Dict[str, Optional[float]] = {}
    full_delta: Dict[str, Optional[float]] = {}
    for organ in organs:
        base = select_record(
            index,
            organ,
            views,
            "baseline",
            required_iter=prefer_iter,
            strict_iter=strict_iter,
            require_volume=False,
            require_metric=True,
        )
        full = select_record(
            index,
            organ,
            views,
            "spags",
            required_iter=prefer_iter,
            strict_iter=strict_iter,
            require_volume=False,
            require_metric=True,
        )
        baseline_psnr[organ] = base.psnr if base else None
        if base and full and base.psnr is not None and full.psnr is not None:
            full_delta[organ] = full.psnr - base.psnr
        else:
            full_delta[organ] = None

    # 绘制网格
    selected: Dict[str, Dict[str, Optional[Dict]]] = {}
    for r, organ in enumerate(organs):
        selected.setdefault(organ, {})
        roi_cfg = ROI_CONFIGS.get(organ, {"roi": [60, 110, 60, 110], "slice_idx": 64})
        slice_idx = int(roi_cfg.get("slice_idx", 64))
        roi_128 = list(roi_cfg.get("roi", [60, 110, 60, 110]))

        # GT：优先用 baseline 或 spags 的 vol_gt
        gt_src = select_record(
            index,
            organ,
            views,
            "baseline",
            required_iter=prefer_iter,
            strict_iter=strict_iter,
            require_volume=True,
            require_metric=False,
        ) or select_record(
            index,
            organ,
            views,
            "spags",
            required_iter=prefer_iter,
            strict_iter=strict_iter,
            require_volume=True,
            require_metric=False,
        )
        gt_img_norm = None
        if gt_src and gt_src.vol_gt_path is not None:
            gt_vol = np.load(gt_src.vol_gt_path, mmap_mode="r")
            if slice_idx < 0 or slice_idx >= gt_vol.shape[axis]:
                slice_idx = gt_vol.shape[axis] // 2
            gt_slice = _get_slice(gt_vol, slice_idx=slice_idx, axis=axis)
            vmin, vmax = _compute_norm_range(gt_slice)
            gt_img_norm = load_normalized_slice(gt_src.vol_gt_path, slice_idx, axis, vmin, vmax)
        else:
            vmin, vmax = 0.0, 1.0

        for c, cfg in enumerate(COLUMN_ORDER):
            ax = grid_axes[r][c]
            ax.axis("off")

            if r == 0:
                ax.set_title(COLUMN_LABELS.get(cfg, cfg), fontsize=11, fontweight="bold")

            if c == 0:
                delta = full_delta.get(organ)
                delta_text = f" ({delta:+.2f} dB)" if isinstance(delta, (int, float)) else ""
                ax.text(
                    -0.12,
                    0.5,
                    f"{ORGAN_LABELS.get(organ, organ)}{delta_text}",
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

            if cfg == "gt":
                selected[organ][cfg] = {
                    "exp_dir": str(gt_src.exp_dir) if gt_src else None,
                    "vol_gt": str(gt_src.vol_gt_path) if (gt_src and gt_src.vol_gt_path) else None,
                    "slice_idx": int(slice_idx),
                    "axis": int(axis),
                }
                if gt_img_norm is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                    ax.set_facecolor("#f0f0f0")
                    continue
                ax.imshow(gt_img_norm, cmap="gray", vmin=0.0, vmax=1.0)
                roi_adj = _scale_roi(roi_128, gt_img_norm.shape)
                rect = Rectangle(
                    (roi_adj[2], roi_adj[0]),
                    roi_adj[3] - roi_adj[2],
                    roi_adj[1] - roi_adj[0],
                    linewidth=1.5,
                    edgecolor=BOX_COLORS["gt"],
                    facecolor="none",
                )
                ax.add_patch(rect)
                continue

            rec = select_record(
                index,
                organ,
                views,
                cfg,
                required_iter=prefer_iter,
                strict_iter=strict_iter,
                require_volume=True,
                require_metric=True,
            )
            selected[organ][cfg] = {
                "exp_dir": str(rec.exp_dir) if rec else None,
                "vol_pred": str(rec.vol_pred_path) if (rec and rec.vol_pred_path) else None,
                "vol_gt": str(rec.vol_gt_path) if (rec and rec.vol_gt_path) else None,
                "metric_psnr": rec.psnr if rec else None,
                "metric_ssim": rec.ssim if rec else None,
                "metric_iter": rec.metric_iter if rec else None,
                "vol_iter": rec.vol_iter if rec else None,
                "slice_idx": int(slice_idx),
                "axis": int(axis),
            }
            if rec is None or rec.vol_pred_path is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#f0f0f0")
                continue

            img_norm = load_normalized_slice(rec.vol_pred_path, slice_idx, axis, vmin, vmax)
            ax.imshow(img_norm, cmap="gray", vmin=0.0, vmax=1.0)

            # ROI 框（按组件颜色区分）
            if cfg in BOX_COLORS:
                roi_adj = _scale_roi(roi_128, img_norm.shape)
                rect = Rectangle(
                    (roi_adj[2], roi_adj[0]),
                    roi_adj[3] - roi_adj[2],
                    roi_adj[1] - roi_adj[0],
                    linewidth=1.5,
                    edgecolor=BOX_COLORS[cfg],
                    facecolor="none",
                )
                ax.add_patch(rect)

            # Full SPAGS 绿色边框高亮
            if cfg == "spags":
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.0)
                    spine.set_edgecolor("#2ECC71")

            # 指标标注（PSNR 与 Δ）
            psnr = rec.psnr
            base_psnr = baseline_psnr.get(organ)
            delta = (psnr - base_psnr) if (psnr is not None and base_psnr is not None) else None
            if psnr is not None:
                delta_str = f"{delta:+.2f} dB" if delta is not None else "N/A"
                ax.text(
                    0.03,
                    0.06,
                    f"{psnr:.2f} dB\n({delta_str})",
                    transform=ax.transAxes,
                    fontsize=8.5,
                    color="#FFD54F",
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.55),
                )

            # 如果允许 fallback 且不是 prefer_iter，标注实际迭代（避免误读）
            if not strict_iter and prefer_iter is not None:
                chosen_iter = rec.metric_iter or rec.vol_iter
                if chosen_iter is not None and int(chosen_iter) != int(prefer_iter):
                    ax.text(
                        0.97,
                        0.97,
                        f"iter={int(chosen_iter)}",
                        transform=ax.transAxes,
                        fontsize=7.5,
                        color="white",
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.45),
                    )

    # =======================
    # 柱状图：PSNR 提升 & 协同增益
    # =======================
    bar_configs = ["baseline", "sps", "gar", "adm", "sps_adm", "spags"]
    bar_labels = [COLUMN_LABELS.get(c, c) for c in bar_configs]

    # 计算平均提升（默认对 5 个器官求均值，可通过 bar_organs 控制）
    def mean_improve(cfg: str) -> Optional[float]:
        if cfg == "baseline":
            return 0.0
        vals = []
        for organ in bar_organs:
            base = select_record(
                index,
                organ,
                views,
                "baseline",
                required_iter=prefer_iter,
                strict_iter=strict_iter,
                require_volume=False,
                require_metric=True,
            )
            rec = select_record(
                index,
                organ,
                views,
                cfg,
                required_iter=prefer_iter,
                strict_iter=strict_iter,
                require_volume=False,
                require_metric=True,
            )
            if base and rec and base.psnr is not None and rec.psnr is not None:
                vals.append(rec.psnr - base.psnr)
        if not vals:
            return None
        return float(np.mean(vals))

    if bar_table:
        improves = [0.0 if c == "baseline" else bar_table.get(c, {}).get("delta") for c in bar_configs]
    else:
        improves = [mean_improve(c) for c in bar_configs]

    colors = {
        "baseline": "#BDBDBD",
        "sps": "#2E7DFF",
        "gar": "#2ECC71",
        "adm": "#FF9F1C",
        "sps_adm": "#F7E967",
        "spags": "#E91E63",
    }

    x = np.arange(len(bar_configs))
    heights = [v if v is not None else 0.0 for v in improves]
    bar_colors = [colors.get(c, "#90A4AE") for c in bar_configs]

    # Full SPAGS：分段显示 “单独效果之和” + “协同增益”
    synergy_color = "#8E44AD"  # 紫色
    spags_idx = bar_configs.index("spags")
    spags_val = improves[spags_idx]
    sps_val = improves[bar_configs.index("sps")]
    gar_val = improves[bar_configs.index("gar")]
    adm_val = improves[bar_configs.index("adm")]

    can_synergy = all(v is not None for v in [spags_val, sps_val, gar_val, adm_val])
    if can_synergy:
        expected = float(sps_val + gar_val + adm_val)
        synergy = float(spags_val - expected)
        base_part = expected
        synergy_part = synergy

        # 先画除 spags 外的 bar
        for i, cfg in enumerate(bar_configs):
            if i == spags_idx:
                continue
            ax_bar.bar(i, heights[i], color=bar_colors[i], edgecolor="#333333", linewidth=0.8)

        # 再画 spags 分段 bar（允许 synergy 为负）
        ax_bar.bar(spags_idx, base_part, color=bar_colors[spags_idx], edgecolor="#333333", linewidth=0.8, label="Full SPAGS")
        ax_bar.bar(spags_idx, synergy_part, bottom=base_part, color=synergy_color, edgecolor="#333333", linewidth=0.8, label="Synergy")
    else:
        ax_bar.bar(x, heights, color=bar_colors, edgecolor="#333333", linewidth=0.8)
        synergy = None

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels, rotation=15, ha="right")
    ax_bar.set_ylabel("PSNR Improvement (dB)")
    ax_bar.set_title("Ablation Study: Component Contributions to PSNR", fontsize=12, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.25, linestyle="--")

    # 标注数值
    for i, val in enumerate(improves):
        if val is None:
            ax_bar.text(i, 0.02, "N/A", ha="center", va="bottom", fontsize=9, color="#616161")
            continue
        ax_bar.text(i, val + (0.02 if val >= 0 else -0.02), f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9, fontweight="bold")

    if can_synergy and synergy is not None:
        ax_bar.text(
            spags_idx,
            (spags_val if spags_val is not None else 0.0) + 0.08,
            f"Synergy: {synergy:+.2f} dB\n({spags_val:+.2f} > {expected:+.2f})",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )
        ax_bar.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax_bar.legend([], [], frameon=False)

    fig.suptitle(f"SPAGS Ablation Study ({views}-View Sparse CT Reconstruction)", fontsize=14, fontweight="bold", y=0.99)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"✓ 已保存: {output_path}")

    if output_path.suffix.lower() != ".png":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ 同时保存 PNG: {png_path}")

    plt.close(fig)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "views": int(views),
            "prefer_iter": int(prefer_iter),
            "strict_iter": bool(strict_iter),
            "organs_grid": list(organs),
            "organs_bar": list(bar_organs),
            "selected": selected,
        }
        with open(report_path, "w") as f:
            yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)
        print(f"✓ 已保存选择清单: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="SPAGS 消融实验可视化（Fig 4-6）")
    parser.add_argument("--output_dir", type=str, default="output", help="实验输出目录（默认 output/）")
    parser.add_argument("--dataset_root", type=str, default="data/369", help="仅使用该数据根目录下的实验（默认 data/369）")
    parser.add_argument("--views", type=int, default=3, help="视角数（默认 3）")
    parser.add_argument("--organs", nargs="+", default=["foot", "abdomen"], help="器官列表（默认 foot abdomen）")
    parser.add_argument("--bar_organs", nargs="+", default=["chest", "foot", "head", "abdomen", "pancreas"], help="柱状图统计的器官列表（默认 5 个器官）")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="切片轴（默认 2，即 z 轴）")
    parser.add_argument("--prefer_iter", type=int, default=30000, help="优先使用的迭代（默认 30000）")
    parser.add_argument("--allow_fallback_iter", action="store_true", default=False, help="允许使用非 prefer_iter 的最近迭代（默认关闭，缺失则 N/A）")
    parser.add_argument("--metric", type=str, default="2d", choices=["2d", "3d"], help="使用 2D 或 3D 指标（默认 2d）")
    parser.add_argument("--metric_split", type=str, default="test", choices=["test", "train"], help="2D 指标选择 test/train（默认 test）")
    parser.add_argument("--include_path_regex", type=str, default=None, help="仅使用路径匹配该正则的实验（可选）")
    parser.add_argument("--exclude_path_regex", type=str, default=None, help="排除路径匹配该正则的实验（可选）")
    parser.add_argument("--report", type=str, default=None, help="保存所选实验清单（yml，可选）")
    parser.add_argument("--gpu", type=int, default=0, help="缺失实验时打印复现实验命令用的 GPU id（默认 0）")
    parser.add_argument("--bar_table_tex", type=str, default=None, help="从该 tex 文件读取柱状图数值（可选，推荐 cc-agent/论文/4-method2-data.tex）")
    parser.add_argument("--dpi", type=int, default=250, help="输出 dpi（默认 250）")
    parser.add_argument("--output", type=str, default="figures/fig4_6_ablation.png", help="输出路径（png/pdf）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = Path(args.output)
    strict_iter = not args.allow_fallback_iter

    index = build_experiment_index(
        output_dir=output_dir,
        dataset_root=args.dataset_root,
        views_filter=args.views,
        prefer_iter=args.prefer_iter,
        metric=args.metric,
        metric_split=args.metric_split,
        include_path_regex=args.include_path_regex,
        exclude_path_regex=args.exclude_path_regex,
    )
    if not index:
        raise SystemExit(f"未找到符合条件的实验（output_dir={output_dir}, dataset_root={args.dataset_root}）。")

    # 打印缺失配置提示（针对网格列）
    missing = []
    for organ in args.organs:
        for cfg in COLUMN_ORDER:
            if cfg == "gt":
                continue
            if select_record(
                index,
                organ,
                args.views,
                cfg,
                required_iter=args.prefer_iter,
                strict_iter=strict_iter,
                require_volume=(cfg != "gt"),
                require_metric=(cfg != "gt"),
            ) is None:
                missing.append((organ, cfg))
    if missing:
        print("注意：以下配置未找到对应实验，将在图中显示 N/A：")
        for organ, cfg in missing:
            print(f"  - {organ} / {cfg}")
        print("\n建议复现实验命令（按需运行）：")
        printed = set()
        for organ, cfg in missing:
            if (organ, cfg) in printed:
                continue
            printed.add((organ, cfg))
            print(f"  ./cc-agent/scripts/run_spags_ablation.sh {cfg} {organ} {args.views} {args.gpu}")

    bar_table = load_ablation_table_from_tex(Path(args.bar_table_tex)) if args.bar_table_tex else None
    if args.bar_table_tex and not bar_table:
        print(f"注意：未能从 {args.bar_table_tex} 解析到消融表格，将回退为从 output/ 统计。")

    plot_ablation_figure(
        index=index,
        organs=args.organs,
        bar_organs=args.bar_organs,
        views=args.views,
        axis=args.axis,
        prefer_iter=args.prefer_iter,
        strict_iter=strict_iter,
        output_path=output_path,
        dpi=args.dpi,
        report_path=Path(args.report) if args.report else None,
        bar_table=bar_table,
    )


if __name__ == "__main__":
    main()
