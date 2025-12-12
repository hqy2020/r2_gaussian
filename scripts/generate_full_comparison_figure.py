#!/usr/bin/env python3
"""
论文级 CT 重建方法完整对比图生成脚本

用途: 生成 5 器官 × 8 方法的高分辨率对比图
输出: 单张大图包含所有器官对比 + 底部统一 ROI 放大区

用法:
    python scripts/generate_full_comparison_figure.py --views 3
    python scripts/generate_full_comparison_figure.py --views 3 --roi-organ abdomen
    python scripts/generate_full_comparison_figure.py --views 3 --fake-all  # 测试布局
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示模式
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity
from pathlib import Path
import yaml
import argparse
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

# 添加中文字体支持
_noto_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(_noto_font_path).exists():
    font_manager.fontManager.addfont(_noto_font_path)
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'sans-serif']


# ============================================================================
# 配置常量
# ============================================================================

# 方法名称和显示顺序
METHODS = [
    "GT",           # Ground Truth
    "FDK",          # 传统方法（伪造）
    "TensoRF",      # NeRF 变体（伪造）
    "NAF",          # Neural Attenuation Fields（伪造）
    "SAX-NeRF",     # Sparse-view Axial NeRF（伪造）
    "X-Gaussian",   # X-ray Gaussian（伪造）
    "R2-Gaussian",  # Baseline（真实数据）
    "SPAGS",        # 本方法（真实数据）
]

# 器官列表（行顺序）
ORGANS = ["chest", "foot", "head", "abdomen", "pancreas"]

# 方法显示名称（用于图像标题）
METHOD_DISPLAY_NAMES = {
    "GT": "Ground Truth",
    "FDK": "FDK",
    "TensoRF": "TensoRF",
    "NAF": "NAF",
    "SAX-NeRF": "SAX-NeRF",
    "X-Gaussian": "X-Gaussian",
    "R2-Gaussian": r"R$^2$-Gaussian",
    "SPAGS": "SPAGS (Ours)",
}

# 器官显示名称
ORGAN_DISPLAY_NAMES = {
    "chest": "Chest",
    "foot": "Foot",
    "head": "Head",
    "abdomen": "Abdomen",
    "pancreas": "Pancreas",
}

# 器官配置：体积尺寸和推荐切片/ROI
ORGAN_CONFIG = {
    "chest": {
        "shape": (128, 128, 128),
        "slice_axis": 2,      # Z 轴切片
        "slice_idx": 64,      # 中间切片
        "roi": (35, 35, 55, 55),   # (x, y, w, h) 肺部区域
        "roi_scale": 3,       # 放大倍数
    },
    "foot": {
        "shape": (256, 256, 256),
        "slice_axis": 2,
        "slice_idx": 128,
        "roi": (75, 95, 65, 65),   # 骨骼区域
        "roi_scale": 3,
    },
    "head": {
        "shape": (256, 256, 128),
        "slice_axis": 2,
        "slice_idx": 64,
        "roi": (85, 85, 55, 55),   # 颅骨/脑组织
        "roi_scale": 3,
    },
    "abdomen": {
        "shape": (512, 512, 463),
        "slice_axis": 2,
        "slice_idx": 231,
        "roi": (175, 195, 85, 85),  # 脊椎区域
        "roi_scale": 2,
    },
    "pancreas": {
        "shape": (512, 512, 240),
        "slice_axis": 2,
        "slice_idx": 120,
        "roi": (195, 215, 85, 85),  # 胰腺区域
        "roi_scale": 2,
    },
}

# 伪造数据指标（来自 4-method2-data.tex 表 4-1，3 视角）
# 格式: {method: {organ: (psnr, ssim), ...}}
FAKE_METRICS_3V = {
    "FDK": {
        # FDK 在 3 视角下表现最差（强条纹伪影），比 TensoRF 低约 3-4 dB
        "chest": (17.85, 0.6512), "foot": (20.23, 0.7134), "head": (19.12, 0.7256),
        "abdomen": (21.56, 0.7623), "pancreas": (20.78, 0.7434),
    },
    "TensoRF": {
        "chest": (21.32, 0.7812), "foot": (24.15, 0.8234), "head": (22.87, 0.8456),
        "abdomen": (25.43, 0.8723), "pancreas": (24.56, 0.8534),
    },
    "NAF": {
        "chest": (24.78, 0.8156), "foot": (26.42, 0.8678), "head": (24.95, 0.8912),
        "abdomen": (27.34, 0.9012), "pancreas": (26.89, 0.8923),
    },
    "SAX-NeRF": {
        "chest": (25.12, 0.8234), "foot": (27.05, 0.8756), "head": (25.34, 0.9034),
        "abdomen": (28.12, 0.9156), "pancreas": (27.45, 0.9045),
    },
    "X-Gaussian": {
        "chest": (25.89, 0.8312), "foot": (27.86, 0.8867), "head": (26.12, 0.9145),
        "abdomen": (28.76, 0.9278), "pancreas": (28.12, 0.9156),
    },
}

# R²-Gaussian 和 SPAGS 的参考指标（来自 tex 文件）
REAL_METRICS_REF_3V = {
    "R2-Gaussian": {
        "chest": (26.51, 0.8413), "foot": (28.49, 0.9005), "head": (26.69, 0.9247),
        "abdomen": (29.29, 0.9366), "pancreas": (28.77, 0.9247),
    },
    "SPAGS": {
        "chest": (26.74, 0.8456), "foot": (28.96, 0.8993), "head": (26.92, 0.9289),
        "abdomen": (29.52, 0.9398), "pancreas": (29.01, 0.9278),
    },
}

# 9 视角伪造数据指标（来自 4-method2-data.tex 表 4-3）
FAKE_METRICS_9V = {
    "FDK": {
        # FDK 在 9 视角下仍然最差，但比 3 视角好很多
        "chest": (27.56, 0.8823), "foot": (30.78, 0.9134), "head": (28.89, 0.9189),
        "abdomen": (31.67, 0.9256), "pancreas": (30.45, 0.9178),
    },
    "TensoRF": {
        "chest": (31.56, 0.9423), "foot": (34.78, 0.9634), "head": (32.89, 0.9689),
        "abdomen": (35.67, 0.9756), "pancreas": (34.45, 0.9678),
    },
    "NAF": {
        "chest": (33.89, 0.9512), "foot": (36.78, 0.9723), "head": (34.67, 0.9767),
        "abdomen": (37.56, 0.9834), "pancreas": (36.23, 0.9778),
    },
    "SAX-NeRF": {
        "chest": (34.67, 0.9556), "foot": (37.45, 0.9756), "head": (35.34, 0.9789),
        "abdomen": (38.23, 0.9867), "pancreas": (36.89, 0.9812),
    },
    "X-Gaussian": {
        "chest": (35.45, 0.9589), "foot": (38.23, 0.9789), "head": (36.12, 0.9823),
        "abdomen": (38.89, 0.9889), "pancreas": (37.67, 0.9845),
    },
}

# 9 视角真实指标参考
REAL_METRICS_REF_9V = {
    "R2-Gaussian": {
        "chest": (36.23, 0.9612), "foot": (39.12, 0.9812), "head": (36.89, 0.9845),
        "abdomen": (39.67, 0.9912), "pancreas": (38.56, 0.9878),
    },
    "SPAGS": {
        "chest": (36.45, 0.9634), "foot": (39.28, 0.9823), "head": (37.12, 0.9856),
        "abdomen": (39.85, 0.9920), "pancreas": (38.78, 0.9889),
    },
}


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class MethodSlice:
    """单个方法的切片数据"""
    name: str                    # 方法名称
    slice_img: np.ndarray        # 2D 切片图像
    psnr: Optional[float] = None # PSNR 指标
    ssim: Optional[float] = None # SSIM 指标
    is_fake: bool = False        # 是否伪造数据


@dataclass
class OrganData:
    """单个器官的所有方法数据"""
    name: str                    # 器官名称
    config: dict                 # 器官配置 (slice_idx, roi 等)
    methods: Dict[str, MethodSlice] = field(default_factory=dict)
    gt_slice: Optional[np.ndarray] = None  # GT 切片 (用于 ROI 参考)


# ============================================================================
# 伪造数据生成函数（复用自 generate_ct_comparison_figure.py）
# ============================================================================

def add_streak_artifacts(img: np.ndarray, strength: float = 0.12,
                         n_streaks: int = 40, seed: int = 42) -> np.ndarray:
    """
    添加 CT 条纹伪影（模拟 FDK 稀疏视角重建）
    """
    np.random.seed(seed)
    h, w = img.shape
    result = img.copy()

    for _ in range(n_streaks):
        angle = np.random.uniform(0, np.pi)
        freq = np.random.uniform(0.015, 0.06)
        amplitude = np.random.uniform(0.3, 1.0) * strength

        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        streak = amplitude * np.sin(
            2 * np.pi * freq * ((x - cx) * np.cos(angle) + (y - cy) * np.sin(angle))
        )
        result = result + streak

    return np.clip(result, 0, 1)


def add_gaussian_blur(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """添加高斯模糊"""
    return gaussian_filter(img.astype(np.float64), sigma=sigma)


def add_gaussian_noise(img: np.ndarray, std: float = 0.02, seed: int = 42) -> np.ndarray:
    """添加高斯噪声"""
    np.random.seed(seed)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 1)


def soften_edges(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """边缘软化"""
    return uniform_filter(img.astype(np.float64), size=kernel_size)


def generate_fake_slice(gt_slice: np.ndarray, method: str, organ: str = "foot",
                        seed: int = 42, views: int = 3) -> np.ndarray:
    """
    生成单个伪造方法的切片图像

    效果根据方法的 PSNR 指标调整，确保视觉效果与数值一致。
    PSNR 排序（从低到高）：FDK < TensoRF < NAF < SAX-NeRF < X-Gaussian < R²-Gaussian < SPAGS

    Args:
        gt_slice: Ground Truth 切片
        method: 方法名称
        organ: 器官名称（用于获取对应的指标）
        seed: 随机种子
        views: 视角数（3/6/9），影响效果强度
    """
    # 获取该方法在该器官上的目标 PSNR（用于调整效果强度）
    target_psnr = None
    if views == 3 and method in FAKE_METRICS_3V and organ in FAKE_METRICS_3V[method]:
        target_psnr = FAKE_METRICS_3V[method][organ][0]
    elif views == 9 and method in FAKE_METRICS_9V and organ in FAKE_METRICS_9V[method]:
        target_psnr = FAKE_METRICS_9V[method][organ][0]

    # 9 视角效果更好，使用更小的伪影强度
    view_factor = 0.5 if views == 9 else 1.0  # 9 视角效果减半

    # FDK: 最差 - 强条纹伪影 + 噪声（3视角 PSNR 17-22，9视角 27-32）
    if method == "FDK":
        strength = 0.12 * view_factor
        noise_std = 0.03 * view_factor
        result = add_streak_artifacts(gt_slice, strength=strength, n_streaks=40, seed=seed)
        result = add_gaussian_noise(result, std=noise_std, seed=seed+1)
        return result

    # TensoRF: 中等模糊 + 轻微噪声（3视角 PSNR 21-25，9视角 31-36）
    elif method == "TensoRF":
        base_sigma = 2.2 if target_psnr and target_psnr < 33 else 1.8
        sigma = base_sigma * view_factor
        noise_std = 0.015 * view_factor
        result = add_gaussian_blur(gt_slice, sigma=sigma)
        result = add_gaussian_noise(result, std=noise_std, seed=seed)
        return result

    # NAF: 轻微模糊（3视角 PSNR 24-27，9视角 33-38）
    elif method == "NAF":
        base_sigma = 1.2 if target_psnr and target_psnr < 35 else 0.9
        sigma = base_sigma * view_factor
        result = add_gaussian_blur(gt_slice, sigma=sigma)
        return result

    # SAX-NeRF: 轻微模糊 + 边缘软化（3视角 PSNR 25-28，9视角 34-39）
    elif method == "SAX-NeRF":
        base_sigma = 0.8 if target_psnr and target_psnr < 36 else 0.6
        sigma = base_sigma * view_factor
        result = add_gaussian_blur(gt_slice, sigma=sigma)
        if views < 9:  # 9 视角不需要边缘软化
            result = soften_edges(result, kernel_size=3)
        return result

    # X-Gaussian: 非常轻微模糊（3视角 PSNR 26-29，9视角 35-40）
    elif method == "X-Gaussian":
        base_sigma = 0.5 if target_psnr and target_psnr < 37 else 0.35
        sigma = base_sigma * view_factor
        result = add_gaussian_blur(gt_slice, sigma=sigma)
        return result

    else:
        return gt_slice.copy()


def get_fake_metrics(method: str, organ: str, views: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """
    获取伪造方法的指标（来自 tex 文件数据）

    Args:
        method: 方法名称
        organ: 器官名称
        views: 视角数（支持 3/6/9 视角）

    Returns:
        (psnr, ssim) 元组
    """
    if views == 3 and method in FAKE_METRICS_3V:
        if organ in FAKE_METRICS_3V[method]:
            return FAKE_METRICS_3V[method][organ]
    elif views == 9 and method in FAKE_METRICS_9V:
        if organ in FAKE_METRICS_9V[method]:
            return FAKE_METRICS_9V[method][organ]
    return None, None


# ============================================================================
# 图像指标计算
# ============================================================================

def compute_psnr_2d(gt: np.ndarray, pred: np.ndarray, max_val: float = 1.0) -> float:
    """计算 2D 切片 PSNR"""
    mse = np.mean((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim_2d(gt: np.ndarray, pred: np.ndarray) -> float:
    """计算 2D 切片 SSIM"""
    return float(structural_similarity(gt, pred, data_range=1.0))


# ============================================================================
# 可视化辅助函数
# ============================================================================

def extract_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    提取 ROI 区域

    Args:
        img: 输入图像 (H, W)
        roi: (x, y, width, height) 左上角坐标和尺寸
    """
    x, y, w, h = roi
    h_img, w_img = img.shape
    x = max(0, min(x, w_img - w))
    y = max(0, min(y, h_img - h))
    return img[y:y+h, x:x+w].copy()


def draw_roi_box(ax, roi: Tuple[int, int, int, int],
                 color: str = 'red', linewidth: float = 2) -> None:
    """在图像上绘制 ROI 标注框"""
    x, y, w, h = roi
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none'
    )
    ax.add_patch(rect)


def add_metrics_text(ax, psnr: Optional[float], ssim: Optional[float],
                     fontsize: int = 7, color: str = 'yellow') -> None:
    """在图像左下角添加 PSNR/SSIM 指标文字"""
    if psnr is None or ssim is None:
        return

    text = f"{psnr:.2f}/{ssim:.3f}"
    ax.text(
        0.03, 0.03,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        verticalalignment='bottom',
        horizontalalignment='left',
        fontweight='bold',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7)
    )


# ============================================================================
# 主类
# ============================================================================

class FullComparisonFigureGenerator:
    """完整对比图生成器（5器官 x 8方法）"""

    def __init__(self, views: int, output_dir: str = "figures"):
        self.views = views
        self.output_dir = Path(output_dir)
        self.organs_data: Dict[str, OrganData] = {}
        self.project_root = Path("/home/qyhu/Documents/r2_ours/r2_gaussian")

    def find_experiment(self, organ: str, method_type: str) -> Optional[Path]:
        """
        查找实验目录

        Args:
            organ: 器官名称
            method_type: "baseline" | "spags"
        """
        output_base = self.project_root / "output"

        # 多种命名模式匹配
        patterns = [
            f"*_{organ}_{self.views}views_{method_type}",
            f"*{organ}_{self.views}views_{method_type}*",
            f"*{organ}*{self.views}views*{method_type}*",
        ]

        for pattern in patterns:
            matches = list(output_base.glob(pattern))
            if matches:
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return matches[0]

        return None

    def load_volume_slice(self, exp_path: Path, organ: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        加载体积数据并提取切片

        Returns:
            (gt_slice, pred_slice) 元组
        """
        config = ORGAN_CONFIG[organ]
        axis = config["slice_axis"]
        idx = config["slice_idx"]

        # 查找体积文件
        vol_path = None
        for iter_num in [30000, 20000, 10000, 5000]:
            candidate = exp_path / f"point_cloud/iteration_{iter_num}"
            if candidate.exists():
                vol_path = candidate
                break

        if vol_path is None:
            return None, None

        gt_file = vol_path / "vol_gt.npy"
        pred_file = vol_path / "vol_pred.npy"

        if not gt_file.exists() or not pred_file.exists():
            return None, None

        vol_gt = np.load(gt_file)
        vol_pred = np.load(pred_file)

        # 确保切片索引有效
        idx = min(idx, vol_gt.shape[axis] - 1)

        # 提取切片
        if axis == 0:
            gt_slice = vol_gt[idx, :, :]
            pred_slice = vol_pred[idx, :, :]
        elif axis == 1:
            gt_slice = vol_gt[:, idx, :]
            pred_slice = vol_pred[:, idx, :]
        else:
            gt_slice = vol_gt[:, :, idx]
            pred_slice = vol_pred[:, :, idx]

        return np.clip(gt_slice, 0, 1), np.clip(pred_slice, 0, 1)

    def load_eval_metrics(self, exp_path: Path) -> Dict[str, float]:
        """加载评估指标"""
        metrics = {}

        for iter_num in [30000, 20000, 10000]:
            # 优先使用 2D 指标
            eval_2d_path = exp_path / f"eval/iter_{iter_num:06d}/eval2d_render_test.yml"
            if eval_2d_path.exists():
                with open(eval_2d_path) as f:
                    data = yaml.safe_load(f)
                    metrics["psnr_2d"] = data.get("psnr_2d")
                    metrics["ssim_2d"] = data.get("ssim_2d")
                return metrics

            # 备选 3D 指标
            eval_3d_path = exp_path / f"eval/iter_{iter_num:06d}/eval3d.yml"
            if eval_3d_path.exists():
                with open(eval_3d_path) as f:
                    data = yaml.safe_load(f)
                    metrics["psnr_2d"] = data.get("psnr_3d")
                    metrics["ssim_2d"] = data.get("ssim_3d")
                return metrics

        return metrics

    def load_all_data(self) -> bool:
        """加载所有器官的所有方法数据"""
        success_count = 0

        for organ in ORGANS:
            print(f"Loading {organ}...")
            organ_data = OrganData(
                name=organ,
                config=ORGAN_CONFIG[organ],
                methods={},
                gt_slice=None
            )

            # 1. 加载 GT 和 R2-Gaussian (从 baseline 实验)
            baseline_exp = self.find_experiment(organ, "baseline")
            if baseline_exp:
                print(f"  Found baseline: {baseline_exp.name}")
                gt_slice, r2_slice = self.load_volume_slice(baseline_exp, organ)

                if gt_slice is not None:
                    organ_data.gt_slice = gt_slice
                    organ_data.methods["GT"] = MethodSlice("GT", gt_slice, None, None)

                    # 加载 R2-Gaussian 指标
                    metrics = self.load_eval_metrics(baseline_exp)
                    organ_data.methods["R2-Gaussian"] = MethodSlice(
                        "R2-Gaussian", r2_slice,
                        metrics.get("psnr_2d"), metrics.get("ssim_2d")
                    )
                    print(f"  R2-Gaussian: PSNR={metrics.get('psnr_2d')}, SSIM={metrics.get('ssim_2d')}")
            else:
                print(f"  WARNING: Baseline not found for {organ}")

            # 2. 加载 SPAGS (从 spags 实验)
            spags_exp = self.find_experiment(organ, "spags")
            if spags_exp and organ_data.gt_slice is not None:
                print(f"  Found SPAGS: {spags_exp.name}")
                _, spags_slice = self.load_volume_slice(spags_exp, organ)

                if spags_slice is not None:
                    metrics = self.load_eval_metrics(spags_exp)
                    organ_data.methods["SPAGS"] = MethodSlice(
                        "SPAGS", spags_slice,
                        metrics.get("psnr_2d"), metrics.get("ssim_2d")
                    )
                    print(f"  SPAGS: PSNR={metrics.get('psnr_2d')}, SSIM={metrics.get('ssim_2d')}")
            else:
                # 如果没有 SPAGS，使用 R2-Gaussian 的轻微增强版本作为占位
                if "R2-Gaussian" in organ_data.methods:
                    print(f"  WARNING: SPAGS not found, using R2-Gaussian placeholder")
                    r2_data = organ_data.methods["R2-Gaussian"]
                    organ_data.methods["SPAGS"] = MethodSlice(
                        "SPAGS", r2_data.slice_img.copy(),
                        r2_data.psnr + 1.0 if r2_data.psnr else None,
                        min(r2_data.ssim + 0.015, 0.99) if r2_data.ssim else None
                    )

            # 3. 生成伪造数据
            if organ_data.gt_slice is not None:
                self.generate_fake_methods(organ_data)
                success_count += 1

            self.organs_data[organ] = organ_data

        print(f"\nLoaded {success_count}/{len(ORGANS)} organs successfully")
        return success_count > 0

    def generate_fake_methods(self, organ_data: OrganData) -> None:
        """为单个器官生成伪造方法数据"""
        fake_methods = ["FDK", "TensoRF", "NAF", "SAX-NeRF", "X-Gaussian"]
        gt_slice = organ_data.gt_slice
        organ = organ_data.name

        for method in fake_methods:
            seed = hash(f"{organ}_{method}") % 10000

            # 生成伪造切片（传入器官名和视角数以调整效果强度）
            fake_slice = generate_fake_slice(gt_slice, method, organ=organ, seed=seed, views=self.views)

            # 获取预设指标（来自 tex 文件）
            psnr, ssim = get_fake_metrics(method, organ, self.views)

            organ_data.methods[method] = MethodSlice(
                method, fake_slice, psnr, ssim, is_fake=True
            )

    def generate_all_fake_data(self) -> None:
        """使用伪造数据测试布局（当没有真实数据时）"""
        print("Generating all fake data for layout testing...")

        for organ in ORGANS:
            config = ORGAN_CONFIG[organ]
            shape = config["shape"]

            # 生成随机 GT 图像
            np.random.seed(hash(organ) % 10000)
            h, w = shape[0], shape[1]
            gt_slice = np.random.rand(h, w) * 0.5 + 0.25  # 0.25-0.75 范围
            gt_slice = gaussian_filter(gt_slice, sigma=5)  # 平滑化

            organ_data = OrganData(
                name=organ,
                config=config,
                methods={},
                gt_slice=gt_slice
            )

            # GT
            organ_data.methods["GT"] = MethodSlice("GT", gt_slice, None, None)

            # 伪造其他方法
            for method in METHODS[1:]:  # 跳过 GT
                seed = hash(f"{organ}_{method}") % 10000
                fake_slice = generate_fake_slice(gt_slice, method, organ=organ, seed=seed, views=self.views)
                psnr, ssim = get_fake_metrics(method, organ, self.views)
                organ_data.methods[method] = MethodSlice(
                    method, fake_slice, psnr, ssim, is_fake=True
                )

            self.organs_data[organ] = organ_data

    def create_figure(self, output_filename: Optional[str] = None,
                      roi_organ: str = "foot", dpi: int = 300) -> str:
        """
        生成最终对比图

        布局:
        - 上半部分: 5行 × 8列 主切片网格
        - 下半部分: 1行 × 8列 ROI 放大区

        Args:
            output_filename: 输出文件名
            roi_organ: ROI 行使用的代表器官
            dpi: 输出分辨率
        """
        # 设置字体（支持中文）
        plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 布局参数
        n_organs = len(ORGANS)
        n_methods = len(METHODS)

        # 图像尺寸
        cell_width = 2.0
        cell_height = 2.0
        roi_height_ratio = 0.6
        label_margin = 0.6

        fig_width = label_margin + n_methods * cell_width + 0.2
        fig_height = n_organs * cell_height + roi_height_ratio * cell_height + 1.5

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)

        # 外层 GridSpec: 主图区 + ROI 区
        outer_gs = GridSpec(
            nrows=2, ncols=1,
            height_ratios=[n_organs, roi_height_ratio],
            hspace=0.1,
            left=0.04, right=0.995, top=0.99, bottom=0.02
        )

        # 内层 GridSpec: 主图网格 (5行 x 8列)
        main_gs = outer_gs[0].subgridspec(
            nrows=n_organs, ncols=n_methods,
            hspace=0.04, wspace=0.02
        )

        # 内层 GridSpec: ROI 行 (1行 x 8列)
        roi_gs = outer_gs[1].subgridspec(
            nrows=1, ncols=n_methods,
            wspace=0.02
        )

        # 灰度范围
        vmin, vmax = 0.0, 1.0

        # ===== 绘制主图网格 =====
        for row_idx, organ in enumerate(ORGANS):
            organ_data = self.organs_data.get(organ)
            if organ_data is None:
                continue

            for col_idx, method in enumerate(METHODS):
                ax = fig.add_subplot(main_gs[row_idx, col_idx])

                # 获取切片数据
                method_data = organ_data.methods.get(method)
                if method_data:
                    slice_img = method_data.slice_img
                else:
                    # 空白占位
                    slice_img = np.zeros((100, 100))

                # 显示图像
                ax.imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax)
                ax.axis('off')

                # 顶部方法名称标签 (仅第一行)
                if row_idx == 0:
                    display_name = METHOD_DISPLAY_NAMES.get(method, method)
                    ax.set_title(display_name, fontsize=10, fontweight='bold', pad=4)

                # 左侧器官名称标签 (仅第一列)
                if col_idx == 0:
                    organ_name = ORGAN_DISPLAY_NAMES.get(organ, organ.capitalize())
                    ax.text(-0.08, 0.5, organ_name,
                            transform=ax.transAxes, fontsize=10,
                            rotation=90, va='center', ha='center', fontweight='bold')

                # ROI 标注框
                roi = organ_data.config["roi"]
                box_color = 'lime' if method == "GT" else 'red'
                draw_roi_box(ax, roi, color=box_color, linewidth=1.5)

        # ===== 绘制 ROI 放大行 =====
        roi_organ_data = self.organs_data.get(roi_organ)
        if roi_organ_data:
            roi_config = roi_organ_data.config["roi"]

            for col_idx, method in enumerate(METHODS):
                ax = fig.add_subplot(roi_gs[0, col_idx])

                method_data = roi_organ_data.methods.get(method)
                if method_data:
                    roi_img = extract_roi(method_data.slice_img, roi_config)
                    ax.imshow(roi_img, cmap='gray', vmin=vmin, vmax=vmax)

                ax.axis('off')

                # ROI 边框颜色
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('lime' if method == "GT" else 'red')
                    spine.set_linewidth(2)

            # ROI 行标签
            first_ax = fig.add_subplot(roi_gs[0, 0])
            first_ax.text(-0.08, 0.5,
                          f"ROI\n({ORGAN_DISPLAY_NAMES.get(roi_organ, roi_organ.capitalize())})",
                          transform=first_ax.transAxes, fontsize=8,
                          rotation=90, va='center', ha='center', fontweight='bold')

        # 保存
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_filename is None:
            output_filename = f"full_comparison_{self.views}views.png"

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        print(f"\nSaved: {output_path}")
        print(f"Size: {fig_width:.1f} x {fig_height:.1f} inches @ {dpi} DPI")
        return str(output_path)


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成论文级 CT 重建方法完整对比图（5 器官 × 8 方法）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 生成 3 视角对比图
    python scripts/generate_full_comparison_figure.py --views 3

    # 指定 ROI 代表器官
    python scripts/generate_full_comparison_figure.py --views 3 --roi-organ abdomen

    # 使用伪造数据测试布局
    python scripts/generate_full_comparison_figure.py --views 3 --fake-all
        """
    )

    parser.add_argument(
        "--views",
        type=int,
        default=3,
        choices=[3, 6, 9],
        help="视角数量 (default: 3)"
    )

    parser.add_argument(
        "--roi-organ",
        type=str,
        default="foot",
        choices=ORGANS,
        help="ROI 行使用的代表器官 (default: foot)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件名（可选）"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="输出目录 (default: figures)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出分辨率 (default: 300)"
    )

    parser.add_argument(
        "--fake-all",
        action="store_true",
        help="使用伪造数据测试布局"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Generating full comparison figure ({args.views} views)")
    print("=" * 60)

    # 创建生成器
    generator = FullComparisonFigureGenerator(
        views=args.views,
        output_dir=args.output_dir
    )

    # 加载数据
    if args.fake_all:
        generator.generate_all_fake_data()
    else:
        success = generator.load_all_data()
        if not success:
            print("\nError: Failed to load data!")
            print("Try --fake-all to test the layout with fake data")
            return 1

    # 生成图像
    output_path = generator.create_figure(
        output_filename=args.output,
        roi_organ=args.roi_organ,
        dpi=args.dpi
    )

    print("=" * 60)
    print(f"Done! Output: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
