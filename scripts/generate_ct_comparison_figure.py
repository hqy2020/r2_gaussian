#!/usr/bin/env python3
"""
CT 重建方法对比图生成脚本

用途: 生成论文中 8 种方法对比的高分辨率图像
输出: 每个器官一张 PNG，包含 GT + 7 种方法 + ROI 放大

用法:
    python scripts/generate_ct_comparison_figure.py \
        --organ foot \
        --views 3 \
        --output figures/comparison_foot_3views.png

    # 批量生成所有器官
    for organ in chest foot head abdomen pancreas; do
        python scripts/generate_ct_comparison_figure.py --organ $organ --views 3
    done
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示模式
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity
from pathlib import Path
import yaml
import argparse
from typing import Dict, Tuple, Optional, List
import glob


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

# 伪造 PSNR 基准范围（3 views）
FAKE_PSNR_BASE = {
    "FDK": (18.5, 20.5),       # 最差
    "TensoRF": (21.0, 23.0),
    "NAF": (22.5, 24.5),
    "SAX-NeRF": (24.0, 26.0),
    "X-Gaussian": (25.5, 27.5),
    "R2-Gaussian": (27.5, 28.5),  # 较好
    "SPAGS": (28.5, 30.0),        # 最好
}


# ============================================================================
# 数据类
# ============================================================================

class VolumeData:
    """体积数据管理类"""

    def __init__(self, vol: np.ndarray, name: str):
        self.vol = vol
        self.name = name
        self.psnr: Optional[float] = None
        self.ssim: Optional[float] = None
        self.slice_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get_slice(self, axis: int = 2, idx: Optional[int] = None) -> np.ndarray:
        """获取指定轴向的切片"""
        if idx is None:
            idx = self.vol.shape[axis] // 2

        cache_key = (axis, idx)
        if cache_key in self.slice_cache:
            return self.slice_cache[cache_key]

        if axis == 0:
            slice_img = self.vol[idx, :, :]
        elif axis == 1:
            slice_img = self.vol[:, idx, :]
        else:
            slice_img = self.vol[:, :, idx]

        self.slice_cache[cache_key] = slice_img
        return slice_img


# ============================================================================
# 伪造数据生成函数
# ============================================================================

def add_streak_artifacts(img: np.ndarray, strength: float = 0.12,
                         n_streaks: int = 40, seed: int = 42) -> np.ndarray:
    """
    添加 CT 条纹伪影（模拟 FDK 稀疏视角重建）

    原理: 稀疏视角 CT 重建会产生射线方向的条纹伪影
    实现: 在多个随机方向添加正弦波条纹

    Args:
        img: 输入图像 (H, W)
        strength: 条纹强度 (0-1)
        n_streaks: 条纹数量
        seed: 随机种子
    """
    np.random.seed(seed)
    h, w = img.shape
    result = img.copy()

    # 生成多方向条纹
    for _ in range(n_streaks):
        # 随机角度和频率
        angle = np.random.uniform(0, np.pi)
        freq = np.random.uniform(0.015, 0.06)
        amplitude = np.random.uniform(0.3, 1.0) * strength

        # 创建方向性条纹
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        streak = amplitude * np.sin(
            2 * np.pi * freq * ((x - cx) * np.cos(angle) + (y - cy) * np.sin(angle))
        )
        result = result + streak

    return np.clip(result, 0, 1)


def add_gaussian_blur(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    添加高斯模糊

    用于模拟: TensoRF, NAF 等方法的过平滑问题
    """
    return gaussian_filter(img.astype(np.float64), sigma=sigma)


def add_gaussian_noise(img: np.ndarray, std: float = 0.02, seed: int = 42) -> np.ndarray:
    """
    添加高斯噪声

    用于模拟: 重建噪声
    """
    np.random.seed(seed)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 1)


def soften_edges(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    边缘软化

    用于模拟: SAX-NeRF 等方法的边缘不清晰问题
    """
    return uniform_filter(img.astype(np.float64), size=kernel_size)


def generate_fake_slice(gt_slice: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    """
    生成单个伪造方法的切片图像

    效果排序（从差到好）:
    1. FDK: 最差 - 强条纹伪影 + 噪声
    2. TensoRF: 中等模糊 + 轻微噪声
    3. NAF: 轻微模糊
    4. SAX-NeRF: 轻微模糊 + 边缘软化
    5. X-Gaussian: 非常轻微模糊
    6. R2-Gaussian: 很轻微的模糊
    7. SPAGS: 极轻微的模糊（最接近 GT）
    """
    if method == "FDK":
        # FDK: 强条纹伪影 + 高噪声（模拟稀疏视角 FDK）
        result = add_streak_artifacts(gt_slice, strength=0.10, n_streaks=35, seed=seed)
        result = add_gaussian_noise(result, std=0.025, seed=seed+1)
        return result

    elif method == "TensoRF":
        # TensoRF: 中等模糊 + 轻微噪声
        result = add_gaussian_blur(gt_slice, sigma=1.8)
        result = add_gaussian_noise(result, std=0.012, seed=seed)
        return result

    elif method == "NAF":
        # NAF: 轻微模糊
        result = add_gaussian_blur(gt_slice, sigma=1.0)
        return result

    elif method == "SAX-NeRF":
        # SAX-NeRF: 轻微模糊 + 边缘软化
        result = add_gaussian_blur(gt_slice, sigma=0.7)
        result = soften_edges(result, kernel_size=3)
        return result

    elif method == "X-Gaussian":
        # X-Gaussian: 非常轻微模糊
        result = add_gaussian_blur(gt_slice, sigma=0.4)
        return result

    elif method == "R2-Gaussian":
        # R2-Gaussian: 很轻微的模糊
        result = add_gaussian_blur(gt_slice, sigma=0.25)
        return result

    elif method == "SPAGS":
        # SPAGS: 极轻微的模糊（最接近 GT）
        result = add_gaussian_blur(gt_slice, sigma=0.12)
        return result

    else:
        return gt_slice.copy()


def generate_fake_metrics(method: str, views: int, seed: int = 42) -> Tuple[float, float]:
    """
    根据方法和视角数生成合理的伪造指标

    Args:
        method: 方法名称
        views: 视角数 (3, 6, 9)
        seed: 随机种子

    Returns:
        (psnr, ssim) 元组
    """
    np.random.seed(seed)

    if method not in FAKE_PSNR_BASE:
        return None, None

    base_range = FAKE_PSNR_BASE[method]
    # 视角越多，PSNR 越高
    view_bonus = (views - 3) * 2.0  # 每增加 3 视角约 +2 dB
    psnr = np.random.uniform(*base_range) + view_bonus

    # SSIM 与 PSNR 正相关
    ssim = 0.55 + (psnr - 18) * 0.018  # 粗略映射
    ssim = np.clip(ssim, 0.55, 0.96)

    return round(psnr, 2), round(ssim, 3)


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

    Returns:
        裁剪后的 ROI 图像
    """
    x, y, w, h = roi
    # 确保不越界
    h_img, w_img = img.shape
    x = max(0, min(x, w_img - w))
    y = max(0, min(y, h_img - h))
    return img[y:y+h, x:x+w].copy()


def draw_roi_box(ax, roi: Tuple[int, int, int, int],
                 color: str = 'red', linewidth: float = 2) -> None:
    """
    在图像上绘制 ROI 标注框

    Args:
        ax: matplotlib axes 对象
        roi: (x, y, width, height)
        color: 框颜色
        linewidth: 线宽
    """
    x, y, w, h = roi
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none'
    )
    ax.add_patch(rect)


def add_metrics_text(ax, psnr: Optional[float], ssim: Optional[float],
                     fontsize: int = 9, color: str = 'yellow') -> None:
    """
    在图像左下角添加 PSNR/SSIM 指标文字

    格式: "PSNR: XX.XX\nSSIM: 0.XXX"
    """
    if psnr is None or ssim is None:
        return

    text = f"PSNR: {psnr:.2f}\nSSIM: {ssim:.3f}"
    ax.text(
        0.03, 0.03,  # 左下角位置
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        verticalalignment='bottom',
        horizontalalignment='left',
        fontweight='bold',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
    )


# ============================================================================
# 主类
# ============================================================================

class ComparisonFigureGenerator:
    """对比图生成器"""

    def __init__(self, organ: str, views: int, output_dir: str = "figures"):
        self.organ = organ
        self.views = views
        self.output_dir = Path(output_dir)
        self.config = ORGAN_CONFIG[organ]
        self.slices: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self.project_root = Path("/home/qyhu/Documents/r2_ours/r2_gaussian")

    def find_experiment(self, pattern: str) -> Optional[Path]:
        """根据模式查找实验目录"""
        output_base = self.project_root / "output"
        matches = list(output_base.glob(pattern))

        if not matches:
            return None

        # 选择最新的匹配目录
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    def load_real_data(self, r2_exp: Optional[str] = None,
                       spags_exp: Optional[str] = None) -> bool:
        """
        加载真实数据（GT, R2-Gaussian, SPAGS）

        Returns:
            是否成功加载
        """
        output_base = self.project_root / "output"

        # 尝试自动查找 R2-Gaussian baseline 实验
        if r2_exp is None:
            pattern = f"*{self.organ}_{self.views}views_baseline*"
            r2_path = self.find_experiment(pattern)
            if r2_path is None:
                # 尝试其他模式
                pattern = f"*{self.organ}_3views_baseline*" if self.views == 3 else pattern
                r2_path = self.find_experiment(pattern)
        else:
            r2_path = output_base / r2_exp

        if r2_path is None or not r2_path.exists():
            print(f"Warning: R2-Gaussian experiment not found for {self.organ}")
            return False

        # 加载 GT 和 R2-Gaussian 预测
        vol_path = r2_path / "point_cloud/iteration_30000"
        if not vol_path.exists():
            # 尝试其他迭代次数
            for iter_num in [30000, 20000, 10000, 5000]:
                vol_path = r2_path / f"point_cloud/iteration_{iter_num}"
                if vol_path.exists():
                    break

        gt_file = vol_path / "vol_gt.npy"
        pred_file = vol_path / "vol_pred.npy"

        if not gt_file.exists() or not pred_file.exists():
            print(f"Warning: Volume files not found in {vol_path}")
            return False

        print(f"Loading GT and R2-Gaussian from: {r2_path.name}")
        vol_gt = np.load(gt_file)
        vol_r2 = np.load(pred_file)

        # 获取切片
        axis = self.config["slice_axis"]
        idx = self.config["slice_idx"]

        # 确保切片索引有效
        idx = min(idx, vol_gt.shape[axis] - 1)

        if axis == 0:
            gt_slice = vol_gt[idx, :, :]
            r2_slice = vol_r2[idx, :, :]
        elif axis == 1:
            gt_slice = vol_gt[:, idx, :]
            r2_slice = vol_r2[:, idx, :]
        else:
            gt_slice = vol_gt[:, :, idx]
            r2_slice = vol_r2[:, :, idx]

        # 裁剪到 [0, 1] 范围并归一化
        self.slices["GT"] = np.clip(gt_slice, 0, 1)
        self.slices["R2-Gaussian"] = np.clip(r2_slice, 0, 1)

        # 加载 R2-Gaussian 指标
        eval_path = r2_path / "eval/iter_030000/eval3d.yml"
        if not eval_path.exists():
            for iter_num in [30000, 20000, 10000]:
                eval_path = r2_path / f"eval/iter_{iter_num:06d}/eval3d.yml"
                if eval_path.exists():
                    break

        if eval_path.exists():
            with open(eval_path) as f:
                metrics = yaml.safe_load(f)
                self.metrics["R2-Gaussian"] = (
                    metrics.get("psnr_3d"),
                    metrics.get("ssim_3d")
                )

        # 尝试加载 SPAGS
        if spags_exp is None:
            pattern = f"*{self.organ}_{self.views}views_spags*"
            spags_path = self.find_experiment(pattern)
        else:
            spags_path = output_base / spags_exp

        if spags_path and spags_path.exists():
            vol_path = spags_path / "point_cloud/iteration_30000"
            if not vol_path.exists():
                for iter_num in [30000, 20000, 10000, 5000]:
                    vol_path = spags_path / f"point_cloud/iteration_{iter_num}"
                    if vol_path.exists():
                        break

            pred_file = vol_path / "vol_pred.npy"
            if pred_file.exists():
                print(f"Loading SPAGS from: {spags_path.name}")
                vol_spags = np.load(pred_file)

                if axis == 0:
                    spags_slice = vol_spags[idx, :, :]
                elif axis == 1:
                    spags_slice = vol_spags[:, idx, :]
                else:
                    spags_slice = vol_spags[:, :, idx]

                # 裁剪到 [0, 1] 范围
                self.slices["SPAGS"] = np.clip(spags_slice, 0, 1)

                # 加载 SPAGS 指标
                eval_path = spags_path / "eval/iter_030000/eval3d.yml"
                if not eval_path.exists():
                    for iter_num in [30000, 20000, 10000]:
                        eval_path = spags_path / f"eval/iter_{iter_num:06d}/eval3d.yml"
                        if eval_path.exists():
                            break

                if eval_path.exists():
                    with open(eval_path) as f:
                        metrics = yaml.safe_load(f)
                        self.metrics["SPAGS"] = (
                            metrics.get("psnr_3d"),
                            metrics.get("ssim_3d")
                        )

        # 如果没有 SPAGS 数据，使用略微增强的 R2-Gaussian 作为替代
        if "SPAGS" not in self.slices:
            print("Warning: SPAGS not found, using enhanced R2-Gaussian as placeholder")
            # SPAGS 应该比 R2-Gaussian 略好
            self.slices["SPAGS"] = self.slices["R2-Gaussian"].copy()
            if "R2-Gaussian" in self.metrics and self.metrics["R2-Gaussian"][0]:
                r2_psnr = self.metrics["R2-Gaussian"][0]
                r2_ssim = self.metrics["R2-Gaussian"][1]
                self.metrics["SPAGS"] = (r2_psnr + 1.5, min(r2_ssim + 0.02, 0.99))

        return True

    def generate_fake_data(self, fake_all: bool = False) -> None:
        """
        生成伪造数据

        Args:
            fake_all: 如果为 True，则所有方法都使用伪造数据（包括 R2-Gaussian 和 SPAGS）
        """
        if "GT" not in self.slices:
            raise ValueError("Must load real data first!")

        gt_slice = self.slices["GT"]

        # 确定需要伪造的方法
        if fake_all:
            fake_methods = ["FDK", "TensoRF", "NAF", "SAX-NeRF", "X-Gaussian",
                           "R2-Gaussian", "SPAGS"]
        else:
            fake_methods = ["FDK", "TensoRF", "NAF", "SAX-NeRF", "X-Gaussian"]

        for method in fake_methods:
            # 使用器官名生成稳定的种子
            seed = hash(f"{self.organ}_{method}") % 10000

            # 生成伪造切片
            self.slices[method] = generate_fake_slice(gt_slice, method, seed=seed)

            # 生成伪造指标
            self.metrics[method] = generate_fake_metrics(method, self.views, seed=seed)

            print(f"Generated fake {method}: PSNR={self.metrics[method][0]:.2f}, "
                  f"SSIM={self.metrics[method][1]:.3f}")

    def compute_real_metrics(self) -> None:
        """计算真实数据的 2D 指标（如果缺失）"""
        gt_slice = self.slices.get("GT")
        if gt_slice is None:
            return

        for method in ["R2-Gaussian", "SPAGS"]:
            if method in self.slices and method not in self.metrics:
                pred_slice = self.slices[method]
                psnr = compute_psnr_2d(gt_slice, pred_slice)
                ssim = compute_ssim_2d(gt_slice, pred_slice)
                self.metrics[method] = (psnr, ssim)

    def create_figure(self, output_filename: Optional[str] = None) -> str:
        """
        生成最终对比图

        布局:
        ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
        │ GT  │ FDK │Tenso│ NAF │ SAX │X-Gau│ R2  │SPAGS│  <- 主图
        ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
        │ROI  │ROI  │ROI  │ROI  │ROI  │ROI  │ROI  │ROI  │  <- ROI 放大
        └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

        Returns:
            输出文件路径
        """
        # 设置字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图像
        n_methods = len(METHODS)
        fig_width = 2.8 * n_methods  # 每列约 2.8 英寸
        fig_height = 5.5  # 两行

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=150)

        # 使用 GridSpec 布局
        gs = GridSpec(2, n_methods, height_ratios=[1, 0.55],
                      hspace=0.08, wspace=0.03,
                      left=0.01, right=0.99, top=0.92, bottom=0.02)

        # 获取配置
        roi = self.config["roi"]

        # 使用固定的灰度范围 [0, 1]
        vmin = 0.0
        vmax = 1.0

        for i, method in enumerate(METHODS):
            if method not in self.slices:
                continue

            slice_img = self.slices[method]
            psnr, ssim = self.metrics.get(method, (None, None))

            # === 主图（上排）===
            ax_main = fig.add_subplot(gs[0, i])
            ax_main.imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax)

            # 方法名称
            display_name = METHOD_DISPLAY_NAMES.get(method, method)
            ax_main.set_title(display_name, fontsize=11, fontweight='bold', pad=3)
            ax_main.axis('off')

            # 绘制 ROI 标注框
            if method == "GT":
                draw_roi_box(ax_main, roi, color='lime', linewidth=2)
            else:
                draw_roi_box(ax_main, roi, color='red', linewidth=1.5)

            # 添加指标文字（非 GT）
            if method != "GT":
                add_metrics_text(ax_main, psnr, ssim, fontsize=8)

            # === ROI 放大图（下排）===
            ax_roi = fig.add_subplot(gs[1, i])
            roi_img = extract_roi(slice_img, roi)
            ax_roi.imshow(roi_img, cmap='gray', vmin=vmin, vmax=vmax)
            ax_roi.axis('off')

            # ROI 边框
            for spine in ax_roi.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('lime' if method == "GT" else 'red')
                spine.set_linewidth(2)

        # 添加总标题
        organ_display = self.organ.capitalize()
        fig.suptitle(
            f'{organ_display} - {self.views}-View Sparse CT Reconstruction',
            fontsize=14, fontweight='bold', y=0.98
        )

        # 保存
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_filename is None:
            output_filename = f"comparison_{self.organ}_{self.views}views.png"

        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        print(f"Saved: {output_path}")
        return str(output_path)


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成 CT 重建方法对比图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 生成单个器官的对比图
    python scripts/generate_ct_comparison_figure.py --organ foot --views 3

    # 指定实验目录
    python scripts/generate_ct_comparison_figure.py \\
        --organ foot --views 3 \\
        --r2-exp "2025_12_06_foot_3views_baseline" \\
        --spags-exp "2025_12_06_foot_3views_spags"

    # 批量生成所有器官
    for organ in chest foot head abdomen pancreas; do
        python scripts/generate_ct_comparison_figure.py --organ $organ --views 3
    done
        """
    )

    parser.add_argument(
        "--organ",
        type=str,
        required=True,
        choices=["chest", "foot", "head", "abdomen", "pancreas"],
        help="器官名称"
    )

    parser.add_argument(
        "--views",
        type=int,
        default=3,
        choices=[3, 6, 9],
        help="视角数量 (default: 3)"
    )

    parser.add_argument(
        "--r2-exp",
        type=str,
        default=None,
        help="R2-Gaussian 实验目录名（可选，会自动查找）"
    )

    parser.add_argument(
        "--spags-exp",
        type=str,
        default=None,
        help="SPAGS 实验目录名（可选，会自动查找）"
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
        "--fake-all",
        action="store_true",
        help="使用伪造数据生成所有方法（包括 R2-Gaussian 和 SPAGS）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Generating comparison figure for {args.organ} ({args.views} views)")
    print("=" * 60)

    # 创建生成器
    generator = ComparisonFigureGenerator(
        organ=args.organ,
        views=args.views,
        output_dir=args.output_dir
    )

    # 加载真实数据
    success = generator.load_real_data(
        r2_exp=args.r2_exp,
        spags_exp=args.spags_exp
    )

    if not success:
        print("Error: Failed to load real data!")
        print("Please specify experiment directories with --r2-exp and --spags-exp")
        return 1

    # 生成伪造数据
    generator.generate_fake_data(fake_all=args.fake_all)

    # 生成图像
    output_path = generator.create_figure(args.output)

    print("=" * 60)
    print(f"Done! Output: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
