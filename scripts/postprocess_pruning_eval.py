#!/usr/bin/env python3
"""
点云后处理 Pruning 实验脚本

对已训练的 R²-Gaussian 模型进行后处理剪枝，验证减少点云数量对 PSNR/SSIM 的影响。

用法:
    # 单次评估
    python scripts/postprocess_pruning_eval.py \
        --model_path output/_2025_12_08_23_32_foot_3views_gar \
        --source_path data/369/foot_50_3views.pickle \
        --strategy topk \
        --keep_ratio 0.9

    # 批量测试（自动运行多个比例）
    python scripts/postprocess_pruning_eval.py \
        --model_path output/_2025_12_08_23_32_foot_3views_gar \
        --source_path data/369/foot_50_3views.pickle \
        --strategy topk \
        --batch_test
"""

import os
import sys
import torch
import yaml
import copy
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from r2_gaussian.arguments import ModelParams, PipelineParams, get_combined_args
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, initialize_gaussian
from r2_gaussian.utils.image_utils import metric_proj


# ============================================================================
# Pruning 策略函数
# ============================================================================

def prune_points_simple(gaussians: GaussianModel, keep_mask: torch.Tensor) -> None:
    """
    简化版点云剪枝（无需优化器）

    直接修改 GaussianModel 的内部张量，不依赖优化器。

    Args:
        gaussians: GaussianModel 实例
        keep_mask: 布尔掩码，True = 保留，False = 删除
    """
    import torch.nn as nn

    # 获取保留的点索引
    keep_indices = keep_mask.nonzero(as_tuple=True)[0]

    # 直接修改参数张量
    gaussians._xyz = nn.Parameter(gaussians._xyz.data[keep_indices])
    gaussians._density = nn.Parameter(gaussians._density.data[keep_indices])
    gaussians._scaling = nn.Parameter(gaussians._scaling.data[keep_indices])
    gaussians._rotation = nn.Parameter(gaussians._rotation.data[keep_indices])

    # 更新辅助张量（如果存在且非空）
    if hasattr(gaussians, 'xyz_gradient_accum') and gaussians.xyz_gradient_accum is not None and len(gaussians.xyz_gradient_accum) > 0:
        gaussians.xyz_gradient_accum = gaussians.xyz_gradient_accum[keep_indices]
    if hasattr(gaussians, 'denom') and gaussians.denom is not None and len(gaussians.denom) > 0:
        gaussians.denom = gaussians.denom[keep_indices]
    if hasattr(gaussians, 'max_radii2D') and gaussians.max_radii2D is not None and len(gaussians.max_radii2D) > 0:
        gaussians.max_radii2D = gaussians.max_radii2D[keep_indices]


def topk_density_prune(gaussians: GaussianModel, keep_ratio: float) -> int:
    """
    保留密度最高的 K% 点

    Args:
        gaussians: GaussianModel 实例
        keep_ratio: 保留比例 (0.0 - 1.0)

    Returns:
        删除的点数
    """
    density = gaussians.get_density.squeeze()
    n_original = len(density)
    k = int(n_original * keep_ratio)

    if k >= n_original:
        return 0

    # 获取 Top-K 密度最高的点索引
    _, topk_indices = torch.topk(density, k)

    # 创建保留掩码 (True = 保留)
    keep_mask = torch.zeros(n_original, dtype=torch.bool, device=density.device)
    keep_mask[topk_indices] = True

    # 执行剪枝
    prune_points_simple(gaussians, keep_mask)

    return n_original - k


def density_threshold_prune(gaussians: GaussianModel, threshold: float) -> int:
    """
    删除密度低于阈值的点

    Args:
        gaussians: GaussianModel 实例
        threshold: 密度阈值

    Returns:
        删除的点数
    """
    density = gaussians.get_density.squeeze()
    n_original = len(density)

    # 创建保留掩码 (True = 保留)
    keep_mask = (density >= threshold)
    n_to_delete = n_original - keep_mask.sum().item()

    if n_to_delete > 0:
        prune_points_simple(gaussians, keep_mask)

    return n_to_delete


def random_sample_prune(gaussians: GaussianModel, keep_ratio: float) -> int:
    """
    随机保留一定比例的点（对照组）

    Args:
        gaussians: GaussianModel 实例
        keep_ratio: 保留比例 (0.0 - 1.0)

    Returns:
        删除的点数
    """
    n_original = len(gaussians.get_xyz)
    k = int(n_original * keep_ratio)

    if k >= n_original:
        return 0

    # 随机选择要保留的点
    keep_indices = torch.randperm(n_original, device=gaussians.get_xyz.device)[:k]

    # 创建保留掩码 (True = 保留)
    keep_mask = torch.zeros(n_original, dtype=torch.bool, device=gaussians.get_xyz.device)
    keep_mask[keep_indices] = True

    # 执行剪枝
    prune_points_simple(gaussians, keep_mask)

    return n_original - k


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_2d(views, gaussians: GaussianModel, pipeline) -> dict:
    """
    渲染所有视角并计算 PSNR/SSIM

    Args:
        views: 相机视角列表
        gaussians: GaussianModel 实例
        pipeline: PipelineParams

    Returns:
        包含 psnr_2d, ssim_2d 的字典
    """
    gt_list = []
    render_list = []

    # 清理 GPU 缓存
    torch.cuda.empty_cache()

    with torch.no_grad():
        for view in tqdm(views, desc="Rendering", leave=False):
            rendering = render(view, gaussians, pipeline)["render"]
            gt = view.original_image[0:3, :, :]
            # 将渲染结果移到 CPU 以节省 GPU 内存
            gt_list.append(gt.cpu())
            render_list.append(rendering.cpu())
            # 及时清理 CUDA 缓存
            torch.cuda.empty_cache()

    # 在 CPU 上计算指标
    images = torch.concat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)

    # 移回 GPU 计算指标（如果需要）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images = images.to(device)
    gt_images = gt_images.to(device)

    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")

    # 清理
    del images, gt_images, gt_list, render_list
    torch.cuda.empty_cache()

    return {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
    }


def load_model_and_scene(model_path: str, source_path: str, iteration: int = -1):
    """
    加载已训练模型和场景

    Args:
        model_path: 模型输出目录
        source_path: 数据源路径
        iteration: 加载的迭代次数 (-1 表示最新)

    Returns:
        (gaussians, scene, pipeline, loaded_iter)
    """
    # 创建参数对象
    parser = ArgumentParser()
    model_args = ModelParams(parser, sentinel=True)
    pipeline_args = PipelineParams(parser)

    # 设置路径
    model_args.model_path = model_path
    model_args.source_path = source_path

    # 加载场景
    scene = Scene(model_args, shuffle=False)

    # 加载模型
    gaussians = GaussianModel(None)
    loaded_iter = initialize_gaussian(gaussians, model_args, iteration)

    return gaussians, scene, pipeline_args, loaded_iter


def run_single_experiment(
    model_path: str,
    source_path: str,
    strategy: str,
    param: float,
    iteration: int = -1,
) -> dict:
    """
    运行单次剪枝实验

    Args:
        model_path: 模型输出目录
        source_path: 数据源路径
        strategy: 剪枝策略 ("topk", "threshold", "random")
        param: 策略参数 (keep_ratio 或 threshold)
        iteration: 加载的迭代次数

    Returns:
        实验结果字典
    """
    # 加载模型和场景
    gaussians, scene, pipeline, loaded_iter = load_model_and_scene(
        model_path, source_path, iteration
    )

    n_original = len(gaussians.get_xyz)

    # 应用剪枝策略
    if strategy == "topk":
        n_deleted = topk_density_prune(gaussians, param)
        strategy_desc = f"Top-K {param*100:.0f}%"
    elif strategy == "threshold":
        n_deleted = density_threshold_prune(gaussians, param)
        strategy_desc = f"Threshold {param}"
    elif strategy == "random":
        n_deleted = random_sample_prune(gaussians, param)
        strategy_desc = f"Random {param*100:.0f}%"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    n_remaining = len(gaussians.get_xyz)
    keep_ratio = n_remaining / n_original

    # 评估
    test_cameras = scene.getTestCameras()
    eval_result = evaluate_2d(test_cameras, gaussians, pipeline)

    return {
        "strategy": strategy,
        "strategy_desc": strategy_desc,
        "param": param,
        "n_original": n_original,
        "n_remaining": n_remaining,
        "n_deleted": n_deleted,
        "keep_ratio": keep_ratio,
        "psnr_2d": eval_result["psnr_2d"],
        "ssim_2d": eval_result["ssim_2d"],
        "loaded_iter": loaded_iter,
    }


def run_batch_experiments(
    model_path: str,
    source_path: str,
    strategy: str = "topk",
    iteration: int = -1,
) -> list:
    """
    运行批量剪枝实验

    Args:
        model_path: 模型输出目录
        source_path: 数据源路径
        strategy: 剪枝策略
        iteration: 加载的迭代次数

    Returns:
        实验结果列表
    """
    results = []

    # 定义测试参数
    if strategy == "topk":
        params = [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
    elif strategy == "threshold":
        params = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    elif strategy == "random":
        params = [1.0, 0.90, 0.80, 0.70]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"\n{'='*60}")
    print(f"批量实验: {strategy}")
    print(f"模型: {model_path}")
    print(f"{'='*60}")

    for param in params:
        print(f"\n>>> 测试参数: {param}")
        result = run_single_experiment(
            model_path, source_path, strategy, param, iteration
        )
        results.append(result)

        print(f"    保留: {result['n_remaining']}/{result['n_original']} ({result['keep_ratio']*100:.1f}%)")
        print(f"    PSNR: {result['psnr_2d']:.4f}, SSIM: {result['ssim_2d']:.4f}")

    return results


def print_results_table(results: list, baseline_psnr: float = None):
    """
    打印结果表格
    """
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)

    if baseline_psnr is None and results:
        baseline_psnr = results[0]["psnr_2d"]

    print(f"\n{'策略':<20} {'保留比例':>10} {'点数':>10} {'PSNR':>10} {'SSIM':>10} {'Δ PSNR':>10}")
    print("-"*80)

    for r in results:
        delta_psnr = r["psnr_2d"] - baseline_psnr if baseline_psnr else 0
        delta_str = f"{delta_psnr:+.4f}" if r != results[0] else "-"

        print(f"{r['strategy_desc']:<20} {r['keep_ratio']*100:>9.1f}% {r['n_remaining']:>10} "
              f"{r['psnr_2d']:>10.4f} {r['ssim_2d']:>10.4f} {delta_str:>10}")

    print("="*80)


def save_results(results: list, output_path: str):
    """
    保存结果到 YAML 文件
    """
    # 转换为可序列化格式
    save_data = []
    for r in results:
        save_data.append({
            "strategy": r["strategy"],
            "strategy_desc": r["strategy_desc"],
            "param": float(r["param"]),
            "n_original": int(r["n_original"]),
            "n_remaining": int(r["n_remaining"]),
            "keep_ratio": float(r["keep_ratio"]),
            "psnr_2d": float(r["psnr_2d"]),
            "ssim_2d": float(r["ssim_2d"]),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n结果已保存到: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = ArgumentParser(description="点云后处理 Pruning 实验")

    parser.add_argument("--model_path", type=str, required=True,
                        help="模型输出目录 (e.g., output/_2025_12_08_23_32_foot_3views_gar)")
    parser.add_argument("--source_path", type=str, required=True,
                        help="数据源路径 (e.g., data/369/foot_50_3views.pickle)")
    parser.add_argument("--strategy", type=str, default="topk",
                        choices=["topk", "threshold", "random"],
                        help="剪枝策略")
    parser.add_argument("--keep_ratio", type=float, default=0.9,
                        help="保留比例 (topk/random 策略)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="密度阈值 (threshold 策略)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="加载的迭代次数 (-1 表示最新)")
    parser.add_argument("--batch_test", action="store_true",
                        help="运行批量测试")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出路径")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU 设备 ID")

    args = parser.parse_args()

    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.empty_cache()

    # 设置默认输出路径
    if args.output is None:
        exp_name = os.path.basename(args.model_path)
        args.output = f"output/pruning_experiments/{exp_name}_{args.strategy}_results.yml"

    if args.batch_test:
        # 批量测试
        results = run_batch_experiments(
            args.model_path,
            args.source_path,
            args.strategy,
            args.iteration,
        )
        print_results_table(results)
        save_results(results, args.output)
    else:
        # 单次测试
        param = args.threshold if args.strategy == "threshold" else args.keep_ratio
        result = run_single_experiment(
            args.model_path,
            args.source_path,
            args.strategy,
            param,
            args.iteration,
        )

        print("\n" + "="*60)
        print("单次实验结果")
        print("="*60)
        print(f"策略: {result['strategy_desc']}")
        print(f"原始点数: {result['n_original']}")
        print(f"剩余点数: {result['n_remaining']}")
        print(f"保留比例: {result['keep_ratio']*100:.1f}%")
        print(f"PSNR 2D: {result['psnr_2d']:.4f}")
        print(f"SSIM 2D: {result['ssim_2d']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()
