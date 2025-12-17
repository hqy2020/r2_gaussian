#!/usr/bin/env python
"""
多方法新视角合成对比渲染脚本
支持 SPAGS (R²-Gaussian)、NeRF 系列、X-Gaussian 的统一对比
"""

import os
import os.path as osp
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_metrics(pred, gt):
    """计算单张图像的 PSNR 和 SSIM"""
    # 归一化到 [0, 1]
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)

    psnr_val = psnr(gt_norm, pred_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, pred_norm, data_range=1.0)
    return psnr_val, ssim_val


def compute_avg_metrics(preds, gts):
    """计算平均 PSNR 和 SSIM"""
    if not preds or not gts:
        return None, None

    psnr_list, ssim_list = [], []
    for pred, gt in zip(preds, gts):
        p, s = compute_metrics(pred, gt)
        psnr_list.append(p)
        ssim_list.append(s)

    return np.mean(psnr_list), np.mean(ssim_list)

sys.path.append("./")
from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams, PipelineParams, get_combined_args
from r2_gaussian.gaussian import GaussianModel, render, initialize_gaussian

# NeRF 相关
from r2_gaussian.baselines.nerf_base.trainer import (
    NeRFModel,
    get_method_config as get_nerf_config,
    _render_full_image,
)


def find_best_checkpoint(model_path, target_iter):
    """查找最佳可用 checkpoint (优先目标迭代，否则最大迭代)"""
    point_cloud_dir = osp.join(model_path, "point_cloud")
    if not osp.exists(point_cloud_dir):
        return None, None

    # 查找所有迭代目录
    iter_dirs = [d for d in os.listdir(point_cloud_dir) if d.startswith("iteration_")]
    if not iter_dirs:
        return None, None

    # 提取迭代次数
    iterations = []
    for d in iter_dirs:
        try:
            iter_num = int(d.replace("iteration_", ""))
            ckpt_path = osp.join(point_cloud_dir, d, "point_cloud.pickle")
            if osp.exists(ckpt_path):
                iterations.append((iter_num, ckpt_path))
        except ValueError:
            continue

    if not iterations:
        return None, None

    # 优先使用目标迭代
    for iter_num, path in iterations:
        if iter_num == target_iter:
            return iter_num, path

    # 否则使用最大迭代
    iterations.sort(key=lambda x: x[0], reverse=True)
    return iterations[0]


def render_gaussian(model_path, source_path, iteration, view_indices):
    """渲染 3DGS 方法 (SPAGS/baseline)"""
    # 简化的参数
    class Args:
        def __init__(self):
            self.source_path = source_path
            self.model_path = model_path
            self.eval = True
            self.data_device = "cuda"
            self.scale_min = -1
            self.scale_max = -1
            # K-Planes 相关参数
            self.enable_kplanes = False
            self.kplanes_resolution = 128
            self.kplanes_feature_dim = 32

    args = Args()

    # 加载场景
    scene = Scene(args, shuffle=False)

    # 加载 Gaussians
    gaussians = GaussianModel(None, args=args)

    # 查找最佳可用 checkpoint
    found_iter, ckpt_path = find_best_checkpoint(model_path, iteration)

    if ckpt_path and osp.exists(ckpt_path):
        gaussians.load_ply(ckpt_path)
        print(f"  加载 Gaussian checkpoint (iter {found_iter}): {ckpt_path}")
    else:
        print(f"  警告: 在 {model_path} 中找不到任何 checkpoint")
        return None, None

    # 渲染指定视角
    test_cameras = scene.getTestCameras()
    images = []

    class PipeArgs:
        debug = False
        compute_cov3D_python = False

    pipe = PipeArgs()

    with torch.no_grad():
        for idx in view_indices:
            if idx < len(test_cameras):
                viewpoint = test_cameras[idx]
                render_pkg = render(viewpoint, gaussians, pipe)
                image = render_pkg["render"]
                images.append(image[0].detach().cpu().numpy())

    # 获取 GT
    gt_images = []
    for idx in view_indices:
        if idx < len(test_cameras):
            gt = test_cameras[idx].original_image[0].detach().cpu().numpy()
            gt_images.append(gt)

    return images, gt_images


def render_nerf(model_path, source_path, method, iteration, view_indices):
    """渲染 NeRF 方法"""
    class Args:
        def __init__(self):
            self.source_path = source_path
            self.model_path = model_path
            self.eval = True
            self.data_device = "cuda"
            self.scale_min = -1
            self.scale_max = -1

    args = Args()

    # 获取配置
    config = get_nerf_config(method)

    # 加载场景
    scene = Scene(args, shuffle=False)
    scanner_cfg = scene.scanner_cfg

    # 创建并加载模型
    nerf_model = NeRFModel(config).cuda()

    ckpt_path = osp.join(model_path, f"{method}_iter_{iteration}.pth")
    if not osp.exists(ckpt_path):
        print(f"  警告: 找不到 checkpoint {ckpt_path}")
        return None, None

    ckpt = torch.load(ckpt_path)
    nerf_model.net.load_state_dict(ckpt["network"])
    if nerf_model.net_fine is not None and ckpt.get("network_fine"):
        nerf_model.net_fine.load_state_dict(ckpt["network_fine"])
    print(f"  加载 NeRF checkpoint: {ckpt_path}")

    nerf_model.eval()

    # 渲染指定视角
    test_cameras = scene.getTestCameras()
    chunk_rays = int(getattr(config, "eval_rays_chunk", 8192))
    images = []

    with torch.no_grad():
        for idx in tqdm(view_indices, desc=f"  渲染 {method}", leave=False):
            if idx < len(test_cameras):
                viewpoint = test_cameras[idx]
                pred_image = _render_full_image(
                    viewpoint, scanner_cfg, nerf_model, config, chunk_rays=chunk_rays
                )
                images.append(pred_image[0].cpu().numpy())

    # 获取 GT
    gt_images = []
    for idx in view_indices:
        if idx < len(test_cameras):
            gt = test_cameras[idx].original_image[0].cpu().numpy()
            gt_images.append(gt)

    return images, gt_images


def create_comparison_figure(results, view_indices, save_path, title="Novel View Synthesis Comparison"):
    """创建对比图，显示 PSNR 和 SSIM 指标"""
    methods = list(results.keys())
    n_methods = len(methods)
    n_views = len(view_indices)

    # 计算每个方法的平均指标
    metrics = {}
    for method in methods:
        preds = results[method]['pred']
        gts = results[method]['gt']
        if preds and gts:
            avg_psnr, avg_ssim = compute_avg_metrics(preds, gts)
            metrics[method] = {'psnr': avg_psnr, 'ssim': avg_ssim}
        else:
            metrics[method] = {'psnr': None, 'ssim': None}

    fig, axes = plt.subplots(n_views, n_methods + 1, figsize=(4 * (n_methods + 1), 4 * n_views))

    if n_views == 1:
        axes = axes.reshape(1, -1)

    for i, view_idx in enumerate(view_indices):
        # GT
        gt = results[methods[0]]['gt'][i] if results[methods[0]]['gt'] else None
        if gt is not None:
            axes[i, 0].imshow(gt, cmap='gray')
            axes[i, 0].set_title(f'GT (View {view_idx})', fontsize=12)
        axes[i, 0].axis('off')

        # 各方法
        for j, method in enumerate(methods):
            pred = results[method]['pred'][i] if results[method]['pred'] else None
            if pred is not None:
                axes[i, j + 1].imshow(pred, cmap='gray')
                # 只在第一行显示方法名和指标
                if i == 0:
                    m = metrics[method]
                    if m['psnr'] is not None:
                        title_str = f"{method}\nPSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f}"
                    else:
                        title_str = method
                    axes[i, j + 1].set_title(title_str, fontsize=11)
                else:
                    axes[i, j + 1].set_title('', fontsize=12)
            axes[i, j + 1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图保存: {save_path}")


def main():
    parser = ArgumentParser(description="多方法新视角合成对比")
    parser.add_argument("--source_path", type=str, required=True, help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="comparison_renders", help="输出目录")
    parser.add_argument("--view_indices", type=str, default="0,10,20,30,40", help="要渲染的视角索引")
    parser.add_argument("--iteration", type=int, default=30000, help="迭代次数")

    # 各方法的模型路径
    parser.add_argument("--spags_path", type=str, default=None, help="SPAGS 模型路径")
    parser.add_argument("--saxnerf_path", type=str, default=None, help="SAX-NeRF 模型路径")
    parser.add_argument("--naf_path", type=str, default=None, help="NAF 模型路径")
    parser.add_argument("--tensorf_path", type=str, default=None, help="TensoRF 模型路径")
    parser.add_argument("--baseline_path", type=str, default=None, help="Baseline 模型路径")

    args = parser.parse_args()

    view_indices = [int(x) for x in args.view_indices.split(",")]

    print(f"\n{'='*60}")
    print(f"多方法新视角合成对比")
    print(f"数据集: {args.source_path}")
    print(f"视角: {view_indices}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # 渲染 SPAGS
    if args.spags_path:
        print(f"[SPAGS] 渲染中...")
        pred, gt = render_gaussian(args.spags_path, args.source_path, args.iteration, view_indices)
        results['SPAGS'] = {'pred': pred, 'gt': gt}

    # 渲染 Baseline
    if args.baseline_path:
        print(f"[Baseline] 渲染中...")
        pred, gt = render_gaussian(args.baseline_path, args.source_path, args.iteration, view_indices)
        results['Baseline'] = {'pred': pred, 'gt': gt}

    # 渲染 SAX-NeRF
    if args.saxnerf_path:
        print(f"[SAX-NeRF] 渲染中...")
        pred, gt = render_nerf(args.saxnerf_path, args.source_path, "saxnerf", args.iteration, view_indices)
        results['SAX-NeRF'] = {'pred': pred, 'gt': gt}

    # 渲染 NAF
    if args.naf_path:
        print(f"[NAF] 渲染中...")
        pred, gt = render_nerf(args.naf_path, args.source_path, "naf", args.iteration, view_indices)
        results['NAF'] = {'pred': pred, 'gt': gt}

    # 渲染 TensoRF
    if args.tensorf_path:
        print(f"[TensoRF] 渲染中...")
        pred, gt = render_nerf(args.tensorf_path, args.source_path, "tensorf", args.iteration, view_indices)
        results['TensoRF'] = {'pred': pred, 'gt': gt}

    if not results:
        print("错误: 请至少指定一个方法的模型路径")
        return

    # 创建对比图
    save_path = osp.join(args.output_dir, "comparison.png")
    create_comparison_figure(results, view_indices, save_path)

    print(f"\n{'='*60}")
    print(f"完成！对比图保存在: {save_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
