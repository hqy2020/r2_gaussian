#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPAGS 可视化脚本: 生成渲染图与 Ground Truth 对比 PNG
=====================================================

用法:
    python cc-agent/scripts/save_comparison_png.py --model_path <output_dir> --iteration 30000

输出:
    {model_path}/visualization/
        ├── comparison_train_001.png
        ├── comparison_train_002.png
        ├── ...
        ├── comparison_test_001.png
        ├── comparison_test_002.png
        └── ...
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("./")

from r2_gaussian.arguments import ModelParams, PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render
from r2_gaussian.utils.general_utils import t2a


def load_model(model_path: str, iteration: int, scene: Scene, pipe: PipelineParams):
    """加载训练好的模型"""
    # 确定scale bounds
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
    scale_bound = None

    # 创建 GaussianModel
    gaussians = GaussianModel(scale_bound)

    # 加载checkpoint
    ckpt_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    gaussians.load_ply(ckpt_path)

    # 检查是否有 K-Planes
    kplanes_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "kplanes.pth")
    if os.path.exists(kplanes_path):
        gaussians.load_kplanes(kplanes_path)
        print(f"[INFO] K-Planes loaded from {kplanes_path}")

    return gaussians


def render_and_compare(camera, gaussians, pipe, save_path: str, title: str = ""):
    """渲染并与GT对比，保存为PNG"""
    # 渲染
    with torch.no_grad():
        render_pkg = render(camera, gaussians, pipe)
        rendered = render_pkg["render"]

    # 获取GT
    gt_image = camera.original_image.to("cuda")

    # 转换为numpy
    rendered_np = t2a(rendered[0])  # [H, W]
    gt_np = t2a(gt_image[0])  # [H, W]

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # GT
    axes[0].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    # Rendered
    axes[1].imshow(rendered_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Rendered')
    axes[1].axis('off')

    # Difference
    diff = np.abs(gt_np - rendered_np)
    axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Difference')
    axes[2].axis('off')

    # 添加总标题
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return rendered_np, gt_np


def main():
    parser = argparse.ArgumentParser(description="生成渲染图与GT对比PNG")
    parser.add_argument("--model_path", type=str, required=True, help="模型输出目录")
    parser.add_argument("--iteration", type=int, default=30000, help="加载的iteration")
    parser.add_argument("--num_samples", type=int, default=5, help="每个集合采样数量")
    args = parser.parse_args()

    model_path = args.model_path
    iteration = args.iteration

    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Iteration: {iteration}")

    # 创建可视化目录
    vis_dir = os.path.join(model_path, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    # 读取配置文件获取数据路径
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        print(f"[ERROR] cfg_args not found: {cfg_path}")
        return

    # 解析配置
    with open(cfg_path, 'r') as f:
        cfg_content = f.read()

    # 提取 source_path
    import re
    match = re.search(r"source_path='([^']+)'", cfg_content)
    if not match:
        print("[ERROR] Could not parse source_path from cfg_args")
        return

    source_path = match.group(1)
    print(f"[INFO] Source path: {source_path}")

    # 创建场景和模型参数
    class Args:
        pass

    model_args = Args()
    model_args.source_path = source_path
    model_args.model_path = model_path
    model_args.data_device = "cuda"
    model_args.eval = False
    model_args.ply_path = None
    model_args.scale_min = None
    model_args.scale_max = None

    # FSGS 参数
    model_args.enable_fsgs_proximity = False
    model_args.proximity_threshold = 5.0
    model_args.enable_medical_constraints = False
    model_args.proximity_organ_type = "general"
    model_args.proximity_k_neighbors = 5

    # GAR 参数
    model_args.enable_gar_proximity = False
    model_args.gar_proximity_threshold = 5.0
    model_args.gar_medical_constraints = False
    model_args.gar_organ_type = "general"
    model_args.gar_proximity_k = 5

    # ADM/K-Planes 参数
    model_args.enable_kplanes = True
    model_args.kplanes_resolution = 64
    model_args.kplanes_dim = 32
    model_args.kplanes_decoder_hidden = 64
    model_args.kplanes_decoder_layers = 2
    model_args.enable_adm = False
    model_args.adm_resolution = 64
    model_args.adm_feature_dim = 32
    model_args.adm_decoder_hidden = 64
    model_args.adm_decoder_layers = 2

    pipe_args = Args()
    pipe_args.compute_cov3D_python = False
    pipe_args.debug = False

    # 加载场景
    print("[INFO] Loading scene...")
    scene = Scene(model_args, shuffle=False)

    # 加载模型
    print(f"[INFO] Loading model from iteration {iteration}...")
    try:
        gaussians = load_model(model_path, iteration, scene, pipe_args)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    gaussians.cuda()

    # 渲染训练集样本
    train_cameras = scene.getTrainCameras()
    n_train = len(train_cameras)
    sample_indices_train = np.linspace(0, n_train - 1, min(args.num_samples, n_train)).astype(int)

    print(f"[INFO] Rendering {len(sample_indices_train)} train samples...")
    for idx in sample_indices_train:
        camera = train_cameras[idx]
        save_path = os.path.join(vis_dir, f"comparison_train_{idx:03d}.png")
        title = f"Train View {idx}: {camera.image_name}"
        render_and_compare(camera, gaussians, pipe_args, save_path, title)
        print(f"  Saved: {save_path}")

    # 渲染测试集样本
    test_cameras = scene.getTestCameras()
    n_test = len(test_cameras)
    sample_indices_test = np.linspace(0, n_test - 1, min(args.num_samples, n_test)).astype(int)

    print(f"[INFO] Rendering {len(sample_indices_test)} test samples...")
    for idx in sample_indices_test:
        camera = test_cameras[idx]
        save_path = os.path.join(vis_dir, f"comparison_test_{idx:03d}.png")
        title = f"Test View {idx}: {camera.image_name}"
        render_and_compare(camera, gaussians, pipe_args, save_path, title)
        print(f"  Saved: {save_path}")

    print(f"\n[SUCCESS] Visualization saved to: {vis_dir}")
    print(f"  Total images: {len(sample_indices_train) + len(sample_indices_test)}")


if __name__ == "__main__":
    main()
