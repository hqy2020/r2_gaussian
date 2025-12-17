#!/usr/bin/env python
"""
NeRF 渲染新视角合成图像
"""

import os
import os.path as osp
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import torchvision

sys.path.append("./")
from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams, PipelineParams, get_combined_args
from r2_gaussian.baselines.nerf_base.trainer import (
    NeRFModel,
    get_method_config,
    _render_full_image,
)


def render_and_save(method, model, scene, config, scanner_cfg, iteration, save_path, max_views=10):
    """渲染并保存图像"""
    model.eval()

    render_save_path = osp.join(save_path, "renders", f"iter_{iteration:06d}")
    os.makedirs(render_save_path, exist_ok=True)

    test_cameras = scene.getTestCameras()
    chunk_rays = int(getattr(config, "eval_rays_chunk", 8192))

    actual_views = min(max_views, len(test_cameras))
    print(f"[{method}] 渲染 {actual_views} 个新视角...")

    with torch.no_grad():
        for i, viewpoint in enumerate(tqdm(test_cameras[:actual_views], desc="Rendering")):
            # 渲染预测图像
            pred_image = _render_full_image(
                viewpoint, scanner_cfg, model, config, chunk_rays=chunk_rays
            )
            gt_image = viewpoint.original_image.to("cuda")

            # 保存图像
            pred_np = pred_image[0].cpu().numpy()
            gt_np = gt_image[0].cpu().numpy()

            # 归一化到 0-1
            pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
            gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)

            # 保存为 PNG
            torchvision.utils.save_image(
                torch.from_numpy(pred_np).unsqueeze(0),
                osp.join(render_save_path, f"pred_{i:03d}.png")
            )
            torchvision.utils.save_image(
                torch.from_numpy(gt_np).unsqueeze(0),
                osp.join(render_save_path, f"gt_{i:03d}.png")
            )

            # 保存对比图（左GT右Pred）
            comparison = np.concatenate([gt_np, pred_np], axis=1)
            torchvision.utils.save_image(
                torch.from_numpy(comparison).unsqueeze(0),
                osp.join(render_save_path, f"compare_{i:03d}.png")
            )

    print(f"\n渲染完成！图像保存在: {render_save_path}")
    return render_save_path


def main():
    parser = ArgumentParser(description="NeRF 渲染脚本")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--method", type=str, required=True,
                       choices=["naf", "tensorf", "saxnerf"])
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--max_views", type=int, default=10,
                       help="最多渲染多少个视角")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    method = args.method

    print(f"\n{'='*60}")
    print(f"NeRF 新视角渲染: {method.upper()}")
    print(f"模型路径: {args.model_path}")
    print(f"{'='*60}\n")

    config = get_method_config(method)
    dataset = model.extract(args)
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg

    nerf_model = NeRFModel(config).cuda()

    iteration = args.iteration
    ckpt_path = osp.join(args.model_path, f"{method}_iter_{iteration}.pth")

    print(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    nerf_model.net.load_state_dict(ckpt["network"])
    if nerf_model.net_fine is not None and ckpt.get("network_fine"):
        nerf_model.net_fine.load_state_dict(ckpt["network_fine"])
    print(f"加载成功！")

    with torch.no_grad():
        render_path = render_and_save(
            method, nerf_model, scene, config, scanner_cfg,
            iteration, args.model_path, max_views=args.max_views
        )

    print(f"\n{'='*60}")
    print(f"渲染完成！")
    print(f"图像保存在: {render_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
