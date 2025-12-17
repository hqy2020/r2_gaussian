#!/usr/bin/env python
"""
NeRF 系列方法重新评估脚本

用法：
    python test_nerf.py --method naf --model_path output/xxx --iteration 30000
    python test_nerf.py --method tensorf --model_path output/xxx --iteration 30000
    python test_nerf.py --method saxnerf --model_path output/xxx --iteration 30000

只评估 2D（跳过 3D 体积评估，加快速度）：
    python test_nerf.py --method naf --model_path output/xxx --iteration 30000 --skip_3d

也可以指定数据集路径（如果与训练时不同）：
    python test_nerf.py --method naf --model_path output/xxx --source_path data/369/foot_3.pickle --iteration 30000
"""

import os
import os.path as osp
import sys
import torch
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append("./")
from r2_gaussian.dataset import Scene
from r2_gaussian.arguments import ModelParams, PipelineParams, get_combined_args
from r2_gaussian.baselines.nerf_base.trainer import (
    NeRFModel,
    get_method_config,
    generate_rays_from_camera,
    _render_full_image,
    _query_volume_by_slices,
)
from r2_gaussian.utils.image_utils import metric_vol, metric_proj


def nerf_eval(method, model, scene, config, scanner_cfg, iteration, save_path, skip_3d=False):
    """NeRF 评估（独立版本）"""
    model.eval()
    # 使用 eval 目录（与训练时一致）
    eval_save_path = osp.join(save_path, "eval", f"iter_{iteration:06d}")
    os.makedirs(eval_save_path, exist_ok=True)

    psnr_2d, ssim_2d = None, None

    with torch.no_grad():
        psnr_3d, ssim_3d = None, None

        # 3D 体积评估
        if not skip_3d:
            vol_gt = scene.vol_gt
            if vol_gt is not None:
                print(f"[{method}] 评估 3D 体积重建...")
                vol_pred_np = _query_volume_by_slices(model, config, scanner_cfg)
                if isinstance(vol_gt, torch.Tensor):
                    vol_gt_np = vol_gt.detach().cpu().numpy()
                else:
                    vol_gt_np = vol_gt

                psnr_3d, _ = metric_vol(vol_gt_np, vol_pred_np, "psnr")
                ssim_3d, _ = metric_vol(vol_gt_np, vol_pred_np, "ssim")

                eval_dict_3d = {"psnr_3d": psnr_3d, "ssim_3d": ssim_3d}
                with open(osp.join(eval_save_path, f"eval3d_{method}.yml"), "w") as f:
                    yaml.dump(eval_dict_3d, f, default_flow_style=False, sort_keys=False)
                with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
                    yaml.dump(eval_dict_3d, f, default_flow_style=False, sort_keys=False)

                print(f"  - PSNR_3D: {psnr_3d:.4f}")
                print(f"  - SSIM_3D: {ssim_3d:.4f}")
        else:
            print(f"[{method}] 跳过 3D 评估")

        # 2D 投影评估
        test_cameras = scene.getTestCameras()
        if test_cameras and len(test_cameras) > 0:
            eval_max_views = int(getattr(config, "eval_max_views", 50))
            chunk_rays = int(getattr(config, "eval_rays_chunk", 8192))

            actual_views = min(eval_max_views, len(test_cameras))
            print(f"[{method}] 评估 2D 投影 ({actual_views} views)...")

            images = []
            gt_images = []

            for viewpoint in tqdm(test_cameras[:eval_max_views], desc="Rendering"):
                pred_image = _render_full_image(
                    viewpoint, scanner_cfg, model, config, chunk_rays=chunk_rays
                )
                gt_image = viewpoint.original_image.to("cuda")
                images.append(pred_image)
                gt_images.append(gt_image)

            images = torch.concat(images, 0).permute(1, 2, 0)
            gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)

            psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
            ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")

            eval_dict_2d = {
                "psnr_2d": psnr_2d,
                "ssim_2d": ssim_2d,
                "psnr_2d_projs": psnr_2d_projs,
                "ssim_2d_projs": ssim_2d_projs,
                "eval_num_views": actual_views,
            }
            with open(osp.join(eval_save_path, f"eval2d_{method}.yml"), "w") as f:
                yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)
            with open(osp.join(eval_save_path, "eval2d_render_test.yml"), "w") as f:
                yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

            print(f"  - PSNR_2D: {psnr_2d:.4f}")
            print(f"  - SSIM_2D: {ssim_2d:.4f}")
        else:
            print(f"[{method}] 没有测试视角可评估")

    print(f"\n[{method}] 评估完成！结果保存在: {eval_save_path}")
    return {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
    }


def main():
    parser = ArgumentParser(description="NeRF 重新评估脚本")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--method", type=str, required=True,
                       choices=["naf", "tensorf", "saxnerf"],
                       help="NeRF 方法名")
    parser.add_argument("--iteration", type=int, default=-1,
                       help="要评估的迭代次数 (-1 表示最新)")
    parser.add_argument("--skip_3d", action="store_true", default=False,
                       help="跳过 3D 体积评估（只评估 2D）")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    method = args.method

    print(f"\n{'='*60}")
    print(f"NeRF 重新评估: {method.upper()}")
    print(f"模型路径: {args.model_path}")
    if args.skip_3d:
        print(f"模式: 只评估 2D（跳过 3D）")
    print(f"{'='*60}\n")

    # 获取配置
    config = get_method_config(method)

    # 加载场景
    dataset = model.extract(args)
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg

    # 创建模型
    nerf_model = NeRFModel(config).cuda()

    # 查找 checkpoint
    iteration = args.iteration
    if iteration == -1:
        # 查找最新的 checkpoint
        ckpt_pattern = f"{method}_iter_"
        ckpt_files = [f for f in os.listdir(args.model_path)
                     if f.startswith(ckpt_pattern) and f.endswith(".pth")]
        if not ckpt_files:
            print(f"错误: 在 {args.model_path} 中找不到 {method} 的 checkpoint")
            return
        # 提取迭代次数并找到最大的
        iterations = [int(f.replace(ckpt_pattern, "").replace(".pth", "")) for f in ckpt_files]
        iteration = max(iterations)
        print(f"自动选择最新 checkpoint: iteration {iteration}")

    ckpt_path = osp.join(args.model_path, f"{method}_iter_{iteration}.pth")
    if not osp.exists(ckpt_path):
        print(f"错误: checkpoint 不存在: {ckpt_path}")
        return

    # 加载 checkpoint
    print(f"加载 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    nerf_model.net.load_state_dict(ckpt["network"])
    if nerf_model.net_fine is not None and ckpt.get("network_fine"):
        nerf_model.net_fine.load_state_dict(ckpt["network_fine"])
    print(f"加载成功！")

    # 评估
    with torch.no_grad():
        results = nerf_eval(
            method, nerf_model, scene, config, scanner_cfg,
            iteration, args.model_path, skip_3d=args.skip_3d
        )

    print(f"\n{'='*60}")
    print(f"最终结果 ({method.upper()}, iter {iteration}):")
    if not args.skip_3d:
        print(f"  PSNR_3D: {results['psnr_3d']:.4f}" if results['psnr_3d'] else "  PSNR_3D: N/A")
        print(f"  SSIM_3D: {results['ssim_3d']:.4f}" if results['ssim_3d'] else "  SSIM_3D: N/A")
    print(f"  PSNR_2D: {results['psnr_2d']:.4f}" if results['psnr_2d'] else "  PSNR_2D: N/A")
    print(f"  SSIM_2D: {results['ssim_2d']:.4f}" if results['ssim_2d'] else "  SSIM_2D: N/A")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
