#!/usr/bin/env python3
"""
X-Gaussian 模型渲染脚本

用于从 xgaussian_iter_*.pth 模型文件生成 2D 渲染图像。
"""

import os
import os.path as osp
import sys
import torch
import torchvision
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, PipelineParams
from r2_gaussian.dataset import Scene
from r2_gaussian.baselines.xgaussian.model import XGaussianModel
from r2_gaussian.baselines.xgaussian.renderer import render_xgaussian


def render_and_save(
    model_path: str,
    source_path: str,
    iteration: int,
    max_views: int = 10,
    skip_train: bool = True,
):
    """加载 X-Gaussian 模型并渲染保存图像"""

    # 加载配置
    cfg_path = osp.join(model_path, "cfg_args")
    if not osp.exists(cfg_path):
        print(f"[错误] 找不到配置文件: {cfg_path}")
        return

    # 从配置文件加载参数
    with open(cfg_path, "r") as f:
        cfg_str = f.read()
    cfg_args = eval(cfg_str)
    cfg_args.source_path = source_path
    cfg_args.model_path = model_path

    # 创建参数
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    dataset = model_params.extract(cfg_args)
    pipe = pipeline_params.extract(cfg_args)

    # 加载场景
    scene = Scene(dataset, shuffle=False)

    # 创建 X-Gaussian 模型
    from r2_gaussian.baselines.xgaussian.config import XGaussianConfig
    xg_config = XGaussianConfig()
    sh_degree = getattr(dataset, 'sh_degree', xg_config.sh_degree)
    gaussians = XGaussianModel(sh_degree=sh_degree, scale_bound=None)

    # 加载模型权重
    pth_path = osp.join(model_path, f"xgaussian_iter_{iteration}.pth")
    if not osp.exists(pth_path):
        print(f"[错误] 找不到模型文件: {pth_path}")
        return

    state, loaded_iter = torch.load(pth_path)
    # 手动恢复模型状态，不调用 training_setup
    gaussians.active_sh_degree = state['active_sh_degree']
    gaussians._xyz = state['_xyz']
    gaussians._features_dc = state['_features_dc']
    gaussians._features_rest = state['_features_rest']
    gaussians._scaling = state['_scaling']
    gaussians._rotation = state['_rotation']
    gaussians._opacity = state['_opacity']
    gaussians.max_radii2D = state['max_radii2D']
    gaussians.spatial_lr_scale = state['spatial_lr_scale']
    print(f"Loaded X-Gaussian model at iteration {loaded_iter}")

    # 渲染测试视图
    test_cameras = scene.getTestCameras()
    if max_views > 0:
        test_cameras = test_cameras[:max_views]

    save_dir = osp.join(model_path, "test", f"iter_{loaded_iter}", "render_test")
    os.makedirs(save_dir, exist_ok=True)

    images = []
    gt_images = []

    for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Rendering")):
        render_result = render_xgaussian(viewpoint, gaussians, pipe)
        image = render_result["render"]
        gt_image = viewpoint.original_image

        images.append(image)
        gt_images.append(gt_image)

        # 保存图像
        torchvision.utils.save_image(image, osp.join(save_dir, f"{idx:03d}_pred.png"))
        torchvision.utils.save_image(gt_image, osp.join(save_dir, f"{idx:03d}_gt.png"))

    # 计算指标
    from r2_gaussian.utils.loss_utils import ssim
    psnr_list = []
    ssim_list = []
    for pred, gt in zip(images, gt_images):
        mse = torch.mean((pred - gt) ** 2)
        psnr = 10 * torch.log10(1.0 / mse)
        psnr_list.append(psnr.item())
        s = ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        ssim_list.append(s.item())
    psnr_2d = sum(psnr_list) / len(psnr_list)
    ssim_2d = sum(ssim_list) / len(ssim_list)

    print(f"render_test complete. psnr_2d: {psnr_2d}, ssim_2d: {ssim_2d}")

    # 保存指标
    eval_dict = {
        "psnr_2d": float(psnr_2d),
        "ssim_2d": float(ssim_2d),
    }
    with open(osp.join(save_dir, "..", "eval2d_render_test.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="X-Gaussian 渲染脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--source_path", type=str, required=True, help="数据源路径")
    parser.add_argument("--iteration", type=int, default=30000, help="迭代次数")
    parser.add_argument("--max_views", type=int, default=10, help="最大渲染视图数")
    args = parser.parse_args()

    render_and_save(
        model_path=args.model_path,
        source_path=args.source_path,
        iteration=args.iteration,
        max_views=args.max_views,
    )


if __name__ == "__main__":
    main()
