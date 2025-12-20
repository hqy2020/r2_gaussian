#
# DNGaussian 训练函数
#
# 核心创新：深度正则化 + Neural Renderer opacity 调制
#

import os
import os.path as osp
import torch
from random import randint
import numpy as np
import yaml
from tqdm import tqdm
import imageio

from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice
from r2_gaussian.utils.unified_logger import get_logger

from .model import DNGaussianModel
from .renderer import render_dngaussian, render_dngaussian_depth, query_dngaussian
from .config import DNGaussianConfig
from .loss_utils import (
    patch_norm_mse_loss,
    patch_norm_mse_loss_global,
)


def training_dngaussian(
    dataset,
    opt,
    pipe,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    """
    DNGaussian 训练函数

    Args:
        dataset: ModelParams (数据集参数)
        opt: OptimizationParams (优化参数)
        pipe: PipelineParams (管线参数)
        tb_writer: TensorBoard writer
        testing_iterations: 测试迭代列表
        saving_iterations: 保存迭代列表
        checkpoint_iterations: 检查点迭代列表
        checkpoint: 起始检查点路径
    """
    first_iter = 0

    # 加载场景
    scene = Scene(dataset, shuffle=False)

    # 获取配置
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])

    # 尺度边界
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    # 体积查询函数
    queryfunc = lambda x: query_dngaussian(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # 创建 DNGaussian 模型
    config = DNGaussianConfig()
    gaussians = DNGaussianModel(scale_bound=scale_bound, args=dataset, config=config)

    # 从 R²-Gaussian 初始化文件加载点云
    if dataset.ply_path and osp.exists(dataset.ply_path):
        gaussians.create_from_r2_init(dataset.ply_path, spatial_lr_scale=1.0)
    else:
        raise ValueError(
            f"DNGaussian requires initialization file. "
            f"Please run initialize_pcd.py first or provide --ply_path"
        )

    scene.gaussians = gaussians

    # 设置优化参数
    gaussians.training_setup(opt)

    logger = get_logger()

    # 加载检查点
    if checkpoint is not None:
        state, first_iter = torch.load(checkpoint)
        gaussians.restore(state, opt)
        logger.config(f"Loaded DNGaussian checkpoint from {checkpoint}")

    # 深度正则化参数
    hard_depth_start = getattr(opt, 'hard_depth_start', config.hard_depth_start)
    soft_depth_start = getattr(opt, 'soft_depth_start', config.soft_depth_start)
    depth_end_iter = getattr(opt, 'depth_end_iter', config.depth_end_iter)
    patch_size_min = getattr(opt, 'patch_size_min', config.patch_size_min)
    patch_size_max = getattr(opt, 'patch_size_max', config.patch_size_max)
    error_tolerance = getattr(opt, 'error_tolerance', config.error_tolerance)
    lambda_depth_local = getattr(opt, 'lambda_depth_local', config.lambda_depth_local)
    lambda_depth_global = getattr(opt, 'lambda_depth_global', config.lambda_depth_global)

    # 密集化参数
    densify_from_iter = getattr(opt, 'densify_from_iter', config.densify_from_iter)
    densify_until_iter = getattr(opt, 'densify_until_iter', config.densify_until_iter)
    densification_interval = getattr(opt, 'densification_interval', config.densification_interval)
    densify_grad_threshold = getattr(opt, 'densify_grad_threshold', config.densify_grad_threshold)
    opacity_reset_interval = getattr(opt, 'opacity_reset_interval', config.opacity_reset_interval)
    min_opacity = getattr(opt, 'min_opacity', config.min_opacity)
    lambda_dssim = getattr(opt, 'lambda_dssim', config.lambda_dssim)

    # 场景范围
    cameras_extent = scene.scene_scale

    # 训练循环
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="DNGaussian Training")
    first_iter += 1

    logger.config_block("DNGaussian Training Configuration", {
        "Neural Renderer": f"Hash Grid ({config.num_levels} levels) + MLP",
        "Depth regularization": f"{hard_depth_start} - {depth_end_iter} iterations",
        "Patch size": f"{patch_size_min} - {patch_size_max}",
        "Depth prior": "viewpoint_cam.depth_mono (if provided)",
    })

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 随机选择视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render_dngaussian(viewpoint_cam, gaussians, pipe)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # GT 图像
        gt_image = viewpoint_cam.original_image.to("cuda")

        # 计算损失
        loss_dict = {}

        # 1. 渲染损失 (L1 + SSIM)
        Ll1 = l1_loss(image, gt_image)
        loss_dict["l1"] = Ll1.item()
        loss = Ll1

        if lambda_dssim > 0:
            loss_ssim = 1.0 - ssim(image, gt_image)
            loss_dict["ssim"] = loss_ssim.item()
            loss = loss + lambda_dssim * loss_ssim

        # 2. 深度正则化（DNGaussian 核心创新）
        # 仅在数据提供了可用的深度先验时启用（例如 monocular depth）。
        depth_gt = None
        if hasattr(viewpoint_cam, "depth_mono") and viewpoint_cam.depth_mono is not None:
            depth_gt = viewpoint_cam.depth_mono
            if not torch.is_tensor(depth_gt):
                depth_gt = torch.as_tensor(depth_gt)
            depth_gt = depth_gt.to("cuda").float()
            if depth_gt.max() > 1.5:
                depth_gt = 255.0 - depth_gt

        use_depth_reg = (
            depth_gt is not None
            and iteration >= hard_depth_start
            and iteration < depth_end_iter
        )

        if use_depth_reg:
            # 随机选择 patch 大小
            patch_size = randint(patch_size_min // 2, patch_size_max // 2) * 2 + 1

            # 渲染深度图
            depth_render = render_dngaussian_depth(
                viewpoint_cam, gaussians, pipe, use_base_density=True
            )

            # 确保形状匹配
            if depth_render.dim() == 3:
                depth_render = depth_render.unsqueeze(0)
            if depth_gt.dim() == 2:
                depth_gt = depth_gt.unsqueeze(0).unsqueeze(0)
            elif depth_gt.dim() == 3:
                depth_gt = depth_gt.unsqueeze(0)

            # 全局深度损失
            loss_depth_global = patch_norm_mse_loss_global(
                depth_render, depth_gt, patch_size, error_tolerance
            )
            loss_dict["depth_global"] = loss_depth_global.item()
            loss = loss + lambda_depth_global * loss_depth_global

            # 局部深度损失
            loss_depth_local = patch_norm_mse_loss(
                depth_render, depth_gt, patch_size, error_tolerance
            )
            loss_dict["depth_local"] = loss_depth_local.item()
            loss = loss + lambda_depth_local * loss_depth_local

        loss_dict["total"] = loss.item()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 更新进度
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.6f}",
                    "Points": f"{gaussians.get_num_points()}"
                }
                if use_depth_reg:
                    postfix["Depth"] = "ON"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # TensorBoard 日志
            if tb_writer:
                for key, val in loss_dict.items():
                    tb_writer.add_scalar(f"train/loss_{key}", val, iteration)
                tb_writer.add_scalar("train/total_points", gaussians.get_num_points(), iteration)

            # 评估
            if iteration in testing_iterations:
                _dngaussian_eval(
                    tb_writer, iteration, scene, gaussians, pipe, queryfunc
                )

            # 保存
            if iteration in saving_iterations:
                logger.info("Saving DNGaussian model", iteration=iteration)
                save_path = osp.join(scene.model_path, f"dngaussian_iter_{iteration}.pth")
                torch.save((gaussians.capture(), iteration), save_path)
                scene.save(iteration, queryfunc)

            # 密集化
            if iteration < densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        densify_grad_threshold, min_opacity, cameras_extent, size_threshold
                    )

                if iteration % opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            # 优化步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 检查点
            if iteration in checkpoint_iterations:
                logger.info("Saving Checkpoint", iteration=iteration)
                ckpt_path = osp.join(scene.model_path, f"chkpnt_dngaussian_{iteration}.pth")
                torch.save((gaussians.capture(), iteration), ckpt_path)


def _dngaussian_eval(tb_writer, iteration, scene, gaussians, pipe, queryfunc):
    """DNGaussian 评估"""
    eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
    os.makedirs(eval_save_path, exist_ok=True)
    torch.cuda.empty_cache()

    # 2D 评估
    test_cameras = scene.getTestCameras()
    psnr_2d, ssim_2d = 0.0, 0.0

    if test_cameras and len(test_cameras) > 0:
        images = []
        gt_images = []

        for viewpoint in test_cameras:
            render_result = render_dngaussian(viewpoint, gaussians, pipe)
            image = render_result["render"]
            gt_image = viewpoint.original_image.to("cuda")
            images.append(image)
            gt_images.append(gt_image)

        images = torch.concat(images, 0).permute(1, 2, 0)
        gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)

        psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
        ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")

        eval_dict_2d = {
            "psnr_2d": psnr_2d,
            "ssim_2d": ssim_2d,
        }
        with open(osp.join(eval_save_path, "eval2d_dngaussian.yml"), "w") as f:
            yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

        if tb_writer:
            tb_writer.add_scalar("dngaussian/psnr_2d", psnr_2d, iteration)
            tb_writer.add_scalar("dngaussian/ssim_2d", ssim_2d, iteration)

    # 3D 评估
    vol_pred = queryfunc(gaussians)["vol"]
    vol_gt = scene.vol_gt
    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")

    eval_dict_3d = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
    }
    with open(osp.join(eval_save_path, "eval3d_dngaussian.yml"), "w") as f:
        yaml.dump(eval_dict_3d, f, default_flow_style=False, sort_keys=False)

    if tb_writer:
        tb_writer.add_scalar("dngaussian/psnr_3d", psnr_3d, iteration)
        tb_writer.add_scalar("dngaussian/ssim_3d", ssim_3d, iteration)

    # 保存可视化
    logger = get_logger()
    try:
        # 取中间切片进行可视化
        mid_slice = vol_gt.shape[0] // 2
        slice_gt = vol_gt[mid_slice].cpu().numpy()
        slice_pred = vol_pred[mid_slice].cpu().numpy()
        vis_data = show_two_slice(
            slice_gt,
            slice_pred,
            "GT",
            "DNGaussian",
            save=True,
        )
        imageio.imwrite(osp.join(eval_save_path, "slices_dngaussian.png"), vis_data)
    except Exception as e:
        logger.warn(f"Failed to save slice visualization: {e}", iteration=iteration)

    logger.eval(f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}", iteration=iteration)

    torch.cuda.empty_cache()
