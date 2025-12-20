#
# FSGS training function adapted for CT reconstruction
#
# Based on: FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting
# Paper: https://arxiv.org/abs/2312.00451
#
# 核心差异 (相比 X-Gaussian):
# - 使用 FSGSModel (带 confidence 和 proximity)
# - 可选伪视角训练
# - proximity 密化 (iter < 2000)
# - SH 升级 (每 500 迭代)
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

from .model import FSGSModel
from .renderer import render_fsgs, query_fsgs
from .config import FSGSConfig
from .pseudo_camera import CTPseudoCameraGenerator, tv_loss


def training_fsgs(
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
    FSGS 训练函数 (CT 适配版)

    核心功能:
    - proximity 密化 (FSGS 核心, iter < 2000)
    - 伪视角深度 TV 正则化
    - confidence 加权渲染
    - 动态 SH 升级

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
    queryfunc = lambda x: query_fsgs(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # 创建 FSGS 配置
    fsgs_config = FSGSConfig()

    # 从 opt 覆盖部分参数
    if hasattr(opt, 'densify_grad_threshold'):
        fsgs_config.densify_grad_threshold = opt.densify_grad_threshold
    if hasattr(opt, 'densify_until_iter'):
        fsgs_config.densify_until_iter = opt.densify_until_iter
    if hasattr(opt, 'min_opacity'):
        fsgs_config.min_opacity = opt.min_opacity

    # 创建 FSGS 模型
    gaussians = FSGSModel(config=fsgs_config, scale_bound=scale_bound)

    # 从 SPS 初始化文件加载点云
    if dataset.ply_path and osp.exists(dataset.ply_path):
        gaussians.create_from_r2_init(dataset.ply_path, spatial_lr_scale=1.0)
    else:
        raise ValueError(
            f"FSGS requires initialization file. "
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
        logger.config(f"Loaded FSGS checkpoint from {checkpoint}")

    # 生成伪相机 (CT 适配)
    pseudo_generator = None
    if fsgs_config.enable_pseudo_view:
        pseudo_generator = CTPseudoCameraGenerator(
            train_cameras=scene.getTrainCameras(),
            scanner_cfg=scanner_cfg,
            n_pseudo=fsgs_config.n_pseudo_cameras,
        )

    # 密集化参数
    densify_from_iter = fsgs_config.densify_from_iter
    densify_until_iter = fsgs_config.densify_until_iter
    densification_interval = fsgs_config.densification_interval
    densify_grad_threshold = fsgs_config.densify_grad_threshold
    opacity_reset_interval = fsgs_config.opacity_reset_interval
    min_opacity = fsgs_config.min_opacity
    lambda_dssim = fsgs_config.lambda_dssim

    # 场景范围
    cameras_extent = scene.scene_scale

    # 训练循环
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="FSGS Training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # SH 升级 (每 sh_upgrade_interval 迭代)
        if iteration % fsgs_config.sh_upgrade_interval == 0:
            gaussians.oneupSHdegree()

        # 随机选择视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render_fsgs(viewpoint_cam, gaussians, pipe)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # GT 图像
        gt_image = viewpoint_cam.original_image.to("cuda")

        # 计算损失
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))

        # ============ 伪视角深度 TV 正则化 (CT 适配) ============
        if (fsgs_config.enable_pseudo_view and
            pseudo_generator is not None and
            iteration > fsgs_config.pseudo_view_start_iter and
            iteration < fsgs_config.pseudo_view_end_iter and
            iteration % fsgs_config.pseudo_view_interval == 0):

            pseudo_cam = pseudo_generator.get_pseudo_camera()
            if pseudo_cam is not None:
                render_pkg_pseudo = render_fsgs(pseudo_cam, gaussians, pipe)
                rendered_pseudo = render_pkg_pseudo["render"]

                # 深度 TV 正则化 (替代 MiDAS 深度监督)
                depth_tv = tv_loss(rendered_pseudo)
                loss_scale = min((iteration - fsgs_config.pseudo_view_start_iter) / 500., 1)
                loss += loss_scale * fsgs_config.pseudo_view_depth_tv_weight * depth_tv

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 更新进度
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.6f}",
                    "Points": f"{gaussians.get_num_points()}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # TensorBoard 日志
            if tb_writer:
                tb_writer.add_scalar("train_fsgs/loss_l1", Ll1.item(), iteration)
                tb_writer.add_scalar("train_fsgs/loss_total", loss.item(), iteration)
                tb_writer.add_scalar("train_fsgs/total_points", gaussians.get_num_points(), iteration)

            # 评估
            if iteration in testing_iterations:
                _fsgs_eval(
                    tb_writer, iteration, scene, gaussians, pipe, queryfunc
                )

            # 保存
            if iteration in saving_iterations:
                logger.info("Saving FSGS model", iteration=iteration)
                save_path = osp.join(scene.model_path, f"fsgs_iter_{iteration}.pth")
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
                        densify_grad_threshold, min_opacity, cameras_extent, size_threshold,
                        iteration=iteration  # 传递 iteration 用于 proximity 判断
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
                ckpt_path = osp.join(scene.model_path, f"chkpnt_fsgs_{iteration}.pth")
                torch.save((gaussians.capture(), iteration), ckpt_path)


def _fsgs_eval(tb_writer, iteration, scene, gaussians, pipe, queryfunc):
    """FSGS 评估"""
    eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
    os.makedirs(eval_save_path, exist_ok=True)
    torch.cuda.empty_cache()

    # 2D 评估
    test_cameras = scene.getTestCameras()
    psnr_2d = 0.0
    ssim_2d = 0.0

    if test_cameras and len(test_cameras) > 0:
        images = []
        gt_images = []

        for viewpoint in test_cameras:
            render_result = render_fsgs(viewpoint, gaussians, pipe)
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
        with open(osp.join(eval_save_path, "eval2d_fsgs.yml"), "w") as f:
            yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

        if tb_writer:
            tb_writer.add_scalar("fsgs/psnr_2d", psnr_2d, iteration)
            tb_writer.add_scalar("fsgs/ssim_2d", ssim_2d, iteration)

    # 3D 评估
    vol_pred = queryfunc(gaussians)["vol"]
    vol_gt = scene.vol_gt
    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")

    eval_dict_3d = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
    }
    with open(osp.join(eval_save_path, "eval3d_fsgs.yml"), "w") as f:
        yaml.dump(eval_dict_3d, f, default_flow_style=False, sort_keys=False)

    if tb_writer:
        tb_writer.add_scalar("fsgs/psnr_3d", psnr_3d, iteration)
        tb_writer.add_scalar("fsgs/ssim_3d", ssim_3d, iteration)

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
            "FSGS",
            save=True,
        )
        imageio.imwrite(osp.join(eval_save_path, "slices_fsgs.png"), vis_data)
    except Exception as e:
        logger.warn(f"Failed to save slice visualization: {e}", iteration=iteration)

    logger.eval(f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}", iteration=iteration)

    torch.cuda.empty_cache()
