#
# X-Gaussian training function adapted for r2_gaussian framework
#

import os
import os.path as osp
import torch
from random import randint
import numpy as np
import yaml
from tqdm import tqdm

from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice

from .model import XGaussianModel
from .renderer import render_xgaussian, query_xgaussian
from .config import XGaussianConfig


def training_xgaussian(
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
    X-Gaussian 训练函数

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

    # 体积查询函数（使用 X-Gaussian 专用查询）
    queryfunc = lambda x: query_xgaussian(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # 创建 X-Gaussian 模型
    xg_config = XGaussianConfig()
    sh_degree = getattr(dataset, 'sh_degree', xg_config.sh_degree)
    gaussians = XGaussianModel(sh_degree=sh_degree, scale_bound=scale_bound)

    # 从 R²-Gaussian 初始化文件加载点云
    if dataset.ply_path and osp.exists(dataset.ply_path):
        gaussians.create_from_r2_init(dataset.ply_path, spatial_lr_scale=1.0)
    else:
        # 如果没有初始化文件，尝试从场景创建
        raise ValueError(
            f"X-Gaussian requires initialization file. "
            f"Please run initialize_pcd.py first or provide --ply_path"
        )

    scene.gaussians = gaussians

    # 设置优化参数
    # 合并 X-Gaussian 默认参数和用户参数
    class XGOptParams:
        def __init__(self, opt, xg_config):
            # 从 opt 获取参数，如果没有则使用 X-Gaussian 默认值
            self.percent_dense = getattr(opt, 'percent_dense', xg_config.percent_dense)
            self.position_lr_init = getattr(opt, 'position_lr_init', xg_config.position_lr_init)
            self.position_lr_final = getattr(opt, 'position_lr_final', xg_config.position_lr_final)
            self.position_lr_delay_mult = getattr(opt, 'position_lr_delay_mult', xg_config.position_lr_delay_mult)
            self.position_lr_max_steps = getattr(opt, 'position_lr_max_steps', xg_config.position_lr_max_steps)
            self.feature_lr = getattr(opt, 'feature_lr', xg_config.feature_lr)
            self.opacity_lr = getattr(opt, 'opacity_lr', xg_config.opacity_lr)
            self.scaling_lr = getattr(opt, 'scaling_lr', xg_config.scaling_lr)
            self.rotation_lr = getattr(opt, 'rotation_lr', xg_config.rotation_lr)

    xg_opt = XGOptParams(opt, xg_config)
    gaussians.training_setup(xg_opt)

    # 加载检查点
    if checkpoint is not None:
        state, first_iter = torch.load(checkpoint)
        gaussians.restore(state, xg_opt)
        print(f"Loaded X-Gaussian checkpoint from {checkpoint}")

    # 密集化参数
    densify_from_iter = getattr(opt, 'densify_from_iter', xg_config.densify_from_iter)
    densify_until_iter = getattr(opt, 'densify_until_iter', xg_config.densify_until_iter)
    densification_interval = getattr(opt, 'densification_interval', xg_config.densification_interval)
    densify_grad_threshold = getattr(opt, 'densify_grad_threshold', xg_config.densify_grad_threshold)
    opacity_reset_interval = getattr(opt, 'opacity_reset_interval', xg_config.opacity_reset_interval)
    min_opacity = getattr(opt, 'min_opacity', xg_config.min_opacity)
    lambda_dssim = getattr(opt, 'lambda_dssim', xg_config.lambda_dssim)

    # 场景范围
    cameras_extent = scene.scene_scale

    # 训练循环
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="X-Gaussian Training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # 每 1000 迭代升级球谐阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 渲染
        render_pkg = render_xgaussian(viewpoint_cam, gaussians, pipe)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # GT 图像
        gt_image = viewpoint_cam.original_image.to("cuda")

        # 计算损失
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
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
                tb_writer.add_scalar("train/loss_l1", Ll1.item(), iteration)
                tb_writer.add_scalar("train/loss_total", loss.item(), iteration)
                tb_writer.add_scalar("train/total_points", gaussians.get_num_points(), iteration)

            # 评估
            if iteration in testing_iterations:
                _xgaussian_eval(
                    tb_writer, iteration, scene, gaussians, pipe, queryfunc
                )

            # 保存
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving X-Gaussian model")
                save_path = osp.join(scene.model_path, f"xgaussian_iter_{iteration}.pth")
                torch.save((gaussians.capture(), iteration), save_path)

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
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                ckpt_path = osp.join(scene.model_path, f"chkpnt_xgaussian_{iteration}.pth")
                torch.save((gaussians.capture(), iteration), ckpt_path)


def _xgaussian_eval(tb_writer, iteration, scene, gaussians, pipe, queryfunc):
    """X-Gaussian 评估"""
    eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
    os.makedirs(eval_save_path, exist_ok=True)
    torch.cuda.empty_cache()

    # 2D 评估
    test_cameras = scene.getTestCameras()
    if test_cameras and len(test_cameras) > 0:
        images = []
        gt_images = []

        for viewpoint in test_cameras:
            render_result = render_xgaussian(viewpoint, gaussians, pipe)
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
        with open(osp.join(eval_save_path, "eval2d_xgaussian.yml"), "w") as f:
            yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

        if tb_writer:
            tb_writer.add_scalar("xgaussian/psnr_2d", psnr_2d, iteration)
            tb_writer.add_scalar("xgaussian/ssim_2d", ssim_2d, iteration)

    # 3D 评估
    vol_pred = queryfunc(gaussians)["vol"]
    vol_gt = scene.vol_gt
    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")

    eval_dict_3d = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
    }
    with open(osp.join(eval_save_path, "eval3d_xgaussian.yml"), "w") as f:
        yaml.dump(eval_dict_3d, f, default_flow_style=False, sort_keys=False)

    if tb_writer:
        tb_writer.add_scalar("xgaussian/psnr_3d", psnr_3d, iteration)
        tb_writer.add_scalar("xgaussian/ssim_3d", ssim_3d, iteration)

    tqdm.write(
        f"[X-Gaussian ITER {iteration}] "
        f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, "
        f"psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
    )

    torch.cuda.empty_cache()
