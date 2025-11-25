#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj

# CoR-GS Stage 3: Pseudo-view Co-regularization (官方对齐版)
try:
    from r2_gaussian.utils.pseudo_view_coreg import (
        generate_pseudo_view_medical,
        compute_pseudo_coreg_loss_medical,
        PseudoViewPool  # 新增：pseudo-view 池支持
    )
    HAS_PSEUDO_COREG = True
    print("✅ CoR-GS Stage 3 Pseudo-view Co-regularization modules loaded (官方对齐版)")
except ImportError as e:
    HAS_PSEUDO_COREG = False
    print(f"📦 Pseudo-view Co-regularization modules not available: {e}")
from r2_gaussian.utils.plot_utils import show_two_slice


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians (支持多模型 CoR-GS 训练)
    gaussiansN = dataset.gaussiansN  # 获取模型数量（默认 2）
    gaussians_list = []

    print(f"🔧 初始化 {gaussiansN} 个 Gaussian 模型...")
    for idx in range(gaussiansN):
        gs = GaussianModel(scale_bound)
        initialize_gaussian(gs, dataset, None)
        gs.training_setup(opt)
        gaussians_list.append(gs)
        print(f"  ✓ 模型 {idx+1}/{gaussiansN} 初始化完成")

    # 向下兼容：单模型时使用原有变量名
    if gaussiansN == 1:
        gaussians = gaussians_list[0]
        scene.gaussians = gaussians
    else:
        gaussians = gaussians_list[0]  # 默认使用第一个模型（用于兼容性）
        scene.gaussians = gaussians_list  # 多模型时存储列表

    # 加载 checkpoint（支持多模型）
    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint)
        if gaussiansN == 1:
            (model_params, first_iter) = checkpoint_data
            gaussians.restore(model_params, opt)
        else:
            # 多模型 checkpoint 加载逻辑
            if isinstance(checkpoint_data, tuple) and len(checkpoint_data) == 2:
                # 单模型 checkpoint -> 复制到所有模型
                (model_params, first_iter) = checkpoint_data
                for idx, gs in enumerate(gaussians_list):
                    gs.restore(model_params, opt)
                    print(f"  ✓ 模型 {idx+1} 从 checkpoint 加载")
            else:
                # 多模型 checkpoint（未来扩展）
                print("⚠️ 多模型 checkpoint 加载暂未实现，使用随机初始化")
        print(f"✓ Checkpoint {osp.basename(checkpoint)} 加载完成")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # ========== CoR-GS: 初始化 Pseudo-view 池（官方对齐版）==========
    pseudo_view_pool = None
    if (HAS_PSEUDO_COREG and
        gaussiansN >= 2 and
        hasattr(dataset, 'enable_corgs') and dataset.enable_corgs):

        # 获取池参数
        pool_size = getattr(dataset, 'corgs_pool_size', 1000)  # 默认 1000 个
        pool_strategy = getattr(dataset, 'corgs_pool_strategy', 'slerp')  # 'slerp' 或 'random'
        noise_std = getattr(dataset, 'corgs_pseudo_noise_std', 0.02)

        try:
            pseudo_view_pool = PseudoViewPool(
                train_cameras=scene.getTrainCameras(),
                num_pseudo=pool_size,
                strategy=pool_strategy,
                noise_std=noise_std,
                seed=42
            )
        except Exception as e:
            print(f"⚠️ Pseudo-view 池初始化失败，将使用实时生成: {e}")
            pseudo_view_pool = None

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate (所有模型)
        for idx in range(gaussiansN):
            current_gs = gaussians_list[idx] if gaussiansN > 1 else gaussians
            current_gs.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection (多模型)
        renders = []
        viewspace_points = []
        visibility_filters = []
        radiis = []

        for idx in range(gaussiansN):
            current_gs = gaussians_list[idx] if gaussiansN > 1 else gaussians
            pkg = render(viewpoint_cam, current_gs, pipe)
            renders.append(pkg)
            viewspace_points.append(pkg["viewspace_points"])
            visibility_filters.append(pkg["visibility_filter"])
            radiis.append(pkg["radii"])

        # Compute loss (每个模型独立计算)
        gt_image = viewpoint_cam.original_image.cuda()
        losses = []

        for idx in range(gaussiansN):
            loss_dict = {"total": 0.0}
            render_loss = l1_loss(renders[idx]["render"], gt_image)
            loss_dict["render"] = render_loss
            loss_dict["total"] += render_loss

            if opt.lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(renders[idx]["render"], gt_image)
                loss_dict["dssim"] = loss_dssim
                loss_dict["total"] += opt.lambda_dssim * loss_dssim

            losses.append(loss_dict)
        # 3D TV loss (仅应用于第一个模型以节省计算)
        if use_tv:
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            current_gs = gaussians_list[0] if gaussiansN > 1 else gaussians
            vol_pred = query(
                current_gs,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            losses[0]["tv"] = loss_tv
            losses[0]["total"] += opt.lambda_tv * loss_tv

        # ========== CoR-GS Stage 3: Pseudo-view Co-regularization (官方对齐版) ==========
        if (HAS_PSEUDO_COREG and
            gaussiansN >= 2 and
            hasattr(dataset, 'enable_corgs') and dataset.enable_corgs and
            hasattr(dataset, 'corgs_pseudo_start_iter') and
            iteration >= dataset.corgs_pseudo_start_iter):

            try:
                # 【修复 1】动态权重 - 线性 ramp-up（官方 CoR-GS 策略）
                # loss_scale 从 0 线性增加到 1，持续 500 iterations
                corgs_ramp_iters = getattr(dataset, 'corgs_ramp_iters', 500)
                loss_scale = min((iteration - dataset.corgs_pseudo_start_iter) / float(corgs_ramp_iters), 1.0)

                # 【修复 3】使用 Pseudo-view 池采样（官方 CoR-GS 策略）
                # 如果池存在则从池中采样，否则回退到实时生成
                if pseudo_view_pool is not None:
                    pseudo_camera = pseudo_view_pool.sample()
                else:
                    # 回退：实时生成 pseudo-view
                    noise_std = dataset.corgs_pseudo_noise_std if hasattr(dataset, 'corgs_pseudo_noise_std') else 0.02
                    pseudo_camera = generate_pseudo_view_medical(
                        scene.getTrainCameras(),
                        current_camera_idx=None,
                        noise_std=noise_std,
                        roi_info=None
                    )

                # 渲染两个模型的 pseudo-view（仅前 2 个模型）
                pseudo_renders = []
                for idx in range(min(2, gaussiansN)):
                    pseudo_pkg = render(pseudo_camera, gaussians_list[idx], pipe)
                    pseudo_renders.append(pseudo_pkg["render"])  # 提取渲染图像

                # 【修复 2】双向独立损失计算（官方 CoR-GS 策略）
                # 每个模型独立计算损失，目标图像使用 detach 阻止梯度回流
                # 模型 0: 约束于 detach(模型 1 的输出)
                # 模型 1: 约束于 detach(模型 0 的输出)
                pseudo_weight = dataset.corgs_pseudo_weight if hasattr(dataset, 'corgs_pseudo_weight') else 1.0

                # 模型 0 的 co-reg 损失: render_0 vs detach(render_1)
                loss_coreg_0 = compute_pseudo_coreg_loss_medical(
                    pseudo_renders[0],  # 当前模型的渲染（需要梯度）
                    pseudo_renders[1].clone().detach(),  # 目标模型的渲染（阻止梯度）
                    lambda_dssim=opt.lambda_dssim,
                    roi_weights=None
                )

                # 模型 1 的 co-reg 损失: render_1 vs detach(render_0)
                loss_coreg_1 = compute_pseudo_coreg_loss_medical(
                    pseudo_renders[1],  # 当前模型的渲染（需要梯度）
                    pseudo_renders[0].clone().detach(),  # 目标模型的渲染（阻止梯度）
                    lambda_dssim=opt.lambda_dssim,
                    roi_weights=None
                )

                # 【修复 3】应用动态权重并叠加到各模型的独立损失
                scaled_weight = pseudo_weight * loss_scale
                losses[0]["pseudo"] = loss_coreg_0['loss']
                losses[1]["pseudo"] = loss_coreg_1['loss']
                losses[0]["total"] += scaled_weight * loss_coreg_0['loss']
                losses[1]["total"] += scaled_weight * loss_coreg_1['loss']

                # TensorBoard 日志（每 10 iterations 记录）
                if iteration % 10 == 0 and tb_writer:
                    avg_pseudo_loss = (loss_coreg_0['loss'].item() + loss_coreg_1['loss'].item()) / 2
                    tb_writer.add_scalar('corgs/pseudo_total', avg_pseudo_loss, iteration)
                    tb_writer.add_scalar('corgs/pseudo_l1_m0', loss_coreg_0['l1'].item(), iteration)
                    tb_writer.add_scalar('corgs/pseudo_l1_m1', loss_coreg_1['l1'].item(), iteration)
                    tb_writer.add_scalar('corgs/loss_scale', loss_scale, iteration)

                # 定期打印日志（每 500 iterations）
                if iteration % 500 == 0:
                    print(f"[CoR-GS] Iter {iteration}: "
                          f"loss_m0={loss_coreg_0['loss'].item():.6f}, "
                          f"loss_m1={loss_coreg_1['loss'].item():.6f}, "
                          f"scale={loss_scale:.3f}")

            except Exception as e:
                # 异常处理：pseudo-view 生成失败不影响主训练
                if iteration % 1000 == 0:
                    print(f"⚠️ [CoR-GS] Pseudo-view failed at iter {iteration}: {e}")

        # 反向传播所有模型的损失
        for idx in range(gaussiansN):
            # 使用 retain_graph 避免计算图被清除（除了最后一个模型）
            losses[idx]["total"].backward(retain_graph=(idx < gaussiansN - 1))

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control (所有模型)
            for idx in range(gaussiansN):
                current_gs = gaussians_list[idx] if gaussiansN > 1 else gaussians
                current_visibility = visibility_filters[idx]
                current_radii = radiis[idx]
                current_viewspace = viewspace_points[idx]

                current_gs.max_radii2D[current_visibility] = torch.max(
                    current_gs.max_radii2D[current_visibility],
                    current_radii[current_visibility]
                )
                current_gs.add_densification_stats(current_viewspace, current_visibility)

                if iteration < opt.densify_until_iter:
                    if (iteration > opt.densify_from_iter and
                        iteration % opt.densification_interval == 0):
                        current_gs.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.density_min_threshold,
                            opt.max_screen_size,
                            max_scale,
                            opt.max_num_gaussians,
                            densify_scale_threshold,
                            bbox,
                        )

                if current_gs.get_density.shape[0] == 0:
                    raise ValueError(
                        f"Model {idx+1}: No Gaussian left. Change adaptive control hyperparameters!"
                    )

            # Optimization (所有模型)
            if iteration < opt.iterations:
                for idx in range(gaussiansN):
                    current_gs = gaussians_list[idx] if gaussiansN > 1 else gaussians
                    current_gs.optimizer.step()
                    current_gs.optimizer.zero_grad(set_to_none=True)

            # Save gaussians (多模型：仅保存第一个模型)
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints (多模型：保存所有模型)
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                if gaussiansN == 1:
                    torch.save(
                        (gaussians.capture(), iteration),
                        ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                    )
                else:
                    # 多模型：保存所有模型的状态
                    checkpoint_dict = {
                        'iteration': iteration,
                        'models': [gs.capture() for gs in gaussians_list]
                    }
                    torch.save(
                        checkpoint_dict,
                        ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                    )

            # Progress bar (显示第一个模型的信息)
            if iteration % 10 == 0:
                first_gs = gaussians_list[0] if gaussiansN > 1 else gaussians
                avg_loss = sum(l["total"].item() for l in losses) / gaussiansN
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.1e}",
                        "pts": f"{first_gs.get_density.shape[0]:2.1e}",
                        "models": gaussiansN,
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging (记录所有模型的平均损失)
            metrics = {}
            # 计算所有模型的平均损失
            for loss_key in losses[0].keys():
                if loss_key in losses[0]:
                    avg_loss_value = sum(l.get(loss_key, 0) for l in losses if isinstance(l.get(loss_key, 0), torch.Tensor)) / gaussiansN
                    if isinstance(avg_loss_value, torch.Tensor):
                        metrics[f"train/loss_{loss_key}"] = avg_loss_value.item()

            # 记录每个模型的独立损失（多模型场景）
            if gaussiansN > 1:
                for idx in range(gaussiansN):
                    for loss_key, loss_val in losses[idx].items():
                        if isinstance(loss_val, torch.Tensor):
                            metrics[f"train/model{idx+1}/loss_{loss_key}"] = loss_val.item()

            # 记录学习率（所有模型）
            first_gs = gaussians_list[0] if gaussiansN > 1 else gaussians
            for param_group in first_gs.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
            )


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)

        # 处理多模型场景
        if isinstance(scene.gaussians, list):
            # 多模型：记录第一个模型的点数
            total_points = scene.gaussians[0].get_xyz.shape[0]
            tb_writer.add_scalar("train/total_points", total_points, iteration)
            # 记录每个模型的点数
            for idx, gs in enumerate(scene.gaussians):
                tb_writer.add_scalar(f"train/model{idx+1}/points", gs.get_xyz.shape[0], iteration)
        else:
            # 单模型
            tb_writer.add_scalar(
                "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
            )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections（多模型场景使用第一个模型）
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                gaussians_for_eval = scene.gaussians[0] if isinstance(scene.gaussians, list) else scene.gaussians
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        gaussians_for_eval,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance（多模型场景使用第一个模型）
        gaussians_for_query = scene.gaussians[0] if isinstance(scene.gaussians, list) else scene.gaussians
        vol_pred = queryFunc(gaussians_for_query)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_two_slice(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics（多模型场景使用第一个模型）
        if tb_writer:
            gaussians_for_hist = scene.gaussians[0] if isinstance(scene.gaussians, list) else scene.gaussians
            tb_writer.add_histogram(
                "scene/density_histogram", gaussians_for_hist.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
