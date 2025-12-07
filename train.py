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
from r2_gaussian.utils.regulation import compute_plane_tv_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice
from r2_gaussian.innovations.fsgs import ProximityGuidedDensifier
import copy


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

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound, args=dataset)  # 传递 dataset (ModelParams) 以支持 K-Planes
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # 🆕 Set up Proximity-Guided Densification (GAR 邻近引导密化)
    # 支持 enable_gar (主开关) 或 enable_gar_proximity / enable_fsgs_proximity (兼容旧名)
    use_proximity = getattr(dataset, 'enable_gar', False) or getattr(dataset, 'enable_gar_proximity', False) or getattr(dataset, 'enable_fsgs_proximity', False)
    proximity_densifier = None
    proximity_start_iter = dataset.proximity_start_iter
    proximity_interval = dataset.proximity_interval
    proximity_until_iter = dataset.proximity_until_iter
    # 🆕 GAR 优化参数
    gar_adaptive_threshold = getattr(dataset, 'gar_adaptive_threshold', False)
    gar_adaptive_method = getattr(dataset, 'gar_adaptive_method', 'percentile')
    gar_adaptive_percentile = getattr(dataset, 'gar_adaptive_percentile', 90.0)
    gar_progressive_decay = getattr(dataset, 'gar_progressive_decay', False)
    gar_decay_start_ratio = getattr(dataset, 'gar_decay_start_ratio', 0.5)
    gar_final_strength = getattr(dataset, 'gar_final_strength', 0.3)
    gar_gradient_filter = getattr(dataset, 'gar_gradient_filter', False)
    gar_gradient_threshold = getattr(dataset, 'gar_gradient_threshold', 0.0002)

    if use_proximity:
        print("Use FSGS proximity-guided densification (GAR)")
        # 支持新参数名 gar_* 和旧参数名 proximity_*（优先使用新名）
        k_neighbors = getattr(dataset, 'gar_proximity_k', None) or getattr(dataset, 'proximity_k_neighbors', 5)
        proximity_threshold = getattr(dataset, 'gar_proximity_threshold', None) or getattr(dataset, 'proximity_threshold', 5.0)
        proximity_densifier = ProximityGuidedDensifier(
            k_neighbors=k_neighbors,
            proximity_threshold=proximity_threshold,
            chunk_size=5000,
            enable=True,
            # 🆕 GAR 优化参数
            adaptive_threshold=gar_adaptive_threshold,
            adaptive_method=gar_adaptive_method,
            adaptive_percentile=gar_adaptive_percentile,
            progressive_decay=gar_progressive_decay,
            decay_start_ratio=gar_decay_start_ratio,
            final_strength=gar_final_strength,
        )
        print(f"  - k_neighbors: {proximity_densifier.k_neighbors}")
        print(f"  - proximity_threshold: {proximity_densifier.proximity_threshold}")
        print(f"  - start_iter: {proximity_start_iter}, interval: {proximity_interval}, until: {proximity_until_iter}")
        # 🆕 打印优化参数
        if gar_adaptive_threshold:
            print(f"  - 🆕 adaptive_threshold: {gar_adaptive_method} (p={gar_adaptive_percentile})")
        if gar_progressive_decay:
            print(f"  - 🆕 progressive_decay: start={gar_decay_start_ratio}, final={gar_final_strength}")
        if gar_gradient_filter:
            print(f"  - 🆕 gradient_filter: threshold={gar_gradient_threshold}")

    # 🎯 K-Planes 诊断信息
    if gaussians.enable_kplanes and gaussians.kplanes_encoder is not None:
        print("=" * 70)
        print("✓ K-Planes Encoder 已启用")
        # 从实际的 encoder 获取参数，而非 dataset（确保显示实际使用的值）
        actual_resolution = gaussians.kplanes_encoder.grid_resolution
        actual_dim = gaussians.kplanes_encoder.feature_dim
        print(f"  - 平面分辨率: {actual_resolution}")
        print(f"  - 特征维度: {actual_dim}")
        print(f"  - 总特征维度: {actual_dim * 3} (3 个平面)")

        kplanes_params = sum(p.numel() for p in gaussians.kplanes_encoder.parameters())
        print(f"  - K-Planes 参数量: {kplanes_params:,}")

        # 检查 TV 正则化
        if opt.lambda_plane_tv > 0:
            print(f"✓ K-Planes TV 正则化已启用")
            print(f"  - lambda_plane_tv: {opt.lambda_plane_tv}")
            print(f"  - TV 权重: {opt.plane_tv_weight_proposal}")
            print(f"  - TV 损失类型: {opt.tv_loss_type}")
        else:
            print("⚠️ 警告：K-Planes 已启用但 TV 正则化未启用 (lambda_plane_tv = 0)")

        print("=" * 70)
    else:
        print("⚠️ K-Planes 未启用（使用标准 R²-Gaussian）")

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

        # 传递当前迭代次数给 ADM 调度
        gaussians.current_iteration = iteration

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        # K-Planes TV 正则化损失（X²-Gaussian）
        if opt.lambda_plane_tv > 0 and gaussians.enable_kplanes and gaussians.kplanes_encoder is not None:
            planes = gaussians.kplanes_encoder.get_plane_params()
            tv_loss_planes = compute_plane_tv_loss(
                planes=planes,
                weights=opt.plane_tv_weight_proposal,
                loss_type=opt.tv_loss_type,
            )
            loss["plane_tv"] = tv_loss_planes
            loss["total"] = loss["total"] + opt.lambda_plane_tv * tv_loss_planes

            # 🎯 在前几个迭代输出诊断信息
            if iteration <= 3:
                kplanes_feat = gaussians.get_kplanes_features()
                print(f"[Iter {iteration}] K-Planes 诊断:")
                print(f"  - K-Planes 特征形状: {kplanes_feat.shape}")
                print(f"  - 特征范围: [{kplanes_feat.min().item():.4f}, {kplanes_feat.max().item():.4f}]")
                print(f"  - TV loss (plane): {tv_loss_planes.item():.6f}")
                print(f"  - TV loss (weighted): {(opt.lambda_plane_tv * tv_loss_planes).item():.6f}")

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )

            # 🆕 FSGS Proximity-Guided Densification (GAR核心组件) - 优化版
            if use_proximity and proximity_densifier is not None:
                if (iteration >= proximity_start_iter and
                    iteration <= proximity_until_iter and
                    iteration % proximity_interval == 0):

                    # 1. 计算邻近分数
                    positions = gaussians.get_xyz
                    proximity_scores, neighbor_indices, _ = proximity_densifier.compute_proximity_scores(
                        positions, return_neighbors=True
                    )

                    # 🆕 2. 计算有效阈值（自适应 + 渐进衰减）
                    effective_threshold = None

                    # 2a. 自适应阈值
                    if gar_adaptive_threshold:
                        effective_threshold = proximity_densifier.compute_adaptive_threshold_value(
                            proximity_scores
                        )

                    # 2b. 渐进衰减
                    if gar_progressive_decay:
                        decay_mult = proximity_densifier.get_decay_multiplier(
                            iteration, proximity_start_iter, proximity_until_iter
                        )
                        if effective_threshold is not None:
                            effective_threshold = effective_threshold * decay_mult
                        else:
                            effective_threshold = proximity_densifier.proximity_threshold * decay_mult

                    # 3. 识别需要密化的候选点
                    densify_mask = proximity_densifier.identify_densify_candidates(
                        proximity_scores,
                        custom_threshold=effective_threshold,
                        hybrid_mode="proximity_only"
                    )

                    # 🆕 4. 梯度过滤（只保留高梯度点）
                    if gar_gradient_filter:
                        grads = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-7)
                        grads[grads.isnan()] = 0.0
                        gradient_mask = grads.squeeze() > gar_gradient_threshold
                        densify_mask = densify_mask & gradient_mask

                    num_candidates = densify_mask.sum().item()

                    if num_candidates > 0:
                        # 限制每次最多密化的点数，避免内存爆炸
                        if num_candidates > 5000:
                            # 随机选择一部分
                            candidate_indices = torch.where(densify_mask)[0]
                            perm = torch.randperm(len(candidate_indices), device=candidate_indices.device)[:5000]
                            selected_indices = candidate_indices[perm]
                            densify_mask.fill_(False)  # 原地操作，减少内存分配
                            densify_mask[selected_indices] = True
                            num_candidates = 5000

                        # 5. 获取需要密化的点及其邻居
                        source_positions = positions[densify_mask]
                        source_neighbor_indices = neighbor_indices[densify_mask]

                        # 6. 准备属性字典
                        all_attributes = {
                            'scales': gaussians._scaling,
                            'opacities': gaussians._density,
                            'rotations': gaussians._rotation,
                        }

                        # 7. 生成新的 Gaussians
                        new_gaussians = proximity_densifier.generate_new_gaussians(
                            source_positions,
                            source_neighbor_indices,
                            positions,
                            all_attributes,
                            max_new_per_source=1  # 每个源点只生成1个新点（最近邻中点）
                        )

                        num_new = new_gaussians['positions'].shape[0]

                        if num_new > 0:
                            # 8. 添加到 GaussianModel
                            new_max_radii = torch.zeros(num_new, device=positions.device)

                            gaussians.densification_postfix(
                                new_xyz=new_gaussians['positions'],
                                new_densities=new_gaussians['opacities'],
                                new_scaling=new_gaussians['scales'],
                                new_rotation=new_gaussians['rotations'],
                                new_max_radii2D=new_max_radii,
                            )

                            if iteration % 1000 == 0:
                                tqdm.write(f"[ITER {iteration}] Proximity densification: +{num_new} Gaussians "
                                          f"(candidates: {num_candidates}, total: {gaussians.get_xyz.shape[0]})")

            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Opacity Decay: 在密度控制开始后的每次迭代应用衰减，过滤低梯度的冗余Gaussians
            if iteration > opt.densify_from_iter and opt.enable_opacity_decay:
                gaussians.opacity_decay(factor=opt.opacity_decay_factor)

            # Optimization（包括最后一次迭代）
            if iteration <= opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                postfix = {
                    "loss": f"{loss['total'].item():.1e}",
                    "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                }
                # 添加 K-Planes TV loss（如果启用）
                if "plane_tv" in loss:
                    postfix["tv_kp"] = f"{loss['plane_tv'].item():.1e}"
                # 添加 3D TV loss（如果启用）
                if "tv" in loss:
                    postfix["tv_3d"] = f"{loss['tv'].item():.1e}"

                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
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
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
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

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
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

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
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
    # 注意：K-Planes 相关参数已在 ModelParams 和 OptimizationParams 中定义，会自动注册
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
