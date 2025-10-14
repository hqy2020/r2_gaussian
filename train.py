###############################################################
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# 本软件仅限非商业、科研和评估用途，具体条款见 LICENSE.md 文件。
# 如有疑问请联系 george.drettakis@inria.fr
###############################################################

import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

# 添加项目路径，导入自定义模块

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams  # 参数定义
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian  # 高斯模型相关
from r2_gaussian.utils.general_utils import safe_state  # 随机种子等系统状态
from r2_gaussian.utils.cfg_utils import load_config  # 配置文件加载
from r2_gaussian.utils.log_utils import prepare_output_and_logger  # 日志与输出
from r2_gaussian.dataset import Scene  # 数据集场景
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss, loss_photometric, l1_loss_mask, depth_loss, pseudo_label_loss  # 损失函数
from r2_gaussian.utils.image_utils import metric_vol, metric_proj  # 评估指标
from r2_gaussian.utils.plot_utils import show_two_slice  # 可视化工具


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    gaussiansN=2,
    coreg=True,
    coprune=True,
    coprune_threshold=5,
    args=None,
):
    """
    训练主循环，负责高斯模型的初始化、损失计算、反向传播、稠密化与剪枝、保存模型和断点，以及日志记录。
    """
    first_iter = 0

    # 初始化数据集场景
    scene = Scene(dataset, shuffle=False)

    # 读取扫描仪配置和体素参数
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
    # 查询函数，用于体素采样
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # 初始化高斯模型
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    
    # 创建高斯场字典 - 参考X-Gaussian-depth实现
    GsDict = {}
    for i in range(gaussiansN):
        if i == 0:
            GsDict[f"gs{i}"] = gaussians
        else:
            GsDict[f"gs{i}"] = GaussianModel(scale_bound)
            initialize_gaussian(GsDict[f"gs{i}"], dataset, None)
            GsDict[f"gs{i}"].training_setup(opt)
            print(f"Create gaussians{i}")
    print(f"GsDict.keys() is {GsDict.keys()}")
    
    # 初始化多高斯和伪标签功能 - 参考X-Gaussian实现
    pseudo_cameras = None
    pseudo_labels = None
    if dataset.multi_gaussian or dataset.pseudo_labels:
        print("Generating pseudo cameras for multi-view training...")
        pseudo_cameras = scene.generate_multi_gaussian_cameras(
            num_additional_views=dataset.num_additional_views
        )
        print(f"Generated {len(pseudo_cameras)} pseudo cameras")
    # 加载断点（如有）
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # 设置损失函数（是否使用 TV 损失）
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # 训练主循环
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

        # 更新学习率 - 为每个高斯场更新
        for i in range(gaussiansN):
            GsDict[f"gs{i}"].update_learning_rate(iteration)

        # 随机选择一个训练视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 为每个高斯场渲染 - 参考X-Gaussian-depth实现
        RenderDict = {}
        for i in range(gaussiansN):
            RenderDict[f"render_pkg_gs{i}"] = render(
                viewpoint_cam,
                GsDict[f'gs{i}'],
                pipe,
                enable_drop=args.enable_drop,
                drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                iteration=iteration,
            )
            RenderDict[f"image_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["render"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = RenderDict[f"render_pkg_gs{i}"]["radii"]

        # 计算每个高斯场的损失
        LossDict = {}
        gt_image = viewpoint_cam.original_image.cuda()
        
        for i in range(gaussiansN):
            LossDict[f"loss_gs{i}"] = l1_loss(RenderDict[f"image_gs{i}"], gt_image)
            
            # DSSIM 损失
            if opt.lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(RenderDict[f"image_gs{i}"], gt_image)
                LossDict[f"loss_gs{i}"] += opt.lambda_dssim * loss_dssim
        
        # 协同训练 - 参考X-Gaussian-depth实现
        if coreg and gaussiansN > 1:
            for i in range(gaussiansN):
                for j in range(gaussiansN):
                    if i != j:
                        coreg_loss = l1_loss(RenderDict[f"image_gs{i}"], RenderDict[f"image_gs{j}"].detach())
                        LossDict[f"loss_gs{i}"] += coreg_loss
        
        # 多高斯训练损失 - 参考X-Gaussian实现
        if dataset.multi_gaussian and pseudo_cameras is not None:
            for pseudo_cam in pseudo_cameras[:3]:  # 限制数量避免计算过载
                for i in range(gaussiansN):
                    pseudo_render_pkg = render(
                        pseudo_cam,
                        GsDict[f'gs{i}'],
                        pipe,
                        enable_drop=args.enable_drop,
                        drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                        iteration=iteration,
                    )
                    pseudo_image = pseudo_render_pkg["render"]
                    # 使用伪标签相机生成的目标（这里简化为当前渲染结果）
                    pseudo_target = pseudo_image.detach()  # 使用当前渲染作为目标
                    multi_view_loss = l1_loss(pseudo_image, pseudo_target)
                    LossDict[f"loss_gs{i}"] += dataset.multi_gaussian_weight * multi_view_loss
        
        # 伪标签训练损失 - 参考X-Gaussian实现
        if dataset.pseudo_labels and pseudo_cameras is not None and iteration > 1000:  # 延迟启动伪标签
            for i, pseudo_cam in enumerate(pseudo_cameras[:2]):  # 限制数量
                for j in range(gaussiansN):
                    pseudo_render_pkg = render(
                        pseudo_cam,
                        GsDict[f'gs{j}'],
                        pipe,
                        enable_drop=args.enable_drop,
                        drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                        iteration=iteration,
                    )
                    pseudo_image = pseudo_render_pkg["render"]
                    
                    # 生成伪标签（使用当前模型预测）
                    with torch.no_grad():
                        pseudo_label = pseudo_cam.generate_pseudo_label(GsDict[f'gs{j}'], lambda cam, gauss: render(cam, gauss, pipe))
                    
                    pseudo_label_loss_val = pseudo_label_loss(pseudo_image, pseudo_label)
                    LossDict[f"loss_gs{j}"] += dataset.pseudo_label_weight * pseudo_label_loss_val
        
        # 深度约束损失 - r2-gaussian不支持深度输出，已禁用此功能
        # if dataset.depth_constraint and hasattr(viewpoint_cam, 'depth_image') and viewpoint_cam.depth_image is not None:
        #     for i in range(gaussiansN):
        #         # 计算深度图（这里需要扩展渲染器支持深度输出）
        #         # 暂时跳过深度损失的具体实现，因为需要修改渲染器
        #         pass
        
        # 3D TV 损失 - 为每个高斯场计算
        if use_tv:
            for i in range(gaussiansN):
                # 随机选取一个小体积中心
                tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                    bbox[1] - tv_vol_sVoxel - bbox[0]
                ) * torch.rand(3)
                vol_pred = query(
                    GsDict[f"gs{i}"],
                    tv_vol_center,
                    tv_vol_nVoxel,
                    tv_vol_sVoxel,
                    pipe,
                )["vol"]
                loss_tv = tv_3d_loss(vol_pred, reduction="mean")
                LossDict[f"loss_gs{i}"] += opt.lambda_tv * loss_tv

        # 反向传播 - 为每个高斯场
        for i in range(gaussiansN):
            LossDict[f"loss_gs{i}"].backward(retain_graph=(i < gaussiansN - 1))

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # 自适应控制：更新高斯半径和统计 - 为每个高斯场
            for i in range(gaussiansN):
                viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]
                visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                radii = RenderDict[f"radii_gs{i}"]
                
                GsDict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(
                    GsDict[f"gs{i}"].max_radii2D[visibility_filter], radii[visibility_filter]
                )
                GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            # 高斯点稠密化与剪枝 - 为每个高斯场
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    for i in range(gaussiansN):
                        GsDict[f"gs{i}"].densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.density_min_threshold,
                            opt.max_screen_size,
                            max_scale,
                            opt.max_num_gaussians,
                            densify_scale_threshold,
                            bbox,
                        )
            
            # 检查高斯场是否为空
            for i in range(gaussiansN):
                if GsDict[f"gs{i}"].get_density.shape[0] == 0:
                    raise ValueError(
                        f"No Gaussian left in gs{i}. Change adaptive control hyperparameters!"
                    )

            # 优化器更新 - 为每个高斯场
            if iteration < opt.iterations:
                for i in range(gaussiansN):
                    GsDict[f"gs{i}"].optimizer.step()
                    GsDict[f"gs{i}"].optimizer.zero_grad(set_to_none=True)

            # 保存高斯模型
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)
                
                # 保存额外的高斯场
                if gaussiansN > 1:
                    for i in range(1, gaussiansN):
                        pcd_path = osp.join(scene.model_path, f"point_cloud_gs{i}/iteration_{iteration}")
                        os.makedirs(pcd_path, exist_ok=True)
                        GsDict[f"gs{i}"].save_ply(osp.join(pcd_path, "point_cloud.ply"))

            # 保存断点
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                checkpoint_data = {}
                for i in range(gaussiansN):
                    checkpoint_data[f"gs{i}"] = GsDict[f"gs{i}"].capture()
                checkpoint_data["iteration"] = iteration
                torch.save(checkpoint_data, ckpt_save_path + "/chkpnt" + str(iteration) + ".pth")

            # 进度条显示
            if iteration % 10 == 0:
                # 计算总损失和总点数
                total_loss = sum(LossDict[f"loss_gs{i}"].item() for i in range(gaussiansN))
                total_points = sum(GsDict[f"gs{i}"].get_density.shape[0] for i in range(gaussiansN))
                
                progress_bar.set_postfix(
                    {
                        "loss": f"{total_loss:.1e}",
                        "pts": f"{total_points:2.1e}",
                        "gs": f"{gaussiansN}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 日志记录与评估
            metrics = {}
            for i in range(gaussiansN):
                metrics[f"loss_gs{i}"] = LossDict[f"loss_gs{i}"].item()
                for param_group in GsDict[f"gs{i}"].optimizer.param_groups:
                    metrics[f"lr_gs{i}_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y, it=iteration: render(
                    x,
                    y,
                    pipe,
                    enable_drop=args.enable_drop,
                    drop_rate=args.drop_rate if hasattr(args, 'drop_rate') else 0.10,
                    iteration=it,
                ),
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
    """
    训练过程中的评估与日志记录，包括训练统计、2D渲染性能、3D重建性能等。
    """
    # 记录训练统计信息
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # 测试与评估
    if iteration in testing_iterations:
        # 2D渲染性能评估
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
                # 渲染所有视角
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

        # 3D重建性能评估
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
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.4f}, ssim3d {ssim_3d:.4f}, psnr2d {psnr_2d:.4f}, ssim2d {ssim_2d:.4f}"
        )

        # 记录其他指标
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 命令行入口，参数解析与训练启动
    # fmt: off
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters") 
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)  # 是否开启异常检测
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])  # 测试迭代
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])  # 保存迭代
    parser.add_argument("--quiet", action="store_true")  # 静默模式
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 断点保存迭代
    parser.add_argument("--start_checkpoint", type=str, default=None)  # 起始断点
    parser.add_argument("--config", type=str, default=None)  # 配置文件路径
    parser.add_argument("--enable_drop", action="store_true", default=False)  # 是否启用 drop 方法
    parser.add_argument("--drop_rate", type=float, default=0.10)  # drop 比例（0~1）
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # 初始化系统状态（如随机种子）
    safe_state(args.quiet)

    # 加载配置文件（如有）
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # 设置日志与输出
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    # 是否开启异常检测
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # 启动训练主循环
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.gaussiansN,
        args.coreg,
        args.coprune,
        args.coprune_threshold,
        args,
    )
    
    # 注意：在训练过程中使用 render 函数时，需要传递 enable_drop=args.enable_drop 参数

    # 训练结束
    print("Training complete.")
