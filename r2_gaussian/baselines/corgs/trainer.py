#
# CoR-GS: Co-Regularization Gaussian Splatting
# Training function for CT sparse-view reconstruction
#

import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import yaml
from random import randint
from tqdm import tqdm
from typing import Dict, Optional

import open3d as o3d

from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice

from ..xgaussian.model import XGaussianModel
from ..xgaussian.renderer import render_xgaussian, query_xgaussian
from .config import CoRGSConfig
from .pseudo_view import generate_pseudo_cameras_ct


def loss_photometric(image, gt_image, lambda_dssim=0.2):
    """
    计算光度损失

    Args:
        image: 渲染图像 [C, H, W]
        gt_image: GT 图像 [C, H, W]
        lambda_dssim: SSIM 损失权重

    Returns:
        loss: 光度损失
    """
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
    return loss


def compute_inconsistent_mask(xyz_source: torch.Tensor, xyz_target: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    使用 Open3D 计算点不一致性掩码

    Args:
        xyz_source: 源点云 [N, 3]
        xyz_target: 目标点云 [M, 3]
        threshold: 距离阈值

    Returns:
        mask_inconsistent: 不一致点的布尔掩码 [N]
    """
    # 转换为 Open3D 点云
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(xyz_source.detach().cpu().numpy())
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(xyz_target.detach().cpu().numpy())

    # 评估配准（计算点对应关系）
    trans_matrix = np.identity(4)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_cloud, target_cloud, threshold, trans_matrix
    )
    correspondence = np.array(evaluation.correspondence_set)

    # 构建一致性掩码
    n_points = xyz_source.shape[0]
    mask_consistent = torch.zeros(n_points, dtype=torch.bool, device=xyz_source.device)
    if len(correspondence) > 0:
        mask_consistent[correspondence[:, 0]] = True

    return ~mask_consistent  # 返回不一致的点


def training_corgs(
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
    CoR-GS 训练函数

    实现双 Gaussian 场训练 + Co-Regularization + Co-Pruning

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

    # 加载配置
    config = CoRGSConfig()

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
    queryfunc = lambda x: query_xgaussian(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # ============ 创建双 Gaussian 场 ============
    gs_dict: Dict[str, XGaussianModel] = {}
    sh_degree = getattr(dataset, 'sh_degree', config.sh_degree)

    for i in range(config.gaussians_n):
        gs_dict[f"gs{i}"] = XGaussianModel(sh_degree=sh_degree, scale_bound=scale_bound)

        # 从 baseline 初始化文件加载点云（不使用 SPS）
        init_path = _get_baseline_init_path(dataset)
        if init_path and osp.exists(init_path):
            gs_dict[f"gs{i}"].create_from_r2_init(init_path, spatial_lr_scale=1.0)
        else:
            raise ValueError(
                f"CoR-GS requires initialization file. "
                f"Please run initialize_pcd.py first. Expected: {init_path}"
            )

        # 设置优化器
        gs_opt = _create_opt_params(opt, config)
        gs_dict[f"gs{i}"].training_setup(gs_opt)
        print(f"[CoR-GS] Created Gaussian field gs{i} with {gs_dict[f'gs{i}'].get_num_points()} points")

    # 保存第一个场到 scene（用于兼容性）
    scene.gaussians = gs_dict["gs0"]

    # ============ 生成伪视图相机池 ============
    train_cameras = scene.getTrainCameras()
    pseudo_cameras = generate_pseudo_cameras_ct(
        train_cameras, scanner_cfg, n_views=config.n_pseudo_views
    )
    pseudo_stack = pseudo_cameras.copy()

    # 计算 Co-Pruning 阈值（基于体素尺寸）
    coprune_threshold = config.coprune_threshold * volume_to_world

    # 场景范围
    cameras_extent = scene.scene_scale

    # ============ 训练循环 ============
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="CoR-GS Training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        # 更新学习率（所有场）
        for i in range(config.gaussians_n):
            gs_dict[f"gs{i}"].update_learning_rate(iteration)

        # 每 500 迭代升级球谐阶数
        if iteration % 500 == 0:
            for i in range(config.gaussians_n):
                gs_dict[f"gs{i}"].oneupSHdegree()

        # 随机选择训练视角
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()

        # ============ 渲染双场并计算光度损失 ============
        loss_dict = {}
        render_dict = {}

        for i in range(config.gaussians_n):
            render_pkg = render_xgaussian(viewpoint_cam, gs_dict[f"gs{i}"], pipe)
            render_dict[f"gs{i}"] = render_pkg
            loss_dict[f"gs{i}"] = loss_photometric(
                render_pkg["render"], gt_image, config.lambda_dssim
            )

        # ============ Co-Regularization（伪视图渲染一致性） ============
        if (config.coreg and
            iteration >= config.start_sample_pseudo and
            iteration <= config.end_sample_pseudo and
            iteration % config.sample_pseudo_interval == 0):

            # 补充伪视图池
            if not pseudo_stack:
                pseudo_stack = pseudo_cameras.copy()

            # 随机选择伪视图
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            # 渲染双场的伪视图
            pseudo_renders = {}
            for i in range(config.gaussians_n):
                pseudo_render_pkg = render_xgaussian(pseudo_cam, gs_dict[f"gs{i}"], pipe)
                pseudo_renders[f"gs{i}"] = pseudo_render_pkg["render"]

            # 添加 Co-Reg 损失：强制双场在伪视图上一致
            for i in range(config.gaussians_n):
                for j in range(config.gaussians_n):
                    if i != j:
                        # detach 阻止梯度传递到另一个场
                        loss_dict[f"gs{i}"] += loss_photometric(
                            pseudo_renders[f"gs{i}"],
                            pseudo_renders[f"gs{j}"].detach(),
                            config.lambda_dssim
                        ) / (config.gaussians_n - 1)

        # ============ 反向传播 ============
        for i in range(config.gaussians_n):
            loss_dict[f"gs{i}"].backward()

        iter_end.record()

        with torch.no_grad():
            # 主损失用于日志
            loss = loss_dict["gs0"]

            # 更新进度
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                total_points = sum(gs_dict[f"gs{i}"].get_num_points() for i in range(config.gaussians_n))
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.6f}",
                    "Points": f"{total_points}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # TensorBoard 日志
            if tb_writer:
                tb_writer.add_scalar("train/loss_total", loss.item(), iteration)
                for i in range(config.gaussians_n):
                    tb_writer.add_scalar(f"train/loss_gs{i}", loss_dict[f"gs{i}"].item(), iteration)
                    tb_writer.add_scalar(f"train/points_gs{i}", gs_dict[f"gs{i}"].get_num_points(), iteration)

            # ============ 评估（使用 gs0 作为主场） ============
            if iteration in testing_iterations:
                _corgs_eval(tb_writer, iteration, scene, gs_dict["gs0"], pipe, queryfunc)

            # ============ 保存模型 ============
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving CoR-GS model")
                save_path = osp.join(scene.model_path, f"corgs_iter_{iteration}.pth")
                torch.save({
                    "gs0": gs_dict["gs0"].capture(),
                    "gs1": gs_dict["gs1"].capture() if config.gaussians_n > 1 else None,
                    "iteration": iteration,
                }, save_path)

            # ============ Co-Pruning（点云位置不一致性剪枝） ============
            if (config.coprune and
                iteration > config.densify_from_iter and
                iteration % config.coprune_interval == 0 and
                iteration < config.densify_until_iter):

                for i in range(config.gaussians_n):
                    for j in range(config.gaussians_n):
                        if i != j:
                            mask_inconsistent = compute_inconsistent_mask(
                                gs_dict[f"gs{i}"].get_xyz,
                                gs_dict[f"gs{j}"].get_xyz,
                                threshold=coprune_threshold
                            )
                            # 只剪枝不一致的点
                            if mask_inconsistent.sum() > 0:
                                gs_dict[f"gs{i}"].prune_points(mask_inconsistent)
                                if iteration % 1000 == 0:
                                    print(f"  [Co-Prune] gs{i}: pruned {mask_inconsistent.sum().item()} "
                                          f"inconsistent points")

            # ============ 密集化 ============
            if iteration < config.densify_until_iter:
                for i in range(config.gaussians_n):
                    viewspace_point_tensor = render_dict[f"gs{i}"]["viewspace_points"]
                    visibility_filter = render_dict[f"gs{i}"]["visibility_filter"]
                    radii = render_dict[f"gs{i}"]["radii"]

                    gs_dict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(
                        gs_dict[f"gs{i}"].max_radii2D[visibility_filter],
                        radii[visibility_filter]
                    )
                    gs_dict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > config.densify_from_iter and iteration % config.densification_interval == 0:
                    size_threshold = 20 if iteration > config.opacity_reset_interval else None
                    for i in range(config.gaussians_n):
                        gs_dict[f"gs{i}"].densify_and_prune(
                            config.densify_grad_threshold,
                            config.min_opacity,
                            cameras_extent,
                            size_threshold
                        )

                if iteration % config.opacity_reset_interval == 0:
                    for i in range(config.gaussians_n):
                        gs_dict[f"gs{i}"].reset_opacity()

            # ============ 优化步骤 ============
            if iteration < opt.iterations:
                for i in range(config.gaussians_n):
                    gs_dict[f"gs{i}"].optimizer.step()
                    gs_dict[f"gs{i}"].optimizer.zero_grad(set_to_none=True)

            # 检查点
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                ckpt_path = osp.join(scene.model_path, f"chkpnt_corgs_{iteration}.pth")
                torch.save({
                    "gs0": gs_dict["gs0"].capture(),
                    "gs1": gs_dict["gs1"].capture() if config.gaussians_n > 1 else None,
                    "iteration": iteration,
                }, ckpt_path)

    print("\n[CoR-GS] Training complete!")


def _get_baseline_init_path(dataset) -> Optional[str]:
    """
    获取 baseline 初始化文件路径

    CoR-GS 使用 baseline 初始化（不使用 SPS）
    """
    # 如果用户指定了 ply_path，使用它
    if hasattr(dataset, 'ply_path') and dataset.ply_path:
        return dataset.ply_path

    # 否则推断 baseline init 路径
    # 格式：data/369/init_<organ>_50_<views>views.npy
    source_path = dataset.source_path
    if source_path and osp.exists(source_path):
        # 从数据路径推断 init 路径
        # 例如：data/369/foot_50_3views.pickle -> data/369/init_foot_50_3views.npy
        basename = osp.basename(source_path).replace('.pickle', '')
        init_path = osp.join(osp.dirname(source_path), f"init_{basename}.npy")
        if osp.exists(init_path):
            return init_path

    return None


def _create_opt_params(opt, config: CoRGSConfig):
    """
    创建优化参数对象（合并用户参数和默认配置）
    """
    class CoRGSOptParams:
        def __init__(self, opt, config):
            self.percent_dense = getattr(opt, 'percent_dense', config.percent_dense)
            self.position_lr_init = getattr(opt, 'position_lr_init', config.position_lr_init)
            self.position_lr_final = getattr(opt, 'position_lr_final', config.position_lr_final)
            self.position_lr_delay_mult = getattr(opt, 'position_lr_delay_mult', config.position_lr_delay_mult)
            self.position_lr_max_steps = getattr(opt, 'position_lr_max_steps', config.position_lr_max_steps)
            self.feature_lr = getattr(opt, 'feature_lr', config.feature_lr)
            self.opacity_lr = getattr(opt, 'opacity_lr', config.opacity_lr)
            self.scaling_lr = getattr(opt, 'scaling_lr', config.scaling_lr)
            self.rotation_lr = getattr(opt, 'rotation_lr', config.rotation_lr)

    return CoRGSOptParams(opt, config)


def _corgs_eval(tb_writer, iteration, scene, gaussians, pipe, queryfunc):
    """CoR-GS 评估函数"""
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
        with open(osp.join(eval_save_path, "eval2d_corgs.yml"), "w") as f:
            yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

        if tb_writer:
            tb_writer.add_scalar("corgs/psnr_2d", psnr_2d, iteration)
            tb_writer.add_scalar("corgs/ssim_2d", ssim_2d, iteration)

    # 3D 评估
    vol_pred = queryfunc(gaussians)["vol"]
    vol_gt = scene.vol_gt
    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, _ = metric_vol(vol_gt, vol_pred, "ssim")

    eval_dict_3d = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
    }
    with open(osp.join(eval_save_path, "eval3d_corgs.yml"), "w") as f:
        yaml.dump(eval_dict_3d, f, default_flow_style=False, sort_keys=False)

    if tb_writer:
        tb_writer.add_scalar("corgs/psnr_3d", psnr_3d, iteration)
        tb_writer.add_scalar("corgs/ssim_3d", ssim_3d, iteration)

    tqdm.write(
        f"[CoR-GS ITER {iteration}] "
        f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, "
        f"psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
    )

    torch.cuda.empty_cache()
