#
# NeRF-based training function for r2_gaussian framework
#

import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm import tqdm
from typing import Optional
from random import randint

from r2_gaussian.dataset import Scene
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.loss_utils import l1_loss

from .network import DensityNetwork, get_network
from .render import render_rays, run_network, generate_rays_from_camera
from .encoder import get_encoder
from ..naf.config import NAFConfig
from ..tensorf.config import TensoRFConfig
from ..saxnerf.config import SAXNeRFConfig


def get_method_config(method: str):
    """获取方法配置"""
    if method == "naf":
        return NAFConfig()
    elif method == "tensorf":
        return TensoRFConfig()
    elif method == "saxnerf":
        return SAXNeRFConfig()
    else:
        raise ValueError(f"Unknown NeRF method: {method}")


class NeRFModel(nn.Module):
    """NeRF 模型包装器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 创建编码器
        encoder_kwargs = {
            'num_levels': getattr(config, 'num_levels', 16),
            'level_dim': getattr(config, 'level_dim', 2),
            'base_resolution': getattr(config, 'base_resolution', 16),
            'log2_hashmap_size': getattr(config, 'log2_hashmap_size', 19),
            'bound': config.bound,
        }

        # TensoRF 特有参数
        if config.encoding == 'tensorf':
            encoder_kwargs['density_n_comp'] = getattr(config, 'density_n_comp', 8)
            encoder_kwargs['app_dim'] = getattr(config, 'app_dim', 32)

        self.encoder = get_encoder(config.encoding, **encoder_kwargs)

        # 获取网络类型
        net_type = getattr(config, 'net_type', 'mlp')

        # 创建网络
        if net_type == 'lineformer':
            from .lineformer import Lineformer
            self.net = Lineformer(
                encoder=self.encoder,
                bound=config.bound,
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                skips=list(config.skips),
                out_dim=config.out_dim,
                last_activation=config.last_activation,
                line_size=getattr(config, 'line_size', 2),
                dim_head=getattr(config, 'dim_head', 4),
                heads=getattr(config, 'heads', 8),
                num_blocks=getattr(config, 'num_blocks', 1),
            )
        else:
            self.net = DensityNetwork(
                encoder=self.encoder,
                bound=config.bound,
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                skips=list(config.skips),
                out_dim=config.out_dim,
                last_activation=config.last_activation,
            )

        # 精细网络 (Lineformer 通常不使用精细网络)
        self.net_fine = None
        if config.n_fine > 0 and net_type != 'lineformer':
            self.net_fine = DensityNetwork(
                encoder=self.encoder,
                bound=config.bound,
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                skips=list(config.skips),
                out_dim=config.out_dim,
                last_activation=config.last_activation,
            )

    @property
    def bound(self):
        return self.net.bound


def training_nerf(
    method: str,
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
    NeRF 系列训练函数

    Args:
        method: 方法名 ('naf', 'tensorf', 'saxnerf')
        dataset: ModelParams
        opt: OptimizationParams
        pipe: PipelineParams
        tb_writer: TensorBoard writer
        testing_iterations: 测试迭代列表
        saving_iterations: 保存迭代列表
        checkpoint_iterations: 检查点迭代列表
        checkpoint: 起始检查点路径
    """
    # 获取配置
    config = get_method_config(method)
    print(f"\n[NeRF Training] Method: {method}")
    print(f"[NeRF Training] Config: {config}")

    # 加载场景
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg

    # 创建模型
    model = NeRFModel(config).cuda()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lrate, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lrate_step, gamma=config.lrate_gamma
    )

    # 加载检查点
    first_iter = 0
    if checkpoint is not None and osp.exists(checkpoint):
        ckpt = torch.load(checkpoint)
        first_iter = ckpt.get("iteration", 0) + 1
        model.net.load_state_dict(ckpt["network"])
        if model.net_fine is not None and ckpt.get("network_fine"):
            model.net_fine.load_state_dict(ckpt["network_fine"])
        if "encoder" in ckpt and model.encoder is not None:
            model.encoder.load_state_dict(ckpt["encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded checkpoint from {checkpoint}, iteration {first_iter}")

    # 准备训练数据
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()

    # 体素网格用于 3D 评估
    nVoxel = scanner_cfg["nVoxel"]
    sVoxel = scanner_cfg["sVoxel"]
    offOrigin = scanner_cfg["offOrigin"]
    scene_bound = max(sVoxel) / 2  # 场景边界用于归一化

    # 创建体素坐标 (归一化到 [-1, 1])
    # 原始范围: [-sVoxel/2, sVoxel/2]
    # 归一化: [-1, 1]
    x = torch.linspace(-sVoxel[0]/2, sVoxel[0]/2, int(nVoxel[0])) / scene_bound
    y = torch.linspace(-sVoxel[1]/2, sVoxel[1]/2, int(nVoxel[1])) / scene_bound
    z = torch.linspace(-sVoxel[2]/2, sVoxel[2]/2, int(nVoxel[2])) / scene_bound
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    voxels = torch.stack([xx, yy, zz], dim=-1).cuda()

    print(f"[NeRF Training] Voxel coords normalized to [{voxels.min():.2f}, {voxels.max():.2f}]")

    print(f"[NeRF Training] Total iterations: {opt.iterations}")
    print(f"[NeRF Training] Training views: {len(train_cameras)}")

    # 训练循环
    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc=f"{method} Training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        model.train()

        # 随机选择视角
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint = viewpoint_stack.pop(viewpoint_idx)

        # 生成射线
        rays, indices = generate_rays_from_camera(
            viewpoint, scanner_cfg, n_rays=config.n_rays
        )

        # 渲染
        render_result = render_rays(
            rays,
            model.net,
            model.net_fine,
            n_samples=config.n_samples,
            n_fine=config.n_fine,
            perturb=config.perturb,
            netchunk=config.netchunk,
            raw_noise_std=config.raw_noise_std,
        )

        # 预测值
        pred = render_result["acc"]  # [N_rays, 1]

        # GT 图像
        gt_image = viewpoint.original_image.to("cuda")  # [1, H, W]
        H, W = gt_image.shape[1], gt_image.shape[2]

        # 获取对应像素的 GT 值
        gt_pixels = gt_image[0, indices[:, 0], indices[:, 1]].unsqueeze(-1)  # [N_rays, 1]

        # 损失
        loss = l1_loss(pred, gt_pixels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss": f"{ema_loss_for_log:.6f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

        # TensorBoard
        if tb_writer and iteration % 100 == 0:
            tb_writer.add_scalar(f"{method}/loss", loss.item(), iteration)
            tb_writer.add_scalar(f"{method}/lr", optimizer.param_groups[0]['lr'], iteration)

        # 评估
        if iteration in testing_iterations:
            _nerf_eval(
                tb_writer, iteration, method, model, scene,
                voxels, config, scanner_cfg
            )

        # 保存
        if iteration in saving_iterations:
            save_path = osp.join(
                scene.model_path, f"{method}_iter_{iteration}.pth"
            )
            save_dict = {
                "iteration": iteration,
                "network": model.net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if model.net_fine is not None:
                save_dict["network_fine"] = model.net_fine.state_dict()
            if model.encoder is not None:
                save_dict["encoder"] = model.encoder.state_dict()
            torch.save(save_dict, save_path)
            print(f"\n[{method}] Saved model to {save_path}")

        # 检查点
        if iteration in checkpoint_iterations:
            ckpt_path = osp.join(scene.model_path, f"chkpnt_{method}_{iteration}.pth")
            save_dict = {
                "iteration": iteration,
                "network": model.net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if model.net_fine is not None:
                save_dict["network_fine"] = model.net_fine.state_dict()
            if model.encoder is not None:
                save_dict["encoder"] = model.encoder.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"\n[{method}] Saved checkpoint to {ckpt_path}")

        # 学习率调度 - 基于固定迭代次数，而非视角循环
        # 每 5000 次迭代衰减一次学习率，确保 3/6/9 视角训练有一致的学习率行为
        if iteration % 5000 == 0:
            lr_scheduler.step()

    print(f"\n[{method}] Training complete!")


def _nerf_eval(tb_writer, iteration, method, model, scene, voxels, config, scanner_cfg):
    """NeRF 评估"""
    model.eval()
    eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
    os.makedirs(eval_save_path, exist_ok=True)

    with torch.no_grad():
        # 3D 体积查询
        vol_pred = run_network(
            voxels.reshape(-1, 3),
            model.net_fine if model.net_fine else model.net,
            config.netchunk,
        )
        vol_pred = vol_pred.reshape(*voxels.shape[:-1])

        # 与 GT 比较
        vol_gt = scene.vol_gt
        if vol_gt is not None:
            vol_pred_np = vol_pred.squeeze().cpu().numpy()
            # 确保 vol_gt 也是 numpy array
            if isinstance(vol_gt, torch.Tensor):
                vol_gt_np = vol_gt.cpu().numpy()
            else:
                vol_gt_np = vol_gt
            psnr_3d, _ = metric_vol(vol_gt_np, vol_pred_np, "psnr")
            ssim_3d, _ = metric_vol(vol_gt_np, vol_pred_np, "ssim")

            eval_dict = {"psnr_3d": psnr_3d, "ssim_3d": ssim_3d}
            with open(osp.join(eval_save_path, f"eval3d_{method}.yml"), "w") as f:
                yaml.dump(eval_dict, f)

            if tb_writer:
                tb_writer.add_scalar(f"{method}/psnr_3d", psnr_3d, iteration)
                tb_writer.add_scalar(f"{method}/ssim_3d", ssim_3d, iteration)

        # 2D 投影评估
        test_cameras = scene.getTestCameras()
        if test_cameras and len(test_cameras) > 0:
            images = []
            gt_images = []

            for viewpoint in test_cameras[:10]:  # 只评估前 10 个视角
                rays, indices = generate_rays_from_camera(
                    viewpoint, scanner_cfg, n_rays=None
                )
                render_result = render_rays(
                    rays,
                    model.net,
                    model.net_fine,
                    n_samples=config.n_samples,
                    n_fine=config.n_fine,
                    perturb=False,
                    netchunk=config.netchunk,
                    raw_noise_std=0.0,
                )

                pred = render_result["acc"]  # [H*W, 1]
                H, W = viewpoint.image_height, viewpoint.image_width
                pred_image = pred.reshape(H, W, 1).permute(2, 0, 1)  # [1, H, W]

                gt_image = viewpoint.original_image.to("cuda")
                images.append(pred_image)
                gt_images.append(gt_image)

            images = torch.concat(images, 0).permute(1, 2, 0)
            gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)

            psnr_2d, _ = metric_proj(gt_images, images, "psnr")
            ssim_2d, _ = metric_proj(gt_images, images, "ssim")

            eval_dict_2d = {"psnr_2d": psnr_2d, "ssim_2d": ssim_2d}
            with open(osp.join(eval_save_path, f"eval2d_{method}.yml"), "w") as f:
                yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

            if tb_writer:
                tb_writer.add_scalar(f"{method}/psnr_2d", psnr_2d, iteration)
                tb_writer.add_scalar(f"{method}/ssim_2d", ssim_2d, iteration)

            tqdm.write(
                f"[{method} ITER {iteration}] "
                f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, "
                f"psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
            )
        else:
            tqdm.write(f"[{method} ITER {iteration}] psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}")

    model.train()
    torch.cuda.empty_cache()
