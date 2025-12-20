#
# NeRF-based training function for r2_gaussian framework
#

import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm
from typing import Optional
from random import randint

from r2_gaussian.dataset import Scene
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.loss_utils import l1_loss
from r2_gaussian.utils.unified_logger import get_logger

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


def _safe_multinomial(weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    """torch.multinomial 的安全封装：全 0 时退化到均匀采样；非零数量不足时自动 replacement=True。"""
    if num_samples <= 0:
        return torch.empty((0,), device=weights.device, dtype=torch.long)

    weights = weights.float()
    if torch.all(weights <= 0):
        return torch.randint(0, weights.numel(), (num_samples,), device=weights.device)

    nonzero = int((weights > 0).sum().item())
    replacement = nonzero < num_samples
    return torch.multinomial(weights, num_samples, replacement=replacement)


def _get_training_rays(
    viewpoint,
    scanner_cfg: dict,
    config,
    ray_cache: dict,
    weight_cache: dict,
):
    """获取训练 rays：优先使用 cache + 重要性采样，避免每次迭代构造整张像素网格。"""
    n_rays = int(getattr(config, "n_rays", 1024))
    sampling = getattr(config, "ray_sampling", "uniform")
    importance_ratio = float(getattr(config, "importance_ratio", 0.0))

    if sampling == "on_the_fly" or viewpoint.uid not in ray_cache:
        return generate_rays_from_camera(viewpoint, scanner_cfg, n_rays=n_rays)

    rays_all, indices_all = ray_cache[viewpoint.uid]
    n_total = rays_all.shape[0]

    if sampling == "uniform" or importance_ratio <= 0 or viewpoint.uid not in weight_cache:
        select = torch.randperm(n_total, device=rays_all.device)[:n_rays]
        return rays_all[select], indices_all[select]

    weights = weight_cache[viewpoint.uid]
    n_imp = max(1, int(round(n_rays * importance_ratio)))
    n_uni = n_rays - n_imp

    imp_idx = _safe_multinomial(weights, n_imp)
    if n_uni > 0:
        uni_idx = torch.randperm(n_total, device=rays_all.device)[:n_uni]
        select = torch.cat([imp_idx, uni_idx], dim=0)
        select = select[torch.randperm(select.shape[0], device=rays_all.device)]
    else:
        select = imp_idx

    return rays_all[select], indices_all[select]


@torch.no_grad()
def _render_full_image(
    viewpoint,
    scanner_cfg: dict,
    model,
    config,
    chunk_rays: int,
) -> torch.Tensor:
    """按 rays 分块渲染整张投影图，避免一次性 H*W 渲染触发 OOM。"""
    rays, _ = generate_rays_from_camera(viewpoint, scanner_cfg, n_rays=None)

    preds = []
    for start in range(0, rays.shape[0], chunk_rays):
        render_result = render_rays(
            rays[start:start + chunk_rays],
            model.net,
            model.net_fine,
            n_samples=config.n_samples,
            n_fine=config.n_fine,
            perturb=False,
            netchunk=config.netchunk,
            raw_noise_std=0.0,
        )
        preds.append(render_result["acc"])

    pred = torch.cat(preds, dim=0)  # [H*W, 1]
    H, W = viewpoint.image_height, viewpoint.image_width
    return pred.reshape(H, W, 1).permute(2, 0, 1)  # [1, H, W]


@torch.no_grad()
def _query_volume_by_slices(model, config, scanner_cfg: dict) -> np.ndarray:
    """逐 z-slice 查询 3D 体素，避免一次性构造/查询超大网格导致 OOM。"""
    device = next(model.parameters()).device
    nVoxel = scanner_cfg["nVoxel"]
    sVoxel = scanner_cfg["sVoxel"]
    offOrigin = scanner_cfg.get("offOrigin", [0.0, 0.0, 0.0])

    nx, ny, nz = int(nVoxel[0]), int(nVoxel[1]), int(nVoxel[2])
    x = torch.linspace(offOrigin[0] - sVoxel[0] / 2, offOrigin[0] + sVoxel[0] / 2, nx, device=device)
    y = torch.linspace(offOrigin[1] - sVoxel[1] / 2, offOrigin[1] + sVoxel[1] / 2, ny, device=device)
    z = torch.linspace(offOrigin[2] - sVoxel[2] / 2, offOrigin[2] + sVoxel[2] / 2, nz, device=device)

    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1)  # [nx, ny, 2]

    net = model.net_fine if model.net_fine is not None else model.net
    vol_pred = np.empty((nx, ny, nz), dtype=np.float32)

    for zi in range(nz):
        zz = z[zi].expand(nx, ny).unsqueeze(-1)
        coords = torch.cat([xy, zz], dim=-1).reshape(-1, 3)
        dens = run_network(coords, net, config.netchunk).reshape(nx, ny, -1)[..., 0]
        vol_pred[:, :, zi] = dens.detach().cpu().numpy()

    return vol_pred


class NeRFModel(nn.Module):
    """NeRF 模型包装器"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 创建编码器（注意：不要在多个子模块中重复注册同一个 encoder，否则优化器会出现重复参数）
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

        encoder = get_encoder(config.encoding, **encoder_kwargs)

        # 获取网络类型
        net_type = getattr(config, 'net_type', 'mlp')

        # 创建网络
        if net_type == 'lineformer':
            from .lineformer import Lineformer
            self.net = Lineformer(
                encoder=encoder,
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
                encoder=encoder,
                bound=config.bound,
                num_layers=config.num_layers,
                hidden_dim=config.hidden_dim,
                skips=list(config.skips),
                out_dim=config.out_dim,
                last_activation=config.last_activation,
            )

        # 精细网络 (Lineformer 通常不使用精细网络)
        self.net_fine = None
        # 说明：
        # - 当前 render_rays 已支持 n_fine>0 且 net_fine=None 时复用 coarse net 做 fine 采样，
        #   默认不开启独立的 fine network（避免重复参数/训练不稳定）。
        use_fine_network = bool(getattr(config, 'use_fine_network', False))
        if use_fine_network and config.n_fine > 0 and net_type != 'lineformer':
            fine_encoder = get_encoder(config.encoding, **encoder_kwargs)
            self.net_fine = DensityNetwork(
                encoder=fine_encoder,
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
    logger = get_logger()
    logger.config(f"Method: {method}")
    logger.config(f"Config: {config}")

    # 加载场景
    scene = Scene(dataset, shuffle=False)
    scanner_cfg = scene.scanner_cfg

    # 创建模型
    model = NeRFModel(config).cuda()

    # 优化器
    params = list(model.parameters())
    unique_params = []
    seen = set()
    for p in params:
        pid = id(p)
        if pid in seen:
            continue
        unique_params.append(p)
        seen.add(pid)
    if len(unique_params) != len(params):
        logger.config(f"Deduplicate optimizer params: {len(params)} -> {len(unique_params)}")

    optimizer = torch.optim.Adam(unique_params, lr=config.lrate, betas=(0.9, 0.999))
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
        # 兼容旧 checkpoint：encoder 可能被单独保存
        if "encoder" in ckpt:
            try:
                model.net.encoder.load_state_dict(ckpt["encoder"])
            except Exception as e:
                logger.warn(f"Failed to load legacy encoder state: {e}")
        optimizer.load_state_dict(ckpt["optimizer"])
        logger.config(f"Loaded checkpoint from {checkpoint}, iteration {first_iter}")

    # 准备训练数据
    train_cameras = scene.getTrainCameras()

    # 预缓存训练视角 rays：可显著减少每次迭代的 H*W 网格构造开销
    ray_cache = {}
    weight_cache = {}
    sampling = getattr(config, "ray_sampling", "uniform")
    importance_ratio = float(getattr(config, "importance_ratio", 0.0))
    importance_power = float(getattr(config, "importance_power", 1.0))
    importance_eps = float(getattr(config, "importance_eps", 1e-4))

    if sampling != "on_the_fly":
        for cam in train_cameras:
            rays_all, indices_all = generate_rays_from_camera(cam, scanner_cfg, n_rays=None)
            ray_cache[cam.uid] = (rays_all, indices_all)

            if sampling != "uniform" and importance_ratio > 0:
                weights = cam.original_image[0].detach().clamp(min=0).flatten()
                weights = (weights + importance_eps) ** importance_power
                weight_cache[cam.uid] = weights

        # 简单 sanity：打印第一张训练视角的 ray-AABB 命中率
        if len(train_cameras) > 0:
            cam0 = train_cameras[0]
            rays0 = ray_cache[cam0.uid][0]
            valid0 = (rays0[:, 7] > rays0[:, 6] + 1e-6).float().mean().item()
            near0 = rays0[:, 6]
            far0 = rays0[:, 7]
            logger.config(
                f"Ray-AABB hit ratio (cam0 uid={cam0.uid}): {valid0:.3f}, "
                f"near[min,max]=({near0.min().item():.3f},{near0.max().item():.3f}), "
                f"far[min,max]=({far0.min().item():.3f},{far0.max().item():.3f})"
            )

    logger.config(f"Total iterations: {opt.iterations}")
    logger.config(f"Training views: {len(train_cameras)}")

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

        # 生成射线（优先 cache + 重要性采样）
        rays, indices = _get_training_rays(viewpoint, scanner_cfg, config, ray_cache, weight_cache)

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
        loss_type = getattr(config, "loss_type", "l1").lower()
        if loss_type in ["l2", "mse"]:
            loss = F.mse_loss(pred, gt_pixels)
        else:
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
                config, scanner_cfg
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
            torch.save(save_dict, save_path)
            logger.info(f"Saved model to {save_path}", iteration=iteration)

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
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}", iteration=iteration)

        # 学习率调度 - 基于固定迭代次数，而非视角循环
        # 每 5000 次迭代衰减一次学习率，确保 3/6/9 视角训练有一致的学习率行为
        if iteration % 5000 == 0:
            lr_scheduler.step()

    logger.info("Training complete!")


def _nerf_eval(tb_writer, iteration, method, model, scene, config, scanner_cfg):
    """NeRF 评估"""
    model.eval()
    eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
    os.makedirs(eval_save_path, exist_ok=True)

    with torch.no_grad():
        psnr_3d, ssim_3d = None, None

        # 3D 体积评估：逐 slice 查询，降低峰值显存
        vol_gt = scene.vol_gt
        if vol_gt is not None:
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
            # 兼容 3DGS 的默认文件名（便于统一脚本抽取）
            with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
                yaml.dump(eval_dict_3d, f, default_flow_style=False, sort_keys=False)

            if tb_writer:
                tb_writer.add_scalar(f"{method}/psnr_3d", psnr_3d, iteration)
                tb_writer.add_scalar(f"{method}/ssim_3d", ssim_3d, iteration)

        # 2D 投影评估：按 rays 分块渲染，避免 OOM
        test_cameras = scene.getTestCameras()
        if test_cameras and len(test_cameras) > 0:
            eval_max_views = int(getattr(config, "eval_max_views", 50))
            chunk_rays = int(getattr(config, "eval_rays_chunk", 8192))

            images = []
            gt_images = []

            for viewpoint in test_cameras[:eval_max_views]:
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
                "eval_num_views": int(min(eval_max_views, len(test_cameras))),
            }
            with open(osp.join(eval_save_path, f"eval2d_{method}.yml"), "w") as f:
                yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)
            # 兼容 3DGS 的默认文件名（便于统一脚本抽取）
            with open(osp.join(eval_save_path, "eval2d_render_test.yml"), "w") as f:
                yaml.dump(eval_dict_2d, f, default_flow_style=False, sort_keys=False)

            if tb_writer:
                tb_writer.add_scalar(f"{method}/psnr_2d", psnr_2d, iteration)
                tb_writer.add_scalar(f"{method}/ssim_2d", ssim_2d, iteration)

            logger = get_logger()
            if psnr_3d is not None and ssim_3d is not None:
                logger.eval(f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}", iteration=iteration)
            else:
                logger.eval(f"psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}", iteration=iteration)
        else:
            logger = get_logger()
            if psnr_3d is not None and ssim_3d is not None:
                logger.eval(f"psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}", iteration=iteration)
            else:
                logger.eval("(no test views / no vol_gt)", iteration=iteration)

    model.train()
    torch.cuda.empty_cache()
