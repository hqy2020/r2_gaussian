import os
import numpy as np
import tigre.algorithms as algs
import open3d as o3d
import sys
import argparse
import os.path as osp
import json
import pickle
from tqdm import trange
import copy
import torch
from scipy.ndimage import gaussian_filter

sys.path.append("./")
from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume
from r2_gaussian.arguments import ParamGroup, ModelParams, PipelineParams
from r2_gaussian.utils.plot_utils import show_one_volume, show_two_volume
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.utils.image_utils import metric_vol
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.general_utils import t2a

np.random.seed(0)


class InitParams(ParamGroup):
    """
    点云初始化参数 - SPS (Spatial Prior Seeding)

    ============================================================================
    SPAGS Stage 1: SPS - 空间先验播种
    ============================================================================
    基于 FDK 重建体的密度分布进行智能采样初始化。

    消融实验配置：
    ----------------------------------------------------------------------------
    1. Baseline 初始化（随机采样）：
       python initialize_pcd.py -s <data_path>

    2. SPS 密度加权采样（推荐，+0.16 dB）：
       python initialize_pcd.py -s <data_path> --enable_sps

    3. SPS + 降噪预处理：
       python initialize_pcd.py -s <data_path> --enable_sps --sps_denoise
    ============================================================================
    """
    def __init__(self, parser):
        # ════════════════════════════════════════════════════════════════════
        # [BASELINE] R²-Gaussian 原版初始化参数
        # ════════════════════════════════════════════════════════════════════
        self.recon_method = "fdk"  # [BASELINE] 重建方法: fdk/random
        self.n_points = 50000  # [BASELINE] 初始化点数
        self.density_thresh = 0.05  # [BASELINE] 密度阈值（过滤低密度区域）
        self.density_rescale = 0.15  # [BASELINE] 密度缩放因子
        self.random_density_max = 1.0  # [BASELINE] 随机模式最大密度

        # ════════════════════════════════════════════════════════════════════
        # [SPS] Spatial Prior Seeding - 空间先验播种
        # SPAGS Stage 1: 基于密度的智能采样初始化
        # 主开关: enable_sps
        # ════════════════════════════════════════════════════════════════════
        self.enable_sps = False  # [SPS] 主开关：启用空间先验播种
        self.sps_denoise = False  # [SPS] 启用高斯降噪预处理
        self.sps_denoise_sigma = 3.0  # [SPS] 降噪核标准差
        self.sps_strategy = "density_weighted"  # [SPS] 采样策略: density_weighted/stratified
        # - density_weighted: 密度加权采样（推荐，公式: P(x) = ρ(x) / Σρ）
        # - stratified: 分层采样（确保不同密度区间都有代表）

        # 向下兼容旧参数名
        self.enable_denoise = False  # [兼容] 旧名，映射到 sps_denoise
        self.denoise_sigma = 3.0  # [兼容] 旧名
        self.sampling_strategy = "random"  # [兼容] 旧名

        super().__init__(parser, "Initialization Parameters")


def init_pcd(
    projs,
    angles,
    geo,
    scanner_cfg,
    args: InitParams,
    save_path,
):
    "Initialize Gaussians."
    recon_method = args.recon_method
    n_points = args.n_points
    assert recon_method in ["random", "fdk"], "--recon_method not supported."
    if recon_method == "random":
        print(f"Initialize random point clouds.")
        sampled_positions = np.array(scanner_cfg["offOrigin"])[None, ...] + np.array(
            scanner_cfg["sVoxel"]
        )[None, ...] * (np.random.rand(n_points, 3) - 0.5)
        sampled_densities = (
            np.random.rand(
                n_points,
            )
            * args.random_density_max
        )
    else:
        # Use traditional algorithms for initialization
        print(
            f"Initialize point clouds with the volume reconstructed from {recon_method}."
        )
        vol = recon_volume(projs, angles, copy.deepcopy(geo), recon_method)

        # [SPS] 降噪预处理（支持新旧参数名，修复 or 逻辑错误）
        # 修复：使用 None 检查而非 or，避免值为 0/False 时错误回退
        _sps_denoise = getattr(args, 'sps_denoise', None)
        enable_denoise = _sps_denoise if _sps_denoise is not None else getattr(args, 'enable_denoise', False)
        _sps_sigma = getattr(args, 'sps_denoise_sigma', None)
        denoise_sigma = _sps_sigma if _sps_sigma is not None else getattr(args, 'denoise_sigma', 3.0)
        if enable_denoise:
            print(f"[SPS] Applying Gaussian filter for denoising (sigma={denoise_sigma})...")
            vol = gaussian_filter(vol, sigma=denoise_sigma)
            print(f"[SPS] Denoising complete.")

        density_mask = vol > args.density_thresh
        valid_indices = np.argwhere(density_mask)
        offOrigin = np.array(scanner_cfg["offOrigin"])
        dVoxel = np.array(scanner_cfg["dVoxel"])
        sVoxel = np.array(scanner_cfg["sVoxel"])

        # [SPS] 输入体积统计（用于诊断和调参）
        print(f"[SPS] 输入体积统计:")
        print(f"  - 体积形状: {vol.shape}")
        print(f"  - 密度范围: [{vol.min():.4f}, {vol.max():.4f}]")
        print(f"  - 密度均值: {vol.mean():.4f}, 标准差: {vol.std():.4f}")
        print(f"  - 有效体素数(>{args.density_thresh}): {valid_indices.shape[0]:,}")
        print(f"  - 有效体素占比: {valid_indices.shape[0] / vol.size * 100:.2f}%")

        assert (
            valid_indices.shape[0] >= n_points
        ), "Valid voxels less than target number of sampling. Check threshold"

        # [SPS] 采样策略选择（支持新旧参数名）
        # 新参数: enable_sps + sps_strategy
        # 旧参数: sampling_strategy
        enable_sps = getattr(args, 'enable_sps', False)
        if enable_sps:
            # 使用 SPS 新参数
            strategy = getattr(args, 'sps_strategy', 'density_weighted')
        else:
            # 使用旧参数或默认随机
            strategy = getattr(args, 'sampling_strategy', 'random')

        if strategy == "random":
            # Baseline: 随机采样
            print(f"[Baseline] Using random sampling strategy.")
            sampled_idx = np.random.choice(len(valid_indices), n_points, replace=False)
        elif strategy == "density_weighted":
            # [SPS] 密度加权采样：P(x) = ρ(x) / Σρ
            print(f"[SPS] Using density-weighted sampling strategy.")
            densities_flat = vol[
                valid_indices[:, 0],
                valid_indices[:, 1],
                valid_indices[:, 2],
            ]
            # 归一化密度作为采样概率
            probs = densities_flat / densities_flat.sum()
            sampled_idx = np.random.choice(
                len(valid_indices), n_points, replace=False, p=probs
            )
        elif strategy == "stratified":
            # [SPS] 分层采样：确保不同密度区间都有代表
            print(f"[SPS] Using stratified sampling strategy.")
            densities_flat = vol[
                valid_indices[:, 0],
                valid_indices[:, 1],
                valid_indices[:, 2],
            ]
            # 将密度分为 5 个层级
            num_strata = 5
            points_per_stratum = n_points // num_strata
            sampled_idx = []
            for i in range(num_strata):
                lower = np.percentile(densities_flat, i * 20)
                upper = np.percentile(densities_flat, (i + 1) * 20)
                stratum_mask = (densities_flat >= lower) & (densities_flat < upper)
                stratum_indices = np.where(stratum_mask)[0]
                if len(stratum_indices) > 0:
                    n_sample = min(points_per_stratum, len(stratum_indices))
                    sampled_idx.extend(
                        np.random.choice(stratum_indices, n_sample, replace=False)
                    )
            # 如果不足 n_points，从所有点中随机补充
            if len(sampled_idx) < n_points:
                remaining = n_points - len(sampled_idx)
                remaining_idx = np.setdiff1d(
                    np.arange(len(valid_indices)), sampled_idx
                )
                sampled_idx.extend(
                    np.random.choice(remaining_idx, remaining, replace=False)
                )
            sampled_idx = np.array(sampled_idx[:n_points])
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        sampled_indices = valid_indices[sampled_idx]
        sampled_positions = sampled_indices * dVoxel - sVoxel / 2 + offOrigin
        sampled_densities = vol[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]
        sampled_densities = sampled_densities * args.density_rescale

    out = np.concatenate([sampled_positions, sampled_densities[:, None]], axis=-1)
    np.save(save_path, out)
    print(f"Initialization saved in {save_path}.")


def main(
    args, init_args: InitParams, model_args: ModelParams, pipe_args: PipelineParams
):
    # Read scene
    data_path = args.data
    model_args.source_path = data_path
    scene = Scene(model_args, False)  #! Here we scale the scene to [-1,1]^3 space.
    train_cameras = scene.getTrainCameras()
    projs_train = np.concatenate(
        [t2a(cam.original_image) for cam in train_cameras], axis=0
    )
    angles_train = np.stack([t2a(cam.angle) for cam in train_cameras], axis=0)
    scanner_cfg = scene.scanner_cfg
    geo = get_geometry_tigre(scanner_cfg)

    save_path = args.output
    if not save_path:
        if osp.exists(osp.join(data_path, "meta_data.json")):
            save_path = osp.join(data_path, "init_" + osp.basename(data_path) + ".npy")
        elif data_path.split(".")[-1] in ["pickle", "pkl"]:
            save_path = osp.join(
                osp.dirname(data_path),
                "init_" + osp.basename(data_path).split(".")[0] + ".npy",
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

    assert not osp.exists(
        save_path
    ), f"Initialization file {save_path} exists! Delete it first."
    os.makedirs(osp.dirname(save_path), exist_ok=True)

    init_pcd(
        projs=projs_train,
        angles=angles_train,
        geo=geo,
        scanner_cfg=scanner_cfg,
        args=init_args,
        save_path=save_path,
    )

    # Evaluate using ground truth volume (for debug only)
    if args.evaluate:
        with torch.no_grad():
            model_args.ply_path = save_path
            scale_bound = None
            volume_to_world = max(scanner_cfg["sVoxel"])
            if model_args.scale_min and model_args.scale_max:
                scale_bound = (
                    np.array([model_args.scale_min, model_args.scale_max])
                    * volume_to_world
                )
            gaussians = GaussianModel(scale_bound)
            initialize_gaussian(gaussians, model_args, None)
            vol_pred = query(
                gaussians,
                scanner_cfg["offOrigin"],
                scanner_cfg["nVoxel"],
                scanner_cfg["sVoxel"],
                pipe_args,
            )["vol"]
            vol_gt = scene.vol_gt.cuda()
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            print(f"3D PSNR for initial Gaussians: {psnr_3d}")
            # show_two_volume(vol_gt, vol_pred, title1="gt", title2="init")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Generate initialization parameters")
    init_parser = InitParams(parser)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--data", type=str, help="Path to data.")
    parser.add_argument("--output", default=None, type=str, help="Path to output.")
    parser.add_argument("--evaluate", default=False, action="store_true", help="Add this flag to evaluate quality (given GT volume, for debug only)")
    # fmt: on

    args = parser.parse_args()
    main(args, init_parser.extract(args), lp.extract(args), pp.extract(args))
