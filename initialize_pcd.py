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

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:
    gaussian_filter = None

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
       python initialize_pcd.py --data <data_path>

    2. SPS（推荐：adaptive，随视角数自动减弱先验，避免高视角过度集中）：
       python initialize_pcd.py --data <data_path> --enable_sps --sps_strategy adaptive

    3. 复现旧版密度加权（不做视角自适应）：
       python initialize_pcd.py --data <data_path> --enable_sps --sps_strategy density_weighted --sps_density_gamma 1.0

    4. SPS + 降噪预处理：
       python initialize_pcd.py --data <data_path> --enable_sps --sps_denoise
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
        self.sps_auto_denoise = True  # [SPS] 自动在稀疏视角下启用去噪（仅影响采样先验）
        self.sps_auto_denoise_max_views = 6  # [SPS][auto] ≤该视角数时自动去噪
        # 采样策略:
        # - density_weighted: 密度加权采样（公式: P(x) = ρ(x)^γ / Σρ^γ）
        # - stratified: 分层采样（确保不同密度区间都有代表）
        # - mixed: 混合采样（均匀覆盖 + 密度加权，避免过度集中）
        # - adaptive: 视角自适应 mixed（views 越多先验越弱，默认推荐）
        self.sps_strategy = "adaptive"
        self.sps_uniform_ratio = 0.2  # [SPS][mixed] 均匀采样占比（其余为密度加权）
        self.sps_density_gamma = 1.0  # [SPS] 密度权重幂指数 γ（<1 更平滑，>1 更尖锐）
        self.sps_density_clip_percentile = 99.5  # [SPS] 采样权重密度裁剪百分位（<=0 或 >=100 表示不裁剪）
        self.sps_view_adaptive = True  # [SPS][adaptive] 按视角数自动调节采样强度
        self.sps_uniform_ratio_max = 0.6  # [SPS][adaptive] 均匀占比上限（防止过度退化为随机）
        self.sps_density_gamma_min = 0.5  # [SPS][adaptive] γ 下限（防止过度拉平）
        self.sps_num_strata = 5  # [SPS][stratified] 分层数量

        # 向下兼容旧参数名
        self.enable_denoise = False  # [兼容] 旧名，映射到 sps_denoise
        self.denoise_sigma = 3.0  # [兼容] 旧名
        self.sampling_strategy = "random"  # [兼容] 旧名

        super().__init__(parser, "Initialization Parameters")

    def extract(self, args):
        g = super().extract(args)

        # ---------------------------------------------------------------------
        # Backward/forward compatibility: sync alias parameters
        # ---------------------------------------------------------------------
        # Denoise flags: either alias enables denoise
        enable_denoise = bool(
            getattr(g, "sps_denoise", False) or getattr(g, "enable_denoise", False)
        )
        g.sps_denoise = enable_denoise
        g.enable_denoise = enable_denoise

        def _sync_scalar_alias(new_name: str, old_name: str):
            new_default = getattr(self, new_name)
            old_default = getattr(self, old_name)
            new_val = getattr(g, new_name, new_default)
            old_val = getattr(g, old_name, old_default)

            new_set = new_val != new_default
            old_set = old_val != old_default

            if new_set and old_set and new_val != old_val:
                raise ValueError(
                    f"Conflicting args: --{new_name}={new_val} vs --{old_name}={old_val}. "
                    f"Please set only one."
                )

            if new_set and not old_set:
                old_val = new_val
            elif old_set and not new_set:
                new_val = old_val

            setattr(g, new_name, new_val)
            setattr(g, old_name, old_val)

        _sync_scalar_alias("sps_denoise_sigma", "denoise_sigma")
        _sync_scalar_alias("sps_strategy", "sampling_strategy")

        return g


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

        num_views = int(projs.shape[0]) if hasattr(projs, "shape") else int(len(angles))

        # [SPS] 降噪预处理（可选）
        # - 手动启用: --sps_denoise
        # - 自动启用: enable_sps 且 views<=sps_auto_denoise_max_views
        denoise_sigma = float(getattr(args, "sps_denoise_sigma", 3.0))
        should_denoise = bool(getattr(args, "sps_denoise", False))
        if (
            (not should_denoise)
            and bool(getattr(args, "enable_sps", False))
            and bool(getattr(args, "sps_auto_denoise", True))
        ):
            auto_max_views = int(getattr(args, "sps_auto_denoise_max_views", 6))
            if num_views <= auto_max_views:
                should_denoise = True
                print(f"[SPS] Auto denoise enabled (views={num_views} <= {auto_max_views}).")

        if should_denoise:
            if gaussian_filter is None:
                print("[SPS] Warning: scipy not installed; skip gaussian denoise.")
            else:
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
        print(f"  - 训练视角数: {num_views}")

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
        elif strategy in ["density_weighted", "mixed", "adaptive", "stratified"]:
            densities_flat = vol[
                valid_indices[:, 0],
                valid_indices[:, 1],
                valid_indices[:, 2],
            ].astype(np.float64)

            # 对采样权重做可选裁剪，减少极少数异常高密度（FDK 伪影/噪声尖峰）对采样的主导
            clip_p = float(getattr(args, "sps_density_clip_percentile", 99.5))
            densities_for_sampling = densities_flat
            if np.isfinite(clip_p) and 0.0 < clip_p < 100.0:
                clip_val = np.percentile(densities_for_sampling, clip_p)
                if np.isfinite(clip_val):
                    densities_for_sampling = np.clip(densities_for_sampling, None, clip_val)

            # 统一处理 gamma（density_weighted / mixed / adaptive 可用）
            gamma = float(getattr(args, "sps_density_gamma", 1.0))
            gamma = max(gamma, 1e-6)

            if strategy == "density_weighted":
                # [SPS] 密度加权采样：P(x) = ρ(x)^γ / Σρ^γ
                print(
                    f"[SPS] Using density-weighted sampling strategy "
                    f"(gamma={gamma:.3f}, clip_p={clip_p:.1f})."
                )
                weights = np.power(densities_for_sampling, gamma)
                weight_sum = weights.sum()
                if not np.isfinite(weight_sum) or weight_sum <= 0:
                    raise ValueError(
                        f"[SPS] Invalid density weights (sum={weight_sum}). "
                        f"Try increasing density_thresh or reducing n_points."
                    )
                probs = weights / weight_sum
                sampled_idx = np.random.choice(
                    len(valid_indices), n_points, replace=False, p=probs
                )
            elif strategy in ["mixed", "adaptive"]:
                # [SPS] 混合采样：先均匀覆盖一部分点，再对剩余点做密度加权（避免过度集中）
                uniform_ratio = float(getattr(args, "sps_uniform_ratio", 0.2))
                uniform_ratio = min(max(uniform_ratio, 0.0), 1.0)

                if strategy == "adaptive" and bool(getattr(args, "sps_view_adaptive", True)):
                    # views 越多，先验越弱：
                    # - 均匀占比 ↑（覆盖更多低密度/边界区域）
                    # - gamma ↓（拉平密度分布，避免骨/高密度过采样）
                    import math
                    scale_up = math.sqrt(max(num_views, 1) / 3.0)
                    uniform_ratio_max = float(getattr(args, "sps_uniform_ratio_max", 0.6))
                    gamma_min = float(getattr(args, "sps_density_gamma_min", 0.5))
                    uniform_ratio = min(uniform_ratio * scale_up, uniform_ratio_max)
                    gamma = max(gamma / scale_up, gamma_min)

                n_uniform = int(round(n_points * uniform_ratio))
                n_uniform = max(0, min(n_points, n_uniform))
                n_weighted = n_points - n_uniform

                print(
                    f"[SPS] Using {strategy} sampling strategy: "
                    f"uniform_ratio={uniform_ratio:.3f} (n={n_uniform}), "
                    f"gamma={gamma:.3f} (weighted_n={n_weighted}), "
                    f"clip_p={clip_p:.1f}, "
                    f"views={num_views}."
                )

                sampled_parts = []
                if n_uniform > 0:
                    uniform_idx = np.random.choice(
                        len(valid_indices), n_uniform, replace=False
                    )
                    sampled_parts.append(uniform_idx)
                else:
                    uniform_idx = None

                if n_weighted > 0:
                    weights = np.power(densities_for_sampling, gamma)
                    if uniform_idx is not None:
                        weights[uniform_idx] = 0.0
                    weight_sum = weights.sum()
                    if not np.isfinite(weight_sum) or weight_sum <= 0:
                        raise ValueError(
                            f"[SPS] Invalid density weights (sum={weight_sum}). "
                            f"Try increasing density_thresh or reducing n_points."
                        )
                    probs = weights / weight_sum
                    weighted_idx = np.random.choice(
                        len(valid_indices), n_weighted, replace=False, p=probs
                    )
                    sampled_parts.append(weighted_idx)

                sampled_idx = np.concatenate(sampled_parts, axis=0)
            elif strategy == "stratified":
                # [SPS] 分层采样：确保不同密度区间都有代表
                print(f"[SPS] Using stratified sampling strategy.")
                # 将密度分为若干层级
                num_strata = int(getattr(args, "sps_num_strata", 5))
                num_strata = max(2, num_strata)
                points_per_stratum = n_points // num_strata
                sampled_idx = []
                for i in range(num_strata):
                    lower_q = (i / num_strata) * 100.0
                    upper_q = ((i + 1) / num_strata) * 100.0
                    lower = np.percentile(densities_for_sampling, lower_q)
                    upper = np.percentile(densities_for_sampling, upper_q)
                    if i == num_strata - 1:
                        stratum_mask = (densities_for_sampling >= lower) & (densities_for_sampling <= upper)
                    else:
                        stratum_mask = (densities_for_sampling >= lower) & (densities_for_sampling < upper)
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
