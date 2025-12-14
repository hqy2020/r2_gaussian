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
import sys
import torch
from torch import nn
import numpy as np
import pickle
from plyfile import PlyData, PlyElement

sys.path.append("./")

from simple_knn._C import distCUDA2
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.utils.system_utils import mkdir_p
from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)

EPS = 1e-5


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            def _scaling_inverse_activation(y):
                normalized = (y - scale_min_bound) / (scale_max_bound - scale_min_bound)
                normalized = normalized.clamp(min=EPS, max=1.0 - EPS)
                return inverse_sigmoid(normalized)

            self.scaling_inverse_activation = _scaling_inverse_activation
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None, args=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        self.setup_functions()

        # K-Planes 支持（可选）
        # 支持新参数名 enable_adm 和旧参数名 enable_kplanes
        self.enable_kplanes = (
            getattr(args, 'enable_adm', False) or
            getattr(args, 'enable_kplanes', False)
        ) if args is not None else False
        if self.enable_kplanes:
            from r2_gaussian.gaussian.kplanes import KPlanesEncoder, DensityMLPDecoder
            # 支持新参数名 adm_* 和旧参数名 kplanes_*（优先使用新名）
            kplanes_resolution = getattr(args, 'adm_resolution', None) or getattr(args, 'kplanes_resolution', 64)
            kplanes_dim = getattr(args, 'adm_feature_dim', None) or getattr(args, 'kplanes_dim', 32)
            self.kplanes_encoder = KPlanesEncoder(
                grid_resolution=kplanes_resolution,
                feature_dim=kplanes_dim,
                num_levels=1,
                bounds=(-1.0, 1.0),
            ).cuda()

            # 🎯 MLP Decoder: 将 K-Planes 特征映射到 density 调制因子
            # 支持新参数名 adm_* 和旧参数名 kplanes_*（优先使用新名）
            kplanes_decoder_hidden = getattr(args, 'adm_decoder_hidden', None) or getattr(args, 'kplanes_decoder_hidden', 128)
            kplanes_decoder_layers = getattr(args, 'adm_decoder_layers', None) or getattr(args, 'kplanes_decoder_layers', 3)
            self.density_decoder = DensityMLPDecoder(
                input_dim=self.kplanes_encoder.get_output_dim(),  # 96
                hidden_dim=kplanes_decoder_hidden,
                num_layers=kplanes_decoder_layers
            ).cuda()
        else:
            self.kplanes_encoder = None
            self.density_decoder = None

        # 存储 ADM 调度参数（用于 _get_adm_strength）
        if self.enable_kplanes:
            # ADM 调制幅度
            self.adm_max_range = getattr(args, 'adm_max_range', 0.3) if args else 0.3

            # ADM 调度参数：训练侧会在 training_setup 中覆盖；评估侧没有 optimizer 时也需要可用
            def _get_or(name: str, default, cast):
                if args is None:
                    return cast(default)
                v = getattr(args, name, None)
                if v is None:
                    return cast(default)
                return cast(v)

            self.adm_warmup_iters = _get_or('adm_warmup_iters', 1000, int)
            self.adm_decay_start = _get_or('adm_decay_start', 20000, int)
            self.adm_final_strength = _get_or('adm_final_strength', 0.5, float)
            # total iters 优先从 cfg_args 中的 iterations 读取
            self.adm_total_iters = _get_or('iterations', 30000, int)

        # 🆕 视角数感知的自适应约束
        # 训练视角数，用于自动缩放 ADM 调制强度
        # 默认 3（稀疏视角），由 train.py 设置实际值
        self.num_train_views = 3
        # 默认开启：不同 views 下自动缩放 ADM 强度与 TV 权重（更符合消融脚本的使用习惯）
        self.adm_view_adaptive = getattr(args, 'adm_view_adaptive', True) if args else True
        # 🆕 X2GS 风格稳定性：去除全局密度缩放偏置（零均值调制）
        self.adm_zero_mean = getattr(args, 'adm_zero_mean', True) if args else True
        # 零均值模式（见 arguments/ 说明）
        self.adm_zero_mean_mode = getattr(args, 'adm_zero_mean_mode', 'density_confidence') if args else 'density_confidence'

    def _adm_apply_zero_mean(self, effective_offset: torch.Tensor, base_density: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        去除 ADM 的全局偏置（避免学成“整体系数缩放”）

        effective_offset: [N, 1]
        base_density:     [N, 1]
        confidence:       [N, 1]
        """
        mode = getattr(self, 'adm_zero_mean_mode', 'density_confidence') or 'density_confidence'

        if mode == 'unweighted':
            mean = effective_offset.mean()
        else:
            if mode == 'confidence':
                w = confidence.detach()
            elif mode == 'density':
                w = base_density.detach()
            elif mode == 'density_confidence':
                w = (base_density.detach() * confidence.detach())
            else:
                raise ValueError(f"Unknown adm_zero_mean_mode: {mode}")

            denom = w.sum().clamp_min(1e-8)
            mean = (effective_offset * w).sum() / denom

        return effective_offset - mean

    def capture(self):
        # K-Planes 状态（如果启用）
        kplanes_state = None
        decoder_state = None
        if self.enable_kplanes:
            if self.kplanes_encoder is not None:
                kplanes_state = self.kplanes_encoder.state_dict()
            if self.density_decoder is not None:
                decoder_state = self.density_decoder.state_dict()

        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
            kplanes_state,      # K-Planes encoder 状态
            decoder_state,      # K-Planes decoder 状态
        )

    def restore(self, model_args, training_args):
        # 处理新旧 checkpoint 格式兼容
        if len(model_args) == 10:
            # 旧格式（无 K-Planes）
            (
                self._xyz,
                self._scaling,
                self._rotation,
                self._density,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.scale_bound,
            ) = model_args
            kplanes_state, decoder_state = None, None
        else:
            # 新格式（包含 K-Planes）
            (
                self._xyz,
                self._scaling,
                self._rotation,
                self._density,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.scale_bound,
                kplanes_state,
                decoder_state,
            ) = model_args

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions

        # 恢复 K-Planes 状态（如果有）
        if self.enable_kplanes:
            if kplanes_state is not None and self.kplanes_encoder is not None:
                self.kplanes_encoder.load_state_dict(kplanes_state)
                print("✓ K-Planes encoder 状态已恢复")
            if decoder_state is not None and self.density_decoder is not None:
                self.density_decoder.load_state_dict(decoder_state)
                print("✓ K-Planes decoder 状态已恢复")

    def set_num_train_views(self, num_views: int):
        """
        设置训练视角数，用于视角自适应的 ADM 调制

        参数:
            num_views: 训练视角数量（如 3, 6, 9）
        """
        self.num_train_views = num_views
        print(f"✓ 设置训练视角数: {num_views}")
        if self.adm_view_adaptive and self.enable_kplanes:
            scale = self.get_view_adaptive_scale()
            print(f"  - ADM max_range 缩放因子: {scale:.3f}")
            print(f"  - 有效 max_range: {self.adm_max_range * scale:.4f}")

    def get_view_adaptive_scale(self) -> float:
        """
        获取视角自适应的缩放因子

        原理：
        - 视角越少（信息不足），需要更强的 ADM 调制（先验）
        - 视角越多（信息充足），需要更弱的 ADM 调制（避免干扰）

        公式: scale = 1 / sqrt(num_views / 3)
        - 3-views: scale = 1.0 (保持强调制)
        - 6-views: scale ≈ 0.71 (中等)
        - 9-views: scale ≈ 0.58 (弱调制)

        返回:
            float: 缩放因子 [0.58, 1.0]
        """
        if not self.adm_view_adaptive:
            return 1.0

        import math
        # 以 3 视角为基准，视角越多缩放越小
        scale = 1.0 / math.sqrt(self.num_train_views / 3.0)
        # 限制最小值，避免过度衰减
        return max(scale, 0.3)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_base_density(self):
        """不含 ADM 的基础密度（Softplus 激活后）。"""
        return self.density_activation(self._density)

    @property
    def get_density(self):
        base_density = self.get_base_density

        # 🎯 ADM 自适应密度调制（双头输出 + 训练调度 + 视角自适应）
        if self.enable_kplanes and self.kplanes_encoder is not None and self.density_decoder is not None:
            kplanes_feat = self.get_kplanes_features()  # [N, 96]

            # 双头输出: offset [-1,1] 控制调制方向, confidence [0,1] 控制调制强度
            offset, confidence = self.density_decoder(kplanes_feat)

            # 获取调制参数
            max_range = getattr(self, 'adm_max_range', 0.3)
            strength = self._get_adm_strength()

            # 🆕 视角自适应缩放: 视角越多，调制越弱
            view_scale = self.get_view_adaptive_scale()

            # 自适应调制公式: modulation = 1 + offset * confidence * max_range * strength * view_scale
            # - offset: 调制方向（增强或减弱密度）
            # - confidence: 网络学习的调制置信度（简单区域趋近0，困难区域趋近1）
            # - max_range: 最大调制范围（默认±30%）
            # - strength: 训练进程调度（warmup -> hold -> decay）
            # - view_scale: 视角自适应缩放（3v=1.0, 6v≈0.71, 9v≈0.58）
            effective_offset = offset * confidence * max_range * strength * view_scale
            # 🆕 去除全局偏置：避免 ADM 学成“整体系数缩放”，导致 densify/prune 不稳定
            if getattr(self, 'adm_zero_mean', False):
                effective_offset = self._adm_apply_zero_mean(
                    effective_offset,
                    base_density=base_density,
                    confidence=confidence,
                )
                # 零均值后可能轻微越界，做一次安全裁剪（保持最大调制范围）
                max_abs = float(max_range) * float(strength) * float(view_scale)
                if max_abs > 0:
                    effective_offset = torch.clamp(effective_offset, -max_abs, max_abs)
            modulation = 1.0 + effective_offset

            base_density = base_density * modulation

        return base_density

    def _get_adm_strength(self):
        """
        训练进程调度: warmup -> hold -> decay

        - [0, warmup): 从 0 线性增加到 1（避免初期干扰收敛）
        - [warmup, decay_start): 保持 1.0（正常调制）
        - [decay_start, total): 从 1.0 衰减到 final_strength（后期稳定）
        """
        if not hasattr(self, 'current_iteration'):
            return 1.0

        it = int(self.current_iteration)
        warmup = int(getattr(self, 'adm_warmup_iters', 1000))
        decay_start = int(getattr(self, 'adm_decay_start', 20000))
        final = float(getattr(self, 'adm_final_strength', 0.5))
        total = int(getattr(self, 'adm_total_iters', 30000))

        # 🆕 视角自适应：视角越多，ADM 越保守（更长 warmup、更低最终强度）
        if getattr(self, 'adm_view_adaptive', False):
            try:
                warmup = int(round(warmup * (float(self.num_train_views) / 3.0)))
            except Exception:
                pass
            try:
                final = float(final) * float(self.get_view_adaptive_scale())
            except Exception:
                pass

        if total <= 0:
            return 1.0

        if warmup > 0 and it < warmup:
            return it / float(warmup)

        if decay_start <= 0 or it < decay_start or decay_start >= total:
            return 1.0

        denom = float(total - decay_start)
        if denom <= 0:
            return 1.0
        progress = (it - decay_start) / denom
        progress = min(1.0, max(0.0, progress))
        return 1.0 - (1.0 - final) * progress

    def get_adm_diagnostics(self):
        """
        获取 ADM 诊断信息

        返回: dict 包含 offset, confidence, modulation 等统计
              如果 ADM 未启用，返回 None
        """
        if not self.enable_kplanes or self.kplanes_encoder is None or self.density_decoder is None:
            return None

        with torch.no_grad():
            kplanes_feat = self.get_kplanes_features()
            offset, confidence = self.density_decoder(kplanes_feat)

            max_range = getattr(self, 'adm_max_range', 0.3)
            strength = self._get_adm_strength()
            view_scale = self.get_view_adaptive_scale()

            # 计算基础密度（用于零均值加权与对比）
            base_density = self.get_base_density

            effective_offset = offset * confidence * max_range * strength * view_scale
            if getattr(self, 'adm_zero_mean', False):
                effective_offset = self._adm_apply_zero_mean(
                    effective_offset,
                    base_density=base_density,
                    confidence=confidence,
                )
                max_abs = float(max_range) * float(strength) * float(view_scale)
                if max_abs > 0:
                    effective_offset = torch.clamp(effective_offset, -max_abs, max_abs)
            modulation = 1.0 + effective_offset

            modulated_density = base_density * modulation
            density_change_pct = ((modulated_density - base_density) / (base_density + 1e-8) * 100)

            return {
                'offset': {
                    'mean': offset.mean().item(),
                    'std': offset.std().item(),
                    'min': offset.min().item(),
                    'max': offset.max().item()
                },
                'confidence': {
                    'mean': confidence.mean().item(),
                    'std': confidence.std().item(),
                    'min': confidence.min().item(),
                    'max': confidence.max().item()
                },
                'effective_offset': {
                    'mean': effective_offset.mean().item(),
                    'std': effective_offset.std().item(),
                    'min': effective_offset.min().item(),
                    'max': effective_offset.max().item()
                },
                'modulation': {
                    'mean': modulation.mean().item(),
                    'std': modulation.std().item(),
                    'min': modulation.min().item(),
                    'max': modulation.max().item()
                },
                'density_change_pct': {
                    'mean': density_change_pct.mean().item(),
                    'std': density_change_pct.std().item(),
                    'min': density_change_pct.min().item(),
                    'max': density_change_pct.max().item()
                },
                'adm_strength': strength,
                'view_scale': view_scale,
                'max_range': max_range,
                'num_gaussians': len(self._xyz)
            }

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def get_kplanes_features(self, xyz=None):
        """
        获取指定位置的 K-Planes 特征

        参数：
            xyz (torch.Tensor): 高斯中心坐标，形状 [N, 3]
                               如果为 None，则使用 self._xyz

        返回：
            features (torch.Tensor): K-Planes 特征，形状 [N, feature_dim * 3]
                                    如果未启用 K-Planes，返回 None
        """
        if not self.enable_kplanes or self.kplanes_encoder is None:
            return None

        if xyz is None:
            xyz = self._xyz

        return self.kplanes_encoder(xyz)

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(
            "Initialize gaussians from {} estimated points".format(
                fused_point_cloud.shape[0]
            )
        )
        fused_density = (
            self.density_inverse_activation(torch.tensor(density)).float().cuda()
        )
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #! Generate one gaussian for debugging purpose
        if False:
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        # K-Planes 参数组（可选）
        if self.enable_kplanes and self.kplanes_encoder is not None:
            kplanes_lr_init = getattr(training_args, 'kplanes_lr_init', 0.002)  # 与 arguments 默认值一致
            l.append({
                "params": self.kplanes_encoder.parameters(),
                "lr": kplanes_lr_init,
                "name": "kplanes"
            })

            # 🎯 K-Planes Decoder 参数组（使用更低学习率防止过拟合）
            # v3 改进：decoder 学习率降低到 encoder 的 0.5 倍
            if self.density_decoder is not None:
                decoder_lr_init = kplanes_lr_init * 0.5
                l.append({
                    "params": self.density_decoder.parameters(),
                    "lr": decoder_lr_init,
                    "name": "kplanes_decoder"
                })

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )

        # K-Planes 学习率调度器（可选）
        if self.enable_kplanes and self.kplanes_encoder is not None:
            kplanes_lr_init = getattr(training_args, 'kplanes_lr_init', 0.002)  # 与 arguments 默认值一致
            kplanes_lr_final = getattr(training_args, 'kplanes_lr_final', 0.0002)  # 与 arguments 默认值一致
            kplanes_lr_max_steps = getattr(training_args, 'kplanes_lr_max_steps', 30000)
            self.kplanes_scheduler_args = get_expon_lr_func(
                lr_init=kplanes_lr_init,
                lr_final=kplanes_lr_final,
                max_steps=kplanes_lr_max_steps,
            )

            # ADM 训练调度参数（用于 _get_adm_strength）
            self.adm_warmup_iters = getattr(training_args, 'adm_warmup_iters', 3000)
            self.adm_decay_start = getattr(training_args, 'adm_decay_start', 20000)
            self.adm_final_strength = getattr(training_args, 'adm_final_strength', 0.5)
            self.adm_total_iters = getattr(training_args, 'iterations', 30000)

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "kplanes":
                if self.enable_kplanes and hasattr(self, 'kplanes_scheduler_args'):
                    lr = self.kplanes_scheduler_args(iteration)
                    param_group["lr"] = lr
            # 🎯 K-Planes Decoder 学习率（v3：使用 0.5 倍防止过拟合）
            if param_group["name"] == "kplanes_decoder":
                if self.enable_kplanes and hasattr(self, 'kplanes_scheduler_args'):
                    lr = self.kplanes_scheduler_args(iteration) * 0.5
                    param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        # We save pickle files to store more information

        mkdir_p(os.path.dirname(path))

        xyz = t2a(self._xyz)
        densities = t2a(self._density)
        scale = t2a(self._scaling)
        rotation = t2a(self._rotation)

        out = {
            "xyz": xyz,
            "density": densities,
            "scale": scale,
            "rotation": rotation,
            "scale_bound": self.scale_bound,
        }

        # 🎯 保存 K-Planes 和 Decoder 状态（ADM 模块）
        if self.enable_kplanes:
            if self.kplanes_encoder is not None:
                out["kplanes_state"] = {k: v.cpu() for k, v in self.kplanes_encoder.state_dict().items()}
            if self.density_decoder is not None:
                out["decoder_state"] = {k: v.cpu() for k, v in self.density_decoder.state_dict().items()}

        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_base_density,
                torch.ones_like(self.get_base_density) * reset_density,
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self.scale_bound = data["scale_bound"]
        self.setup_functions()  # Reset activation functions

        # 🎯 加载 K-Planes 和 Decoder 状态（ADM 模块）
        if self.enable_kplanes:
            if "kplanes_state" in data and self.kplanes_encoder is not None:
                self.kplanes_encoder.load_state_dict(data["kplanes_state"])
                print("✓ K-Planes encoder 状态已从 pickle 恢复")
            if "decoder_state" in data and self.density_decoder is not None:
                self.density_decoder.load_state_dict(data["decoder_state"])
                print("✓ K-Planes decoder 状态已从 pickle 恢复")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 🎯 跳过 K-Planes 参数组（形状不匹配，不需要 prune）
            param = group["params"][0]
            if param.shape[0] != mask.shape[0]:
                continue
            
            stored_state = self.optimizer.state.get(param, None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[param]
                group["params"][0] = nn.Parameter(
                    (param[mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    param[mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 🎯 跳过 K-Planes 参数组（不需要 densification）
            if group["name"] not in tensors_dict:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_base_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_base_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = (self.get_base_density < min_density).squeeze()
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def opacity_decay(self, factor=0.995):
        """
        应用不透明度衰减策略

        对 density 应用指数衰减，自动过滤梯度低的冗余 Gaussians。
        在激活空间应用衰减（density *= factor），然后转回逆激活空间存储。

        Args:
            factor (float): 衰减系数，默认 0.995
                          - 梯度高的点：opacity增长 > 衰减 → 保留
                          - 梯度低的点：opacity增长 < 衰减 → 逐渐被剪枝

        Note:
            - R²-Gaussian使用Softplus激活([0,+∞))，而非标准3DGS的Sigmoid([0,1])
            - 衰减原理：降低低贡献点的密度值
            - 应在 densify_from_iter 后每次迭代调用
        """
        # 在激活空间应用衰减
        density = self.get_base_density * factor
        # 转回逆激活空间存储
        self._density.data = self.density_inverse_activation(density)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
