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
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from r2_gaussian.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min = 0.0005  # percent of volume size
        self.scale_max = 0.5  # percent of volume size
        self.eval = True
        
        # 单模型训练（删除 CoR-GS 双模型架构）
        self.gaussiansN = 1  # 高斯场数量（单模型）

        # 🌟 FSGS 深度监督参数 (完整 FSGS 实现)
        self.enable_fsgs_depth = False  # 是否启用深度监督（FSGS 核心创新）
        self.fsgs_depth_model = "dpt_hybrid"  # 深度估计模型: dpt_large/dpt_hybrid/midas_small/disabled
        self.fsgs_depth_weight = 0.05  # 深度 loss 权重（论文建议 0.01-0.1）
        self.enable_fsgs_pseudo_views = False  # 是否启用伪视角生成（FSGS 核心创新）
        self.num_fsgs_pseudo_views = 10  # 伪视角数量
        self.fsgs_noise_std = 0.05  # 伪视角位置噪声标准差（用于相机位置，论文 Eq.5）
        self.fsgs_start_iter = 5000  # FSGS 功能启动迭代数
        
        # 🌟 FSGS Proximity-guided 密化参数
        self.enable_fsgs_proximity = False  # 是否启用 FSGS proximity-guided 密化
        self.proximity_threshold = 5.0  # proximity score 阈值
        self.enable_medical_constraints = True  # 启用医学约束（增强 FSGS 性能，减少过拟合）
        self.proximity_organ_type = "foot"  # 器官类型
        self.proximity_k_neighbors = 5  # 计算 proximity 的邻居数量

        # 🎯 X²-Gaussian K-Planes 参数
        self.enable_kplanes = False  # 是否启用 K-Planes 空间分解
        self.kplanes_resolution = 64  # K-Planes 平面分辨率 (默认 64)
        self.kplanes_dim = 32  # K-Planes 特征维度 (默认 32)

        # 🎯 K-Planes MLP Decoder 参数
        self.kplanes_decoder_hidden = 128  # Decoder 隐藏层维度 (默认 128)
        self.kplanes_decoder_layers = 3  # Decoder MLP 层数 (默认 3)

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = 30_000
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.density_lr_max_steps = 30_000
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = 30_000
        self.lambda_dssim = 0.25
        self.lambda_tv = 0.05
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 2.0e-4  # 提高阈值减少过拟合（原5e-5过低导致过度密化）
        self.densify_scale_threshold = 0.1  # percent of volume size
        self.max_screen_size = None
        self.max_scale = None  # percent of volume size
        self.max_num_gaussians = 500_000

        # 伪标签和深度相关参数 - 参考X-Gaussian-depth实现
        self.sample_pseudo_interval = 1
        self.start_sample_pseudo = 2000
        self.end_sample_pseudo = 10000
        self.start_perturbation = 2000
        self.depth_weight = 0.05
        self.depth_pseudo_weight = 0.0  # float 类型，默认关闭
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05

        # Bino: Opacity Decay Strategy (不透明度衰减策略)
        # 论文: Binocular-Guided 3D Gaussian Splatting (NeurIPS 2024)
        self.enable_opacity_decay = False  # 默认关闭，保持向下兼容
        self.opacity_decay_factor = 0.995  # 衰减系数，论文推荐值

        # 🆕 Bino: Binocular Stereo Consistency Loss (双目立体一致性损失)
        # 论文核心创新：通过构建虚拟双目视角对，利用视差约束深度
        self.enable_binocular_consistency = False  # 是否启用双目一致性损失
        self.binocular_max_angle_offset = 0.1  # 最大角度偏移(弧度)，CT建议0.05-0.15
        self.binocular_start_iter = 10000  # 开始应用双目损失的迭代数 (CT可提前)
        self.binocular_warmup_iters = 2000  # 损失权重warmup迭代数
        self.binocular_smooth_weight = 0.05  # 视差平滑损失权重，论文推荐0.05
        self.binocular_loss_weight = 0.1  # 双目一致性损失总权重
        self.binocular_depth_method = "weighted_average"  # 深度估计方法: weighted_average/max_density/first_surface

        # 🎯 X²-Gaussian K-Planes 优化参数 (2025-01-18, 修正 2025-01-23)
        # 对齐 X²-Gaussian 原版设置：grid_lr_init=0.002, grid_lr_final=0.0002
        self.kplanes_lr_init = 0.002  # K-Planes 初始学习率（修正：0.00016 → 0.002，提升 12.5 倍）
        self.kplanes_lr_final = 0.0002  # K-Planes 最终学习率（修正：0.0000016 → 0.0002，提升 125 倍）
        self.kplanes_lr_max_steps = 30000  # K-Planes 学习率衰减步数

        # 🎯 X²-Gaussian TV 正则化参数 (2025-01-18, 修正 2025-01-23)
        # 对齐 X²-Gaussian 原版设置：plane_tv_weight=0.0001, L2 损失
        self.lambda_plane_tv = 0.0  # TV 正则化权重 (0 表示不启用)
        self.plane_tv_weight_proposal = [0.0001, 0.0001, 0.0001]  # 每个平面的 TV 权重 [xy, xz, yz]
        self.tv_loss_type = "l2"  # TV 损失类型（修正："l1" → "l2"，对齐原版）

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = osp.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
