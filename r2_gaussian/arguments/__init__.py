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

"""
================================================================================
R²-Gaussian 参数配置 - 消融实验指南
================================================================================

集成的技术模块：
  - [BASELINE] R²-Gaussian 原版参数
  - [X2-GS]    X²-Gaussian K-Planes 密度调制 + TV 正则化
  - [BINO]     Binocular-Guided 双目一致性损失
  - [FSGS]     FSGS 伪视角深度监督 + Proximity 密化
  - [INIT-PCD] 密度加权采样初始化 (需预生成点云，见 initialize_pcd.py)

--------------------------------------------------------------------------------
消融实验配置示例：
--------------------------------------------------------------------------------

1. 纯 BASELINE（不加任何技术）：
   --enable_kplanes false
   --enable_binocular_consistency false
   --enable_fsgs_depth false
   --enable_fsgs_pseudo_views false
   # init-pcd: 使用默认随机采样点云 (sampling_strategy=random)

2. 仅启用 INIT-PCD（密度加权采样初始化）：
   --enable_kplanes false
   --enable_binocular_consistency false
   --enable_fsgs_depth false
   # 需预先生成点云: python initialize_pcd.py -s <data_path> --sampling_strategy density_weighted
   # 然后训练时指定 --ply_path <预生成的点云路径>

3. 仅启用 X2-GS（K-Planes 密度调制）：
   --enable_kplanes true
   --lambda_plane_tv 0.0002
   --enable_binocular_consistency false
   --enable_fsgs_depth false

4. 仅启用 BINO（双目一致性）：
   --enable_kplanes false
   --enable_binocular_consistency true
   --binocular_start_iter 7000
   --binocular_loss_weight 0.15
   --enable_fsgs_depth false

5. 仅启用 FSGS（深度监督 + 伪视角）：
   --enable_kplanes false
   --enable_binocular_consistency false
   --enable_fsgs_depth true
   --enable_fsgs_pseudo_views true
   --enable_fsgs_proximity true
   --enable_medical_constraints true

6. X2-GS + BINO 组合：
   --enable_kplanes true
   --lambda_plane_tv 0.0002
   --enable_binocular_consistency true
   --binocular_start_iter 7000
   --binocular_loss_weight 0.15
   --enable_fsgs_depth false

7. 全部启用（Full Combo）：
   --enable_kplanes true
   --lambda_plane_tv 0.0002
   --enable_binocular_consistency true
   --binocular_start_iter 7000
   --enable_fsgs_depth true
   --enable_fsgs_pseudo_views true
   --enable_fsgs_proximity true
   # 搭配 init-pcd 预生成的密度加权点云效果最佳

================================================================================
"""

import os
import sys
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from r2_gaussian.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # ════════════════════════════════════════════════════════════════════
        # [BASELINE] R²-Gaussian 原版参数
        # ════════════════════════════════════════════════════════════════════
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # [BASELINE] 初始化点云路径 (空则自动查找 init_*.npy)
        self.scale_min = 0.0005  # [BASELINE] 高斯最小尺度 (体积百分比)
        self.scale_max = 0.5  # [BASELINE] 高斯最大尺度 (体积百分比)
        self.eval = True  # [BASELINE] 是否启用评估
        self.gaussiansN = 1  # [BASELINE] 高斯场数量（固定为 1，单模型训练）

        # ════════════════════════════════════════════════════════════════════
        # [FSGS] 深度监督参数 - Few-shot 3D Gaussian Splatting
        # 论文: FSGS (arXiv:2312.00451)
        # 主开关: enable_fsgs_depth, enable_fsgs_pseudo_views
        # ════════════════════════════════════════════════════════════════════
        self.enable_fsgs_depth = False  # [FSGS] 主开关：启用 MiDaS 深度监督
        self.fsgs_depth_model = "dpt_hybrid"  # [FSGS] 深度模型: dpt_large/dpt_hybrid/midas_small
        self.fsgs_depth_weight = 0.05  # [FSGS] 深度 loss 权重（推荐 0.01-0.1）
        self.enable_fsgs_pseudo_views = False  # [FSGS] 主开关：启用伪视角生成
        self.num_fsgs_pseudo_views = 10  # [FSGS] 每次迭代的伪视角数量
        self.fsgs_noise_std = 0.05  # [FSGS] 伪视角位置噪声标准差
        self.fsgs_start_iter = 5000  # [FSGS] 功能启动迭代数

        # ════════════════════════════════════════════════════════════════════
        # [FSGS] Proximity-guided 密化参数
        # 主开关: enable_fsgs_proximity
        # ════════════════════════════════════════════════════════════════════
        self.enable_fsgs_proximity = False  # [FSGS] 主开关：启用 proximity-guided 密化
        self.proximity_threshold = 5.0  # [FSGS] proximity score 阈值
        self.enable_medical_constraints = True  # [FSGS] 医学约束（推荐启用）
        self.proximity_organ_type = "foot"  # [FSGS] 器官类型（影响约束策略）
        self.proximity_k_neighbors = 5  # [FSGS] 邻居数量

        # ════════════════════════════════════════════════════════════════════
        # [X2-GS] X²-Gaussian K-Planes 参数
        # 论文: X²-Gaussian (arXiv:2403.04116)
        # 主开关: enable_kplanes
        # ════════════════════════════════════════════════════════════════════
        self.enable_kplanes = False  # [X2-GS] 主开关：启用 K-Planes 空间分解
        self.kplanes_resolution = 64  # [X2-GS] K-Planes 平面分辨率
        self.kplanes_dim = 32  # [X2-GS] K-Planes 特征维度
        self.kplanes_decoder_hidden = 128  # [X2-GS] MLP Decoder 隐藏层维度
        self.kplanes_decoder_layers = 3  # [X2-GS] MLP Decoder 层数

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
        # ════════════════════════════════════════════════════════════════════
        # [BASELINE] R²-Gaussian 原版优化参数
        # ════════════════════════════════════════════════════════════════════
        self.iterations = 30_000  # [BASELINE] 总迭代次数
        self.position_lr_init = 0.0002  # [BASELINE] 位置学习率（初始）
        self.position_lr_final = 0.00002  # [BASELINE] 位置学习率（最终）
        self.position_lr_max_steps = 30_000  # [BASELINE] 位置学习率衰减步数
        self.density_lr_init = 0.01  # [BASELINE] 密度学习率（初始）
        self.density_lr_final = 0.001  # [BASELINE] 密度学习率（最终）
        self.density_lr_max_steps = 30_000  # [BASELINE] 密度学习率衰减步数
        self.scaling_lr_init = 0.005  # [BASELINE] 尺度学习率（初始）
        self.scaling_lr_final = 0.0005  # [BASELINE] 尺度学习率（最终）
        self.scaling_lr_max_steps = 30_000  # [BASELINE] 尺度学习率衰减步数
        self.rotation_lr_init = 0.001  # [BASELINE] 旋转学习率（初始）
        self.rotation_lr_final = 0.0001  # [BASELINE] 旋转学习率（最终）
        self.rotation_lr_max_steps = 30_000  # [BASELINE] 旋转学习率衰减步数
        self.lambda_dssim = 0.25  # [BASELINE] DSSIM loss 权重
        self.lambda_tv = 0.05  # [BASELINE] 3D TV 正则化权重
        self.tv_vol_size = 32  # [BASELINE] TV 正则化体素尺寸
        self.density_min_threshold = 0.00001  # [BASELINE] 密度最小阈值
        self.densification_interval = 100  # [BASELINE] 密化间隔
        self.densify_from_iter = 500  # [BASELINE] 密化起始迭代
        self.densify_until_iter = 15000  # [BASELINE] 密化结束迭代
        self.densify_grad_threshold = 2.0e-4  # [BASELINE] 密化梯度阈值
        self.densify_scale_threshold = 0.1  # [BASELINE] 密化尺度阈值 (体积百分比)
        self.max_screen_size = None  # [BASELINE] 最大屏幕尺寸
        self.max_scale = None  # [BASELINE] 最大尺度 (体积百分比)
        self.max_num_gaussians = 500_000  # [BASELINE] 最大高斯点数
        self.feature_lr = 0.0025  # [BASELINE] 特征学习率
        self.opacity_lr = 0.05  # [BASELINE] 不透明度学习率

        # ════════════════════════════════════════════════════════════════════
        # [BASELINE] 伪标签和深度相关参数（R²-Gaussian 原版）
        # ════════════════════════════════════════════════════════════════════
        self.sample_pseudo_interval = 1  # [BASELINE] 伪标签采样间隔
        self.start_sample_pseudo = 2000  # [BASELINE] 伪标签起始迭代
        self.end_sample_pseudo = 10000  # [BASELINE] 伪标签结束迭代
        self.start_perturbation = 2000  # [BASELINE] 扰动起始迭代
        self.depth_weight = 0.05  # [BASELINE] 深度 loss 权重
        self.depth_pseudo_weight = 0.0  # [BASELINE] 伪深度 loss 权重（0=关闭）

        # ════════════════════════════════════════════════════════════════════
        # [BINO] Binocular-Guided 3D Gaussian Splatting 参数
        # 论文: Binocular-Guided 3DGS (NeurIPS 2024)
        # 主开关: enable_binocular_consistency
        # ════════════════════════════════════════════════════════════════════
        self.enable_opacity_decay = False  # [BINO] 不透明度衰减（实验证明 CT 场景关闭更优）
        self.opacity_decay_factor = 0.995  # [BINO] 衰减系数
        self.enable_binocular_consistency = False  # [BINO] 主开关：启用双目一致性损失
        self.binocular_max_angle_offset = 0.06  # [BINO] 最大角度偏移（弧度，推荐 0.05-0.08）
        self.binocular_start_iter = 7000  # [BINO] 起始迭代（CT 可提前到 7000）
        self.binocular_warmup_iters = 3000  # [BINO] warmup 迭代数
        self.binocular_smooth_weight = 0.05  # [BINO] 视差平滑权重
        self.binocular_loss_weight = 0.15  # [BINO] 双目损失总权重（推荐 0.1-0.2）
        self.binocular_depth_method = "weighted_average"  # [BINO] 深度估计方法

        # ════════════════════════════════════════════════════════════════════
        # [X2-GS] X²-Gaussian K-Planes 优化参数
        # 论文: X²-Gaussian (arXiv:2403.04116)
        # 需配合 ModelParams.enable_kplanes=True 使用
        # ════════════════════════════════════════════════════════════════════
        self.kplanes_lr_init = 0.002  # [X2-GS] K-Planes 初始学习率
        self.kplanes_lr_final = 0.0002  # [X2-GS] K-Planes 最终学习率
        self.kplanes_lr_max_steps = 30000  # [X2-GS] K-Planes 学习率衰减步数
        self.lambda_plane_tv = 0.0  # [X2-GS] K-Planes TV 正则化权重（0=关闭，推荐 0.0002）
        self.plane_tv_weight_proposal = [0.0001, 0.0001, 0.0001]  # [X2-GS] 每个平面的 TV 权重 [xy, xz, yz]
        self.tv_loss_type = "l2"  # [X2-GS] TV 损失类型（l2 效果更优）

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
