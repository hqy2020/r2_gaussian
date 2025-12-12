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
SPAGS: Spatial-aware Progressive Adaptive Gaussian Splatting
================================================================================

三阶段渐进式优化框架：
  - [SPS]  Stage 1: Spatial Prior Seeding - 空间先验播种（密度加权初始化）
  - [GAR]  Stage 2: Geometry-aware Refinement - 几何感知细化（邻近密化）
  - [ADM]  Stage 3: Adaptive Density Modulation - 自适应密度调制（K-Planes）

--------------------------------------------------------------------------------
SPAGS 消融实验配置：
--------------------------------------------------------------------------------

1. Baseline（不加任何技术）：
   --enable_sps false --enable_gar false --enable_adm false

2. 仅 SPS（空间先验播种）：
   --enable_sps true --enable_gar false --enable_adm false
   # 需预生成点云: python initialize_pcd.py -s <data_path> --enable_sps

3. 仅 GAR（几何感知细化）：
   --enable_sps false --enable_gar true --enable_adm false

4. 仅 ADM（自适应密度调制）：
   --enable_sps false --enable_gar false --enable_adm true

5. SPS + GAR：
   --enable_sps true --enable_gar true --enable_adm false

6. SPS + ADM：
   --enable_sps true --enable_gar false --enable_adm true

7. GAR + ADM：
   --enable_sps false --enable_gar true --enable_adm true

8. Full SPAGS（全部启用）：
   --enable_sps true --enable_gar true --enable_adm true

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

        # ════════════════════════════════════════════════════════════════════
        # [GAR] Geometry-aware Refinement - 邻近感知密化参数
        # Stage 2: 几何感知细化（Proximity-guided Densification）
        # 主开关: enable_gar (或兼容旧名 enable_gar_proximity / enable_fsgs_proximity)
        # ════════════════════════════════════════════════════════════════════
        self.enable_gar = False  # [GAR] 主开关：启用几何感知细化（邻近密化）
        self.enable_gar_proximity = False  # [兼容] 旧名，映射到 enable_gar
        self.gar_proximity_threshold = 0.05  # [GAR] proximity score 阈值（场景归一化到[-1,1]³，典型分数范围0.01-0.5）
        self.gar_proximity_k = 5  # [GAR] 邻居数量
        # 向下兼容旧参数名
        self.enable_fsgs_proximity = False  # [兼容] 旧名，映射到 enable_gar_proximity
        self.proximity_threshold = 0.05  # [兼容] 旧名（注意：场景归一化后邻近分数范围约0.01-0.5）
        self.proximity_k_neighbors = 5  # [兼容] 旧名
        # 邻近密化时间参数
        self.proximity_start_iter = 1000  # [GAR] 邻近密化开始迭代
        self.proximity_interval = 500  # [GAR] 邻近密化间隔
        self.proximity_until_iter = 15000  # [GAR] 邻近密化结束迭代

        # 🆕 GAR 优化参数（基于诊断分析优化，避免 77% 过度密化问题）
        self.gar_adaptive_threshold = True  # [GAR] 启用自适应阈值（基于邻近分数分布）- 默认启用解决固定阈值0.05导致的过度密化
        self.gar_adaptive_method = "percentile"  # [GAR] 自适应方法: percentile/std/iqr
        self.gar_adaptive_percentile = 90.0  # [GAR] percentile 百分位（90=只密化最稀疏10%，理想范围5-15%）
        self.gar_progressive_decay = False  # [GAR] 启用渐进衰减（训练后期减少密化）
        self.gar_decay_start_ratio = 0.7  # [GAR] 衰减开始进度（0.7=70%进度后开始）
        self.gar_final_strength = 0.5  # [GAR] 最终强度（0.5=阈值提高2倍）
        self.gar_gradient_filter = False  # [GAR] 启用梯度过滤（只密化高梯度点）
        self.gar_gradient_threshold = 0.0002  # [GAR] 梯度过滤阈值
        self.gar_max_candidates = 5000  # [GAR] 每次密化最大候选点数（避免 OOM）

        # ════════════════════════════════════════════════════════════════════
        # [ADM] Adaptive Density Modulation - K-Planes 空间调制参数
        # Stage 3: 自适应密度调制
        # 主开关: enable_adm (或兼容旧名 enable_kplanes)
        # ════════════════════════════════════════════════════════════════════
        self.enable_adm = False  # [ADM] 主开关：启用自适应密度调制
        self.adm_resolution = 64  # [ADM] K-Planes 平面分辨率
        self.adm_feature_dim = 32  # [ADM] K-Planes 特征维度
        self.adm_decoder_hidden = 128  # [ADM] MLP Decoder 隐藏层维度
        self.adm_decoder_layers = 3  # [ADM] MLP Decoder 层数
        self.adm_max_range = 0.3  # [ADM] 最大调制范围 (±30%)
        self.adm_view_adaptive = True  # [ADM] 🆕 视角自适应：自动根据训练视角数调整调制强度和TV正则化
        # 向下兼容旧参数名
        self.enable_kplanes = False  # [兼容] 旧名，映射到 enable_adm
        self.kplanes_resolution = 64  # [兼容] 旧名
        self.kplanes_dim = 32  # [兼容] 旧名
        self.kplanes_decoder_hidden = 128  # [兼容] 旧名
        self.kplanes_decoder_layers = 3  # [兼容] 旧名

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
        # [GAR] Geometry-aware Refinement - 辅助参数
        # 注意: GAR 主开关 enable_gar 在 ModelParams 中定义
        # ════════════════════════════════════════════════════════════════════
        self.enable_opacity_decay = False  # [GAR] 不透明度衰减（CT 场景关闭）
        self.opacity_decay_factor = 0.995  # [GAR] 衰减系数

        # ════════════════════════════════════════════════════════════════════
        # [ADM] Adaptive Density Modulation - K-Planes 优化参数
        # Stage 3: 自适应密度调制
        # 需配合 ModelParams.enable_adm=True 使用
        # ════════════════════════════════════════════════════════════════════
        self.adm_lr_init = 0.002  # [ADM] K-Planes 初始学习率
        self.adm_lr_final = 0.0002  # [ADM] K-Planes 最终学习率
        self.adm_lr_max_steps = 30000  # [ADM] K-Planes 学习率衰减步数
        self.adm_lambda_tv = 0.002  # [ADM] Plane TV 正则化权重（最优值）
        self.adm_tv_type = "l2"  # [ADM] TV 损失类型
        # [ADM] 训练调度参数（自适应置信度调制）
        self.adm_warmup_iters = 3000  # [ADM] warmup 迭代数（避免初期干扰）
        self.adm_decay_start = 20000  # [ADM] 调制衰减开始迭代
        self.adm_final_strength = 0.5  # [ADM] 最终调制强度（后期稳定）
        # 向下兼容旧参数名
        self.kplanes_lr_init = 0.002  # [兼容] 旧名
        self.kplanes_lr_final = 0.0002  # [兼容] 旧名
        self.kplanes_lr_max_steps = 30000  # [兼容] 旧名
        self.lambda_plane_tv = 0.002  # [兼容] 旧名，使用最优值
        self.plane_tv_weight_proposal = [1.0, 1.0, 1.0]  # [兼容] 均匀权重，由 lambda_plane_tv 控制总强度
        self.tv_loss_type = "l2"  # [兼容] 旧名

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
