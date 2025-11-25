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
        self.ply_path = ""
        self.scale_min = 0.0005
        self.scale_max = 0.5
        self.eval = True

        # Coarse-Registration (CoReg) 参数
        self.coreg = True  # 是否启用粗配准
        self.coprune = True  # 是否启用协同剪枝
        self.coprune_threshold = 5  # 协同剪枝阈值
        self.perturbation = False  # 是否对参数加噪声
        self.onlyrgb = False  # 是否仅使用 RGB 数据
        self.normal = True  # 是否使用法向量
        self.pseudo_strategy = "single"  # 伪标签策略 single or multi
        self.sample_method = "uniform"  # 采样方法：uniform or maxmin
        self.add_num = 50  # 添加多少个伪视角

        # CoRGS (Co-Regularized Gaussian Splatting) 参数
        self.enable_corgs = False  # 是否启用 CoRGS 正则化
        self.corgs_tau = 0.3  # 温度参数，控制稀疏性
        self.corgs_coprune_freq = 500  # 协同剪枝频率
        self.corgs_pseudo_weight = 1.0  # 伪标签损失权重
        self.corgs_log_freq = 500  # 日志记录频率

        # Opacity Decay 相关参数
        self.opacity_decay = False  # 是否启用透明度衰减

        # Depth-supervised learning 深度监督学习参数
        self.enable_depth = False  # 是否启用深度监督
        self.depth_loss_weight = 0.0  # 深度损失权重
        self.depth_loss_type = "pearson"  # 深度损失类型：pearson, l1, l2
        self.depth_threshold = 0.01  # 深度有效性阈值

        # Multi-Gaussian and Pseudo-Label parameters
        self.multi_gaussian = True  # 是否使用多高斯拟合
        self.pseudo_labels = True  # 是否生成伪标签视角
        self.depth_constraint = False  # 是否使用深度约束
        self.num_additional_views = 50  # 额外生成的伪标签视角数量
        self.pseudo_confidence_threshold = 0.8  # 伪标签置信度阈值
        self.multi_gaussian_weight = 0.05  # 多高斯损失权重
        self.pseudo_label_weight = 0.05  # 伪标签损失权重

        # FSGS (Few-Shot Gaussian Splatting) 参数
        self.enable_fsgs_pseudo = False  # 是否启用 FSGS 伪标签生成
        self.fsgs_version = "improved"  # FSGS 版本: original, improved, adaptive
        self.fsgs_noise_std = 0.05  # 伪视角添加的噪声标准差
        self.fsgs_proximity_threshold = 8.0  # proximity 阈值
        self.fsgs_depth_model = "dpt_large"  # 深度估计模型
        self.fsgs_depth_weight = 0.05  # 深度损失权重
        self.fsgs_start_iter = 2000  # 开始应用 FSGS 的迭代次数
        self.enable_fsgs_proximity = False  # 是否启用 proximity 约束
        self.proximity_threshold = 6.0  # proximity 阈值
        self.enable_medical_constraints = False  # 是否启用医学约束
        self.proximity_organ_type = "foot"  # 器官类型
        self.proximity_k_neighbors = 3  # k 近邻数量
        self.enable_fsgs_depth = True  # 是否启用深度约束
        self.enable_fsgs_pseudo_views = True  # 是否生成伪视角
        self.num_fsgs_pseudo_views = 10  # 伪视角数量

        # Graph Laplacian Regularization 参数 (SSS/GGS)
        self.enable_graph_laplacian = False  # 是否启用图拉普拉斯正则化
        self.graph_k = 6  # k-NN 图的邻居数量
        self.graph_lambda_lap = 0.0008  # 拉普拉斯正则化权重
        self.graph_update_interval = 100  # 图重建间隔 (iterations)

        # 🎯 DropGaussian 参数 (CVPR 2025)
        # 严格对齐官方实现: https://github.com/DCVL-3D/DropGaussian_release
        self.use_drop_gaussian = False  # 是否启用 DropGaussian 稀疏视角正则化
        self.drop_gamma = 0.2  # 最大 drop rate（官方默认 0.2）
        self.drop_full_iter = 10000  # 达到最大 drop rate 的迭代次数（官方默认 10000）

        # 🎯 DropGaussian 改进版参数（视角感知 + 分阶段）
        self.drop_view_aware = False  # 是否启用视角感知自适应 drop rate
        self.drop_warmup_iter = 5000  # warmup 阶段不 drop（分阶段策略）
        self.drop_dist_scale = 0.6  # 距离衰减强度，越大远视角 drop 越少
        self.drop_min_factor = 0.2  # 最小 drop rate 因子（远离训练视角时）
        self.num_train_views = 3  # 训练视角数量（用于计算距离）

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
        self.densify_grad_threshold = 5.0e-5
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
        self.depth_pseudo_weight = 0
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        
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
