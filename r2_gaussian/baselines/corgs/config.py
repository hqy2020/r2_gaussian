#
# CoR-GS: Co-Regularization Gaussian Splatting
# Configuration for CT sparse-view reconstruction
#

from dataclasses import dataclass


@dataclass
class CoRGSConfig:
    """CoR-GS 配置参数类"""

    # ============ 双场设置 ============
    gaussians_n: int = 2              # 固定为双场

    # ============ Co-Regularization 参数 ============
    coreg: bool = True                # 启用 Co-Reg
    start_sample_pseudo: int = 2000   # Co-Reg 起始迭代
    end_sample_pseudo: int = 15000    # Co-Reg 终止迭代
    sample_pseudo_interval: int = 1   # 每次迭代采样
    n_pseudo_views: int = 1000        # 伪视图数量

    # ============ Co-Pruning 参数 ============
    coprune: bool = True              # 启用 Co-Prune
    coprune_threshold: float = 0.05   # 点对应距离阈值（world coords，相对于体素尺寸）
    coprune_interval: int = 500       # 每 500 迭代触发

    # ============ 密集化参数（与 X-Gaussian 一致） ============
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005

    # ============ 学习率参数 ============
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01

    # ============ 损失函数 ============
    lambda_dssim: float = 0.2

    # ============ 球谐 ============
    sh_degree: int = 3
