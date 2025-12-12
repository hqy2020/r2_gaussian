#
# X-Gaussian configuration parameters
#

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class XGaussianConfig:
    """X-Gaussian 特有参数配置"""

    # 球谐参数
    sh_degree: int = 3

    # 背景设置
    white_background: bool = False
    random_background: bool = False

    # 密集化参数
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002

    # Opacity 控制
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005

    # 学习率
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000

    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # 损失权重
    lambda_dssim: float = 0.2

    # 密集化比例
    percent_dense: float = 0.01


def get_xgaussian_config() -> XGaussianConfig:
    """获取默认 X-Gaussian 配置"""
    return XGaussianConfig()
