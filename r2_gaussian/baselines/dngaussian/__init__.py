#
# DNGaussian baseline module
#
# CVPR 2024: DNGaussian - Optimizing Sparse-View 3D Gaussian Radiance Fields
# with Global-Local Depth Normalization
#
# 适配 CT 重建场景：
# - 使用投影深度作为伪深度先验
# - Neural Renderer (Hash Grid + MLP) 作为 opacity 调制器
#

from .config import DNGaussianConfig, get_dngaussian_config
from .model import DNGaussianModel
from .renderer import render_dngaussian, render_dngaussian_depth, query_dngaussian
from .trainer import training_dngaussian
from .neural_renderer import GridRenderer
from .loss_utils import (
    patch_norm_mse_loss,
    patch_norm_mse_loss_global,
    compute_projection_depth_prior,
    depth_smoothness_loss,
)

__all__ = [
    # Config
    "DNGaussianConfig",
    "get_dngaussian_config",
    # Model
    "DNGaussianModel",
    # Renderer
    "render_dngaussian",
    "render_dngaussian_depth",
    "query_dngaussian",
    # Trainer
    "training_dngaussian",
    # Neural Renderer
    "GridRenderer",
    # Loss utils
    "patch_norm_mse_loss",
    "patch_norm_mse_loss_global",
    "compute_projection_depth_prior",
    "depth_smoothness_loss",
]
