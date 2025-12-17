#
# FSGS baseline module for CT reconstruction
#
# FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting
# Paper: https://arxiv.org/abs/2312.00451
#
# CT 适配版:
# - 使用 xray-gaussian-rasterization 替代 RGB 光栅化器
# - 使用深度 TV 正则化替代 MiDAS 深度监督
# - 使用圆弧插值生成伪视角
#

from .config import FSGSConfig, get_fsgs_config
from .model import FSGSModel
from .renderer import render_fsgs, query_fsgs, FSGSRenderer
from .trainer import training_fsgs
from .pseudo_camera import CTPseudoCameraGenerator, CTPseudoCamera, tv_loss

__all__ = [
    # Config
    "FSGSConfig",
    "get_fsgs_config",
    # Model
    "FSGSModel",
    # Renderer
    "render_fsgs",
    "query_fsgs",
    "FSGSRenderer",
    # Trainer
    "training_fsgs",
    # Pseudo Camera
    "CTPseudoCameraGenerator",
    "CTPseudoCamera",
    "tv_loss",
]
