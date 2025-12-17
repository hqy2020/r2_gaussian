#
# CoR-GS: Co-Regularization Gaussian Splatting
# ECCV 2024 paper: https://arxiv.org/pdf/2405.12110
#

from .config import CoRGSConfig
from .trainer import training_corgs

__all__ = ['CoRGSConfig', 'training_corgs']
