#
# NAF (Neural Attenuation Fields) configuration
#

from dataclasses import dataclass


@dataclass
class NAFConfig:
    """NAF 配置"""

    # 编码器
    encoding: str = "hashgrid"
    num_levels: int = 16
    level_dim: int = 2
    base_resolution: int = 16
    log2_hashmap_size: int = 19

    # 网络
    num_layers: int = 4
    hidden_dim: int = 32
    skips: tuple = (2,)
    out_dim: int = 1
    last_activation: str = "sigmoid"
    bound: float = 1.0  # 场景归一化到 [-1, 1]^3

    # 渲染
    n_samples: int = 192
    n_fine: int = 0
    perturb: bool = True
    raw_noise_std: float = 0.0
    netchunk: int = 409600

    # 训练
    epochs: int = 1500
    n_batch: int = 1
    n_rays: int = 1024
    lrate: float = 0.001
    lrate_gamma: float = 0.1
    lrate_step: int = 500
