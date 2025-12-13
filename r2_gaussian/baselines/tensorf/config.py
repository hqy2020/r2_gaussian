#
# TensoRF configuration
#

from dataclasses import dataclass


@dataclass
class TensoRFConfig:
    """TensoRF 配置"""

    # 编码器
    encoding: str = "tensorf"
    num_levels: int = 256
    density_n_comp: int = 8   # TensoRF VM 分解的密度分量数
    app_dim: int = 32         # TensoRF 输出特征维度

    # 网络
    net_type: str = "mlp"     # TensoRF 使用 MLP 网络
    num_layers: int = 4
    hidden_dim: int = 64
    skips: tuple = (2,)
    out_dim: int = 1
    last_activation: str = "sigmoid"  # X-ray CT 需要 sigmoid 将输出限制在 [0,1]
    bound: float = 1.0  # 场景归一化到 [-1, 1]^3

    # 渲染
    n_samples: int = 192
    n_fine: int = 192
    perturb: bool = True
    raw_noise_std: float = 0.0
    netchunk: int = 409600

    # 训练
    epochs: int = 1500
    n_batch: int = 1
    n_rays: int = 1024
    lrate: float = 0.001
    lrate_gamma: float = 0.1
    lrate_step: int = 1500
