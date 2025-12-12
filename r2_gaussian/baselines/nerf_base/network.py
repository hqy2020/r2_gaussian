#
# MLP density network for NeRF-based methods
#

import torch
import torch.nn as nn
from typing import List


def get_network(net_type: str):
    """
    网络工厂函数

    Args:
        net_type: 网络类型 ('mlp', 'lineformer')

    Returns:
        网络类
    """
    if net_type == "mlp":
        return DensityNetwork
    elif net_type == "lineformer":
        from .lineformer import Lineformer
        return Lineformer
    else:
        raise NotImplementedError(f"Unknown network type: {net_type}")


class DensityNetwork(nn.Module):
    """
    MLP 密度网络

    用于 NAF/TensoRF/SAX-NeRF 等 NeRF-based 方法
    输入编码后的特征，输出密度值
    """

    def __init__(
        self,
        encoder: nn.Module,
        bound: float = 0.3,
        num_layers: int = 4,
        hidden_dim: int = 32,
        skips: List[int] = [2],
        out_dim: int = 1,
        last_activation: str = "sigmoid",
    ):
        """
        Args:
            encoder: 位置编码器
            bound: 场景边界
            num_layers: 网络层数
            hidden_dim: 隐藏层维度
            skips: skip connection 层索引
            out_dim: 输出维度
            last_activation: 最后一层激活函数
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound

        # 构建 MLP 层
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.in_dim, hidden_dim))

        for i in range(1, num_layers - 1):
            if i in skips:
                self.layers.append(nn.Linear(hidden_dim + self.in_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # 激活函数
        self.activations = nn.ModuleList()
        for i in range(num_layers - 1):
            self.activations.append(nn.LeakyReLU())

        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        elif last_activation == "softplus":
            self.activations.append(nn.Softplus())
        else:
            raise NotImplementedError(f"Unknown activation: {last_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入坐标 [N, 3] 或 [N_rays, N_samples, 3]

        Returns:
            density: 密度值 [N, out_dim] 或 [N_rays, N_samples, out_dim]
        """
        # 保存原始形状
        original_shape = x.shape[:-1]

        # 展平为 [N, 3]
        x = x.reshape(-1, x.shape[-1])

        # 编码
        x = self.encoder(x, self.bound)
        input_pts = x.clone()

        # MLP 前向传播
        for i in range(len(self.layers)):
            if i in self.skips:
                x = torch.cat([input_pts, x], dim=-1)

            x = self.layers[i](x)
            x = self.activations[i](x)

        # 恢复形状
        x = x.reshape(*original_shape, -1)

        return x
