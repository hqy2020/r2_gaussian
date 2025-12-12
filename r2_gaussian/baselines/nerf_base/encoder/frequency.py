#
# Frequency encoder for NeRF-based methods
#
# Positional encoding using Fourier features
#

import torch
import torch.nn as nn


class FreqEncoder(nn.Module):
    """
    频率编码器 (Positional Encoding)

    将 3D 坐标编码为高维特征，使用 sin/cos 周期函数
    """

    def __init__(
        self,
        input_dim: int = 3,
        max_freq_log2: int = 5,
        N_freqs: int = 6,
        log_sampling: bool = True,
        include_input: bool = True,
    ):
        """
        Args:
            input_dim: 输入维度 (默认 3 for xyz)
            max_freq_log2: 最大频率 log2 值
            N_freqs: 频率数量
            log_sampling: 是否使用对数采样
            include_input: 是否包含原始输入
        """
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = (torch.sin, torch.cos)

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.tolist()

    def forward(self, x: torch.Tensor, bound: float = 1.0) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入坐标 [N, input_dim]
            bound: 场景边界 (未使用，与 hashgrid 兼容)

        Returns:
            encoded: 编码后的特征 [N, output_dim]
        """
        out = []

        if self.include_input:
            out.append(x)

        for freq in self.freq_bands:
            for p_fn in self.periodic_fns:
                out.append(p_fn(x * freq))

        return torch.cat(out, dim=-1)
