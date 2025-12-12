#
# Lineformer network implementation
# Copied from SAX-NeRF-master/src/network/Lineformer.py
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """截断正态分布初始化（无梯度）"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态分布初始化"""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    """预归一化模块"""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        """
        前向传播

        Args:
            x: 输入张量 [b, N, c]

        Returns:
            归一化后通过 fn 的输出
        """
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    """GELU 激活函数"""

    def forward(self, x):
        return F.gelu(x)


def ray_partition(x, line_size):
    """
    将点的 batch 转换为线段的 batch

    Args:
        x: 点特征 [N_ray * N_samples, c]
        line_size: 每个线段的点数

    Returns:
        line_batch: 线段特征 [N_ray * N_samples // line_size, line_size, c]
    """
    n, c = x.shape
    line_batch = x.view(n // line_size, line_size, c)
    return line_batch


def ray_merge(x):
    """
    将线段的 batch 转换为点的 batch

    Args:
        x: 线段特征 [N_ray * N_samples // line_size, line_size, c]

    Returns:
        point_batch: 点特征 [N_ray * N_samples, c]
    """
    line_batch_num, line_size, c = x.shape
    point_batch = x.view(line_batch_num * line_size, c)
    return point_batch


class LineAttention(nn.Module):
    """
    线段注意力机制

    对射线上的采样点进行线段级别的自注意力
    """

    def __init__(
        self,
        dim,
        line_size=24,
        dim_head=64,
        heads=8
    ):
        """
        Args:
            dim: 输入维度
            line_size: 线段大小（每段包含的点数）
            dim_head: 每个注意力头的维度
            heads: 注意力头数
        """
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.line_size = line_size

        # 位置编码
        seq_l = line_size
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [N_ray * N_samples, c]

        Returns:
            out: 输出特征 [N_ray * N_samples, c]
        """
        n, c = x.shape
        l_size = self.line_size

        # 将点 batch 转换为线段 batch
        # [N_ray * N_samples, c] -> [N_ray * N_samples // line_size, line_size, c]
        x_inp = ray_partition(x, line_size=l_size)

        # 生成 query, key, value
        q = self.to_q(x_inp)  # [b, n, inner_dim]
        k, v = self.to_kv(x_inp).chunk(2, dim=-1)  # 2 * [b, n, inner_dim]

        # 分离注意力头
        # [b, n, inner_dim] -> [b, n, heads, dim_head] -> [b, heads, n, dim_head]
        q, k, v = map(
            lambda t: t.contiguous().view(
                t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads
            ).permute(0, 2, 1, 3),
            (q, k, v)
        )

        # 缩放
        q *= self.scale

        # 注意力计算: Q x K^T
        # [b, heads, n, dim_head] x [b, heads, dim_head, n] -> [b, heads, n, n]
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)

        # 聚合: attn x V
        # [b, heads, n, n] x [b, heads, n, dim_head] -> [b, heads, n, dim_head]
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并注意力头
        # [b, heads, n, dim_head] -> [b, n, heads, dim_head] -> [b, n, inner_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)

        # 将线段 batch 转换回点 batch
        out = ray_merge(out)

        return out


class FFN(nn.Module):
    """前馈网络"""

    def __init__(self, dim, mult=4):
        """
        Args:
            dim: 输入/输出维度
            mult: 隐藏层维度倍数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult, bias=False),
            GELU(),
            nn.Linear(dim * mult, dim * mult, bias=False),
            GELU(),
            nn.Linear(dim * mult, dim, bias=False),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [N_ray * N_samples, c]

        Returns:
            out: 输出特征 [N_ray * N_samples, c]
        """
        return self.net(x)


class Line_Attention_Blcok(nn.Module):
    """
    线段注意力块

    包含多个 (PreNorm + LineAttention) 和 (PreNorm + FFN) 的组合
    """

    def __init__(
        self,
        dim,
        line_size=24,
        dim_head=32,
        heads=8,
        num_blocks=1
    ):
        """
        Args:
            dim: 特征维度
            line_size: 线段大小
            dim_head: 每个注意力头的维度
            heads: 注意力头数
            num_blocks: 块数量
        """
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LineAttention(
                    dim=dim,
                    line_size=line_size,
                    dim_head=dim_head,
                    heads=heads
                )),
                PreNorm(dim, FFN(dim=dim))
            ]))

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入特征 [N_ray * N_samples, c]

        Returns:
            out: 输出特征 [N_ray * N_samples, c]
        """
        for (attn, ff) in self.blocks:
            x = attn(x) + x  # 残差连接
            x = ff(x) + x    # 残差连接
        return x


class Lineformer_no_encoder(nn.Module):
    """
    Lineformer 网络（无编码器版本）

    假设输入已经过编码
    """

    def __init__(
        self,
        bound=0.2,
        num_layers=8,
        hidden_dim=256,
        skips=[4],
        out_dim=1,
        last_activation="sigmoid",
        line_size=32,
        dim_head=32,
        heads=8,
        num_blocks=1
    ):
        """
        Args:
            bound: 场景边界
            num_layers: 网络层数
            hidden_dim: 隐藏层维度
            skips: skip connection 层索引
            out_dim: 输出维度
            last_activation: 最后一层激活函数
            line_size: 线段大小
            dim_head: 注意力头维度
            heads: 注意力头数
            num_blocks: 注意力块数
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.bound = bound
        self.in_dim = 32  # 假设输入维度为 32

        # 构建网络层
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.in_dim, hidden_dim))

        for i in range(1, num_layers - 1):
            if i not in skips:
                self.layers.append(Line_Attention_Blcok(
                    dim=hidden_dim,
                    line_size=line_size,
                    dim_head=dim_head,
                    heads=heads,
                    num_blocks=num_blocks
                ))
            else:
                self.layers.append(nn.Linear(hidden_dim + self.in_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # 激活函数
        self.activations = nn.ModuleList()
        for i in range(num_layers - 1):
            self.activations.append(nn.LeakyReLU())

        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        """
        前向传播

        Args:
            x: 编码后的输入 [N_rays * N_samples, in_dim]

        Returns:
            out: 密度输出 [N_rays * N_samples, out_dim]
        """
        input_pts = x

        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = layer(x)
            x = activation(x)

        return x


class Lineformer(nn.Module):
    """
    Lineformer 网络（带编码器版本）

    SAX-NeRF 的核心网络，使用线段注意力机制
    """

    def __init__(
        self,
        encoder,
        bound=0.2,
        num_layers=8,
        hidden_dim=256,
        skips=[4],
        out_dim=1,
        last_activation="sigmoid",
        line_size=16,
        dim_head=32,
        heads=8,
        num_blocks=1
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
            line_size: 线段大小
            dim_head: 注意力头维度
            heads: 注意力头数
            num_blocks: 注意力块数
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.bound = bound
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        # 构建网络层
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.in_dim, hidden_dim))

        for i in range(1, num_layers - 1):
            if i not in skips:
                self.layers.append(Line_Attention_Blcok(
                    dim=hidden_dim,
                    line_size=line_size,
                    dim_head=dim_head,
                    heads=heads,
                    num_blocks=num_blocks
                ))
            else:
                self.layers.append(nn.Linear(hidden_dim + self.in_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # 激活函数
        self.activations = nn.ModuleList()
        for i in range(num_layers - 1):
            self.activations.append(nn.LeakyReLU())

        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入坐标 [N_rays * N_samples, 3]

        Returns:
            out: 密度输出 [N_rays * N_samples, out_dim]
        """
        # 编码
        x = self.encoder(x, self.bound)

        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = layer(x)
            x = activation(x)

        return x
