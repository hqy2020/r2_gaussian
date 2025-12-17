#
# DNGaussian Neural Renderer (GridRenderer)
#
# 使用 Hash Grid 编码 + MLP 预测 opacity 调制值
# 复用项目已有的 HashEncoder
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from r2_gaussian.baselines.nerf_base.encoder.hashgrid import HashEncoder


class MLP(nn.Module):
    """简单的 MLP 网络"""

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=False):
        """
        Args:
            dim_in: 输入维度
            dim_out: 输出维度
            dim_hidden: 隐藏层维度
            num_layers: 层数
            bias: 是否使用偏置
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            in_dim = dim_in if l == 0 else dim_hidden
            out_dim = dim_out if l == num_layers - 1 else dim_hidden
            net.append(nn.Linear(in_dim, out_dim, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class GridRenderer(nn.Module):
    """
    DNGaussian 的 Neural Renderer

    使用 Hash Grid 编码位置，通过 MLP 预测 sigma（opacity 调制值）

    适配 CT 场景：
    - 坐标范围 [-1, 1]³（R²-Gaussian 归一化）
    - 仅预测 sigma，不预测颜色
    """

    def __init__(
        self,
        bound=1.0,
        coord_center=None,
        num_levels=16,
        level_dim=2,
        base_resolution=16,
        log2_hashmap_size=19,
        hidden_dim=64,
        num_layers=3,
        geo_feat_dim=64,
    ):
        """
        Args:
            bound: 坐标范围 [-bound, bound]
            coord_center: 坐标中心偏移
            num_levels: Hash Grid 级数
            level_dim: 每级特征维度
            base_resolution: 基础分辨率
            log2_hashmap_size: 哈希表大小
            hidden_dim: MLP 隐藏层维度
            num_layers: MLP 层数
            geo_feat_dim: 几何特征维度
        """
        super().__init__()

        self.bound = bound
        if coord_center is None:
            coord_center = [0.0, 0.0, 0.0]
        self.register_buffer(
            'coord_center',
            torch.tensor(coord_center, dtype=torch.float32)
        )

        # Hash Grid 编码器（复用项目已有实现）
        self.encoder = HashEncoder(
            input_dim=3,
            num_levels=num_levels,
            level_dim=level_dim,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
            bound=bound,
        )
        self.in_dim = self.encoder.output_dim

        # Sigma Network (MLP)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.geo_feat_dim = geo_feat_dim
        self.sigma_net = MLP(
            dim_in=self.in_dim,
            dim_out=1 + geo_feat_dim,  # sigma + geo_feat
            dim_hidden=hidden_dim,
            num_layers=num_layers,
        )

        # 缓存（用于推理加速）
        self.keep_sigma = False
        self.sigma_results_cache = None

    def encode(self, x):
        """
        编码位置坐标

        Args:
            x: [N, 3] 位置坐标，在 [-bound, bound] 范围内

        Returns:
            [N, in_dim] 编码后的特征
        """
        # 减去中心偏移
        x_centered = x - self.coord_center
        return self.encoder(x_centered, bound=self.bound)

    def density(self, x, enc_x=None):
        """
        预测 sigma（密度/opacity 调制值）

        Args:
            x: [N, 3] 位置坐标
            enc_x: 可选，预编码的特征

        Returns:
            dict:
                - sigma: [N] sigma 值
                - geo_feat: [N, geo_feat_dim] 几何特征
        """
        if self.keep_sigma and self.sigma_results_cache is not None:
            return self.sigma_results_cache

        if enc_x is None:
            enc_x = self.encode(x)

        h = self.sigma_net(enc_x)
        sigma = h[..., 0]
        geo_feat = h[..., 1:]

        result = {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

        if self.keep_sigma:
            self.sigma_results_cache = result

        return result

    def forward(self, x):
        """
        前向传播

        Args:
            x: [N, 3] 位置坐标

        Returns:
            [N] sigma 值
        """
        result = self.density(x)
        return result['sigma']

    def get_params(self, lr_encoder=0.01, lr_network=0.001, weight_decay=0):
        """
        获取优化器参数组

        Args:
            lr_encoder: Hash Grid 编码器学习率
            lr_network: MLP 学习率
            weight_decay: 权重衰减

        Returns:
            参数组列表
        """
        params = [
            {
                'params': self.encoder.parameters(),
                'name': 'neural_encoder',
                'lr': lr_encoder,
            },
            {
                'params': self.sigma_net.parameters(),
                'name': 'neural_sigma',
                'lr': lr_network,
                'weight_decay': weight_decay,
            },
        ]
        return params

    def set_keep_sigma(self, keep):
        """设置是否缓存 sigma 结果"""
        self.keep_sigma = keep
        if not keep:
            self.sigma_results_cache = None

    def clear_cache(self):
        """清除缓存"""
        self.sigma_results_cache = None
