"""
K-Planes 空间分解编码器

将 3D 空间分解为 3 个正交平面特征网格，用于增强 3D Gaussian Splatting 的几何约束。
改编自 X²-Gaussian 论文，仅保留空间维度（去除时间维度）。

作者：Claude Code Agent
日期：2025-01-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class KPlanesEncoder(nn.Module):
    """
    K-Planes 空间分解编码器（仅空间维度，无时间维度）

    将 3D 空间 (x,y,z) 分解为 3 个正交平面特征网格：
    - plane_xy: 特征平面 [1, resolution, resolution, feature_dim]
    - plane_xz: 特征平面 [1, resolution, resolution, feature_dim]
    - plane_yz: 特征平面 [1, resolution, resolution, feature_dim]

    每个平面使用双线性插值提取特征，最终拼接为 [N, feature_dim * 3] 的特征向量。

    参数：
        grid_resolution (int): 单平面分辨率（默认 64）
        feature_dim (int): 特征维度（默认 32）
        num_levels (int): 多分辨率层数（默认 1，暂不支持多分辨率）
        bounds (tuple): 空间边界（默认 (-1.0, 1.0)，匹配 R²-GS 归一化）
    """

    def __init__(
        self,
        grid_resolution: int = 64,
        feature_dim: int = 32,
        num_levels: int = 1,
        bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()

        self.grid_resolution = grid_resolution
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        self.bounds = bounds

        # 初始化 3 个空间平面特征网格
        # 形状：[1, feature_dim, resolution, resolution]
        # 使用 Xavier 均匀初始化（小的随机值，避免初始特征过大）
        self.plane_xy = nn.Parameter(
            torch.empty(1, feature_dim, grid_resolution, grid_resolution)
        )
        self.plane_xz = nn.Parameter(
            torch.empty(1, feature_dim, grid_resolution, grid_resolution)
        )
        self.plane_yz = nn.Parameter(
            torch.empty(1, feature_dim, grid_resolution, grid_resolution)
        )

        # 初始化参数
        nn.init.xavier_uniform_(self.plane_xy)
        nn.init.xavier_uniform_(self.plane_xz)
        nn.init.xavier_uniform_(self.plane_yz)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        提取指定位置的 K-Planes 特征

        参数：
            xyz (torch.Tensor): 高斯中心坐标，形状 [N, 3]，范围在 self.bounds 内

        返回：
            features (torch.Tensor): 拼接后的特征，形状 [N, feature_dim * 3]

        实现步骤：
            1. 将 xyz 归一化到 [-1, 1]（grid_sample 要求）
            2. 对每个平面执行双线性插值
            3. 拼接 3 个平面的特征
        """
        N = xyz.shape[0]

        # 归一化坐标到 [-1, 1]（grid_sample 的输入要求）
        # 假设输入 xyz 已经在 self.bounds 范围内
        xyz_normalized = self._normalize_coords(xyz)

        # 提取 x, y, z 坐标
        x = xyz_normalized[:, 0]  # [N]
        y = xyz_normalized[:, 1]  # [N]
        z = xyz_normalized[:, 2]  # [N]

        # 从 3 个平面提取特征
        # grid_sample 需要输入形状：[N, H, W, 2]
        # 输出形状：[1, feature_dim, N, 1] -> [N, feature_dim]

        # Plane XY：使用 (x, y) 坐标
        grid_xy = torch.stack([x, y], dim=-1).view(1, N, 1, 2)  # [1, N, 1, 2]
        feat_xy = F.grid_sample(
            self.plane_xy,  # [1, feature_dim, resolution, resolution]
            grid_xy,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'  # 边界外的点使用边界值
        )  # [1, feature_dim, N, 1]
        feat_xy = feat_xy.squeeze(-1).squeeze(0).t()  # [N, feature_dim]

        # Plane XZ：使用 (x, z) 坐标
        grid_xz = torch.stack([x, z], dim=-1).view(1, N, 1, 2)  # [1, N, 1, 2]
        feat_xz = F.grid_sample(
            self.plane_xz,
            grid_xz,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )  # [1, feature_dim, N, 1]
        feat_xz = feat_xz.squeeze(-1).squeeze(0).t()  # [N, feature_dim]

        # Plane YZ：使用 (y, z) 坐标
        grid_yz = torch.stack([y, z], dim=-1).view(1, N, 1, 2)  # [1, N, 1, 2]
        feat_yz = F.grid_sample(
            self.plane_yz,
            grid_yz,
            align_corners=True,
            mode='bilinear',
            padding_mode='border'
        )  # [1, feature_dim, N, 1]
        feat_yz = feat_yz.squeeze(-1).squeeze(0).t()  # [N, feature_dim]

        # 拼接 3 个平面的特征
        features = torch.cat([feat_xy, feat_xz, feat_yz], dim=-1)  # [N, feature_dim * 3]

        return features

    def _normalize_coords(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        将坐标从 self.bounds 归一化到 [-1, 1]

        参数：
            xyz (torch.Tensor): 原始坐标，形状 [N, 3]

        返回：
            xyz_normalized (torch.Tensor): 归一化后的坐标，形状 [N, 3]
        """
        min_bound, max_bound = self.bounds
        # 线性映射：[min_bound, max_bound] -> [-1, 1]
        xyz_normalized = 2.0 * (xyz - min_bound) / (max_bound - min_bound) - 1.0
        # 裁剪到 [-1, 1] 范围（防止数值误差导致的越界）
        xyz_normalized = torch.clamp(xyz_normalized, -1.0, 1.0)
        return xyz_normalized

    def get_plane_params(self) -> List[nn.Parameter]:
        """
        返回所有平面参数（用于优化器配置和 TV 正则化）

        返回：
            List[nn.Parameter]: [plane_xy, plane_xz, plane_yz]
        """
        return [self.plane_xy, self.plane_xz, self.plane_yz]

    def get_output_dim(self) -> int:
        """
        返回输出特征维度

        返回：
            int: feature_dim * 3
        """
        return self.feature_dim * 3


# 测试代码（可选）
if __name__ == "__main__":
    # 简单功能测试
    print("Testing KPlanesEncoder...")

    # 创建编码器
    encoder = KPlanesEncoder(
        grid_resolution=64,
        feature_dim=32,
        num_levels=1,
        bounds=(-1.0, 1.0)
    ).cuda()

    # 测试前向传播
    xyz = torch.randn(1000, 3).cuda() * 0.5  # [-0.5, 0.5] 范围内的随机点
    features = encoder(xyz)

    print(f"Input shape: {xyz.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected output shape: (1000, 96)")

    # 验证形状
    assert features.shape == (1000, 96), f"输出形状错误：{features.shape}"

    # 测试参数提取
    params = encoder.get_plane_params()
    print(f"Number of plane parameters: {len(params)}")
    print(f"Plane XY shape: {params[0].shape}")

    # 测试边界情况
    xyz_boundary = torch.tensor([
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ]).cuda()
    features_boundary = encoder(xyz_boundary)
    print(f"Boundary test passed, output shape: {features_boundary.shape}")

    print("All tests passed!")
