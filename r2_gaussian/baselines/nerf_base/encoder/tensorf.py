#
# TensoRF encoder - VM decomposition implementation
# Copied from SAX-NeRF-master/src/encoder/tensorf_encoder.py
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorfEncoder(torch.nn.Module):
    """
    TensoRF 编码器 - Vector-Matrix (VM) 分解实现

    基于 TensoRF 论文的平面-向量分解:
    - 使用 3 个平面和 3 个向量来表示 3D 特征
    - 通过 grid_sample 进行双线性插值
    """

    def __init__(self, num_levels, density_n_comp=8, app_dim=32, device='cpu', **kwargs):
        """
        Args:
            num_levels: 网格分辨率
            density_n_comp: 密度分量数量
            app_dim: 输出特征维度
            device: 计算设备
        """
        super().__init__()

        # 平面和向量的索引配置
        # matMode: 定义每个平面对应的两个坐标轴
        # vecMode: 定义每个向量对应的坐标轴
        self.matMode = [[0, 1], [0, 2], [1, 2]]  # xy, xz, yz 平面
        self.vecMode = [2, 1, 0]  # z, y, x 向量

        self.density_n_comp = density_n_comp
        self.output_dim = app_dim
        self.app_dim = app_dim

        self.init_svd_volume(num_levels, device)

    def init_svd_volume(self, res, device):
        """
        初始化 SVD 分解体积

        Args:
            res: 网格分辨率
            device: 计算设备
        """
        # 初始化平面和线特征
        self.density_plane, self.density_line = self.init_one_svd(
            [self.density_n_comp] * 3,
            [res] * 3,
            0.1,
            device
        )

        # 基矩阵: 将分解特征转换为输出维度
        self.basis_mat = torch.nn.Linear(
            self.density_n_comp * 3,
            self.app_dim,
            bias=False
        ).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        """
        初始化一组 SVD 分解参数

        Args:
            n_component: 每个分量的数量列表 [3]
            gridSize: 网格尺寸列表 [3]
            scale: 初始化缩放因子
            device: 计算设备

        Returns:
            plane_coef: 平面系数 ParameterList
            line_coef: 线系数 ParameterList
        """
        plane_coef, line_coef = [], []

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            # 平面系数: [1, n_comp, H, W]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))
            ))

            # 线系数: [1, n_comp, L, 1]
            line_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))
            ))

        return (
            torch.nn.ParameterList(plane_coef).to(device),
            torch.nn.ParameterList(line_coef).to(device)
        )

    def forward(self, inputs, size=1):
        """
        前向传播

        Args:
            inputs: 输入坐标 [N, 3] 或 [N_rays, N_samples, 3]
            size: 坐标范围 (inputs 应在 [-size, size] 内)

        Returns:
            outputs: 编码后的特征 [N, app_dim]
        """
        # 保存原始形状
        original_shape = inputs.shape[:-1]
        inputs = inputs.reshape(-1, 3)

        # 坐标范围检查和归一化
        # assert not (inputs.min().item() < -size or inputs.max().item() > size)
        inputs = inputs / size  # 映射到 [-1, 1]

        outputs = self.compute_densityfeature(inputs)

        # 恢复原始形状
        outputs = outputs.reshape(*original_shape, -1)

        return outputs

    def compute_densityfeature(self, xyz_sampled):
        """
        计算密度特征

        使用 VM 分解: 平面特征 * 线特征

        Args:
            xyz_sampled: 采样点坐标 [N, 3], 范围 [-1, 1]

        Returns:
            features: 密度特征 [N, app_dim]
        """
        # 构建平面坐标: 取 xyz 中对应平面的两个坐标
        # shape: [3, N, 1, 2]
        coordinate_plane = torch.stack((
            xyz_sampled[..., self.matMode[0]],  # xy
            xyz_sampled[..., self.matMode[1]],  # xz
            xyz_sampled[..., self.matMode[2]]   # yz
        )).detach().view(3, -1, 1, 2)

        # 构建线坐标: 取 xyz 中对应的单个坐标
        # shape: [3, N, 1, 2] (第二维填 0，用于 grid_sample)
        coordinate_line = torch.stack((
            xyz_sampled[..., self.vecMode[0]],  # z
            xyz_sampled[..., self.vecMode[1]],  # y
            xyz_sampled[..., self.vecMode[2]]   # x
        ))
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line),
            dim=-1
        ).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []

        for idx_plane in range(len(self.density_plane)):
            # 从平面采样: grid_sample 输入 [N, C, H, W], grid [N, H_out, W_out, 2]
            # 输出: [N, C, H_out, W_out]
            plane_coef_point.append(
                F.grid_sample(
                    self.density_plane[idx_plane],
                    coordinate_plane[[idx_plane]],
                    align_corners=True
                ).view(-1, *xyz_sampled.shape[:1])
            )

            # 从线采样
            line_coef_point.append(
                F.grid_sample(
                    self.density_line[idx_plane],
                    coordinate_line[[idx_plane]],
                    align_corners=True
                ).view(-1, *xyz_sampled.shape[:1])
            )

        # 拼接所有分量: [n_comp * 3, N]
        plane_coef_point = torch.cat(plane_coef_point)
        line_coef_point = torch.cat(line_coef_point)

        # 平面 * 线，然后通过基矩阵: [N, app_dim]
        return self.basis_mat((plane_coef_point * line_coef_point).T)
