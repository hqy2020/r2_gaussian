#
# X-Gaussian model adapted for r2_gaussian framework
#
# 基于 X-Gaussian 的 GaussianModel_Xray，适配到 r2_gaussian 框架
#

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    inverse_softplus,
    get_expon_lr_func,
    build_rotation,
    strip_symmetric,
    build_scaling_rotation,
)
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.utils.system_utils import mkdir_p
from simple_knn._C import distCUDA2

from ..registry import GaussianBaseModel


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    """RGB 转球谐系数 (0阶)"""
    return (rgb - 0.5) / 0.28209479177387814


class XGaussianModel(GaussianBaseModel):
    """
    X-Gaussian 模型实现

    与 R²-Gaussian 的主要差异：
    - 使用 opacity (_opacity) 而非 density (_density)
    - 使用球谐特征 (_features_dc, _features_rest) 表示颜色
    - 渲染输出 RGB 后取平均得到灰度
    """

    def setup_functions(self):
        """设置激活函数"""

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int = 3, scale_bound=None, args=None):
        """
        初始化 X-Gaussian 模型

        Args:
            sh_degree: 球谐阶数 (默认 3)
            scale_bound: 尺度边界 (与 R²-Gaussian 兼容)
            args: 额外参数
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        self.setup_functions()

    # ============ 属性访问器 ============

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        scaling = self.scaling_activation(self._scaling)
        if self.scale_bound is not None:
            scaling = torch.clamp(scaling, min=self.scale_bound[0], max=self.scale_bound[1])
        return scaling

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation_activation(self._rotation)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        """获取球谐特征"""
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_density(self) -> torch.Tensor:
        """兼容 R²-Gaussian 接口，返回 opacity"""
        return self.get_opacity

    def get_covariance(self, scaling_modifier=1.0) -> torch.Tensor:
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def get_num_points(self) -> int:
        return self._xyz.shape[0]

    # ============ 初始化方法 ============

    def create_from_pcd(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                        spatial_lr_scale: float = 1.0):
        """
        从点云创建高斯

        Args:
            points: [N, 3] 点云坐标
            colors: [N, 3] 颜色 (可选，默认灰色)
            spatial_lr_scale: 空间学习率缩放
        """
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(points).float().cuda()
        num_points = fused_point_cloud.shape[0]

        # 颜色处理
        if colors is None:
            # 默认灰色
            colors = np.ones((num_points, 3)) * 0.5
        fused_color = RGB2SH(torch.tensor(colors).float().cuda())

        # 球谐特征
        features = torch.zeros(
            (num_points, 3, (self.max_sh_degree + 1) ** 2)
        ).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"X-Gaussian: Number of points at initialization: {num_points}")

        # 基于 KNN 距离估计尺度
        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud), 0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # 初始旋转 (单位四元数)
        rots = torch.zeros((num_points, 4), device="cuda")
        rots[:, 0] = 1

        # 初始 opacity
        opacities = inverse_sigmoid(
            0.1 * torch.ones((num_points, 1), dtype=torch.float, device="cuda")
        )

        # 设置参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((num_points,), device="cuda")

    def create_from_r2_init(self, init_path: str, spatial_lr_scale: float = 1.0):
        """
        从 R²-Gaussian 初始化文件创建高斯

        Args:
            init_path: 初始化文件路径 (.npy 或 .ply)
            spatial_lr_scale: 空间学习率缩放
        """
        if init_path.endswith('.npy'):
            data = np.load(init_path)
            points = data[:, :3]
            # R²-Gaussian init 文件可能包含密度信息
            densities = data[:, 3] if data.shape[1] > 3 else None
            # 将密度映射为灰度颜色
            if densities is not None:
                densities_norm = (densities - densities.min()) / (densities.max() - densities.min() + 1e-8)
                colors = np.stack([densities_norm] * 3, axis=1)
            else:
                colors = None
            self.create_from_pcd(points, colors, spatial_lr_scale)
        else:
            raise ValueError(f"Unsupported init file format: {init_path}")

    # ============ 训练设置 ============

    def get_trainable_params(self) -> List[Dict]:
        """返回可训练参数组"""
        return [
            {'params': [self._xyz], 'name': 'xyz'},
            {'params': [self._features_dc], 'name': 'f_dc'},
            {'params': [self._features_rest], 'name': 'f_rest'},
            {'params': [self._opacity], 'name': 'opacity'},
            {'params': [self._scaling], 'name': 'scaling'},
            {'params': [self._rotation], 'name': 'rotation'},
        ]

    def training_setup(self, opt):
        """设置训练优化器"""
        self.percent_dense = getattr(opt, 'percent_dense', 0.01)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz],
             'lr': opt.position_lr_init * self.spatial_lr_scale,
             'name': 'xyz'},
            {'params': [self._features_dc], 'lr': opt.feature_lr, 'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': opt.feature_lr / 20.0, 'name': 'f_rest'},
            {'params': [self._opacity], 'lr': opt.opacity_lr, 'name': 'opacity'},
            {'params': [self._scaling], 'lr': opt.scaling_lr, 'name': 'scaling'},
            {'params': [self._rotation], 'lr': opt.rotation_lr, 'name': 'rotation'},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=opt.position_lr_init * self.spatial_lr_scale,
            lr_final=opt.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=opt.position_lr_delay_mult,
            max_steps=opt.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration: int) -> float:
        """更新学习率"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
        return 0.0

    # ============ 状态管理 ============

    def capture(self) -> Dict:
        """捕获模型状态"""
        return {
            'active_sh_degree': self.active_sh_degree,
            '_xyz': self._xyz,
            '_features_dc': self._features_dc,
            '_features_rest': self._features_rest,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            '_opacity': self._opacity,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'spatial_lr_scale': self.spatial_lr_scale,
        }

    def restore(self, state: Dict, opt):
        """恢复模型状态"""
        self.active_sh_degree = state['active_sh_degree']
        self._xyz = state['_xyz']
        self._features_dc = state['_features_dc']
        self._features_rest = state['_features_rest']
        self._scaling = state['_scaling']
        self._rotation = state['_rotation']
        self._opacity = state['_opacity']
        self.max_radii2D = state['max_radii2D']
        self.spatial_lr_scale = state['spatial_lr_scale']

        self.training_setup(opt)
        self.xyz_gradient_accum = state['xyz_gradient_accum']
        self.denom = state['denom']
        if state['optimizer_state']:
            self.optimizer.load_state_dict(state['optimizer_state'])

    def save_ply(self, path):
        """保存为与 R2-Gaussian 兼容的 pickle 点云"""
        mkdir_p(os.path.dirname(path))

        with torch.no_grad():
            density = self.get_density
            density = torch.clamp(density, min=1e-6)
            density_raw = inverse_softplus(density)

            if self._rotation.numel() == 0:
                normals = self._rotation.new_empty((0, 3))
            else:
                scales = self.get_scaling
                min_axis = torch.argmin(scales, dim=1)
                R = build_rotation(self._rotation)
                idx = torch.arange(scales.shape[0], device=scales.device)
                normals = R[idx, :, min_axis]

            out = {
                "xyz": t2a(self._xyz),
                "normals": t2a(normals),
                "density": t2a(density_raw),
                "scale": t2a(self._scaling),
                "rotation": t2a(self._rotation),
                "scale_bound": self.scale_bound,
            }

        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    # ============ 球谐升级 ============

    def oneupSHdegree(self):
        """升级球谐阶数"""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ============ 密集化相关 ============

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """添加密度化统计信息"""
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def replace_tensor_to_optimizer(self, tensor, name):
        """替换优化器中的张量"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """剪枝优化器"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """剪枝点"""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """添加张量到优化器"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation):
        """密集化后处理"""
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0],), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """分裂密集化"""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points,), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
                  self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation
        )

        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """克隆密集化"""
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """密集化和剪枝"""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def reset_opacity(self):
        """重置 opacity"""
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
