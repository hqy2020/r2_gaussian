#
# DNGaussian Model - 带 Neural Renderer 的 Gaussian 模型
#
# 基于 X-Gaussian 架构，添加 GridRenderer 进行 opacity 调制
#

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    strip_symmetric,
    build_scaling_rotation,
)
from simple_knn._C import distCUDA2

from ..registry import GaussianBaseModel
from .neural_renderer import GridRenderer
from .config import DNGaussianConfig


class DNGaussianModel(GaussianBaseModel):
    """
    DNGaussian 模型实现

    核心特点：
    1. 使用 GridRenderer (Hash Grid + MLP) 调制 opacity/density
    2. 支持深度正则化（通过 renderer 获取深度信息）
    3. 兼容 R²-Gaussian 的 X-ray 渲染流程
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
        # 密度激活（与 R²-Gaussian 一致）
        self.density_activation = torch.nn.functional.softplus
        self.density_inverse_activation = lambda x: x + torch.log(-torch.expm1(-x))

    def __init__(self, scale_bound=None, args=None, config: DNGaussianConfig = None):
        """
        初始化 DNGaussian 模型

        Args:
            scale_bound: 尺度边界
            args: 额外参数
            config: DNGaussian 配置
        """
        if config is None:
            config = DNGaussianConfig()
        self.config = config

        # Gaussian 参数
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._density = torch.empty(0)  # 基础密度（R²-Gaussian 风格）
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound

        # Neural Renderer
        self.neural_renderer = None
        self._neural_renderer_initialized = False

        # 调制参数
        self.modulation_mode = config.modulation_mode
        self.modulation_strength = config.modulation_strength

        self.setup_functions()

    def _init_neural_renderer(self):
        """延迟初始化 Neural Renderer"""
        if self._neural_renderer_initialized:
            return

        config = self.config
        self.neural_renderer = GridRenderer(
            bound=1.0,  # R²-Gaussian 坐标范围
            num_levels=config.num_levels,
            level_dim=config.level_dim,
            base_resolution=config.base_resolution,
            log2_hashmap_size=config.log2_hashmap_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            geo_feat_dim=config.geo_feat_dim,
        ).cuda()
        self._neural_renderer_initialized = True

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
    def get_base_density(self) -> torch.Tensor:
        """获取基础密度（未经 neural renderer 调制）"""
        return self.density_activation(self._density)

    @property
    def get_density(self) -> torch.Tensor:
        """
        获取调制后的密度

        使用 Neural Renderer 预测的 sigma 调制基础密度
        """
        base_density = self.get_base_density

        if self.neural_renderer is None or not self._neural_renderer_initialized:
            return base_density

        # 获取 neural renderer 的调制值
        with torch.set_grad_enabled(self.training if hasattr(self, 'training') else True):
            sigma = self.neural_renderer(self._xyz)
            # sigma 可以是正负值，通过 sigmoid 映射到 [0, 1] 作为调制因子
            modulation = torch.sigmoid(sigma).view(-1, 1)

        # 应用调制
        if self.modulation_mode == 'multiplicative':
            # 乘法调制: density * (1 + strength * (modulation - 0.5))
            # modulation=0.5 时不变，>0.5 增强，<0.5 减弱
            factor = 1.0 + self.modulation_strength * (modulation - 0.5) * 2.0
            return base_density * factor
        elif self.modulation_mode == 'additive':
            # 加法调制
            offset = self.modulation_strength * (modulation - 0.5)
            return base_density + offset
        else:
            return base_density

    @property
    def get_opacity(self) -> torch.Tensor:
        """兼容接口，返回密度"""
        return self.get_density

    def get_covariance(self, scaling_modifier=1.0) -> torch.Tensor:
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def get_num_points(self) -> int:
        return self._xyz.shape[0]

    # ============ 初始化方法 ============

    def create_from_pcd(self, points: np.ndarray, densities: Optional[np.ndarray] = None,
                        spatial_lr_scale: float = 1.0):
        """
        从点云创建高斯

        Args:
            points: [N, 3] 点云坐标
            densities: [N] 密度值 (可选)
            spatial_lr_scale: 空间学习率缩放
        """
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(points).float().cuda()
        num_points = fused_point_cloud.shape[0]

        print(f"DNGaussian: Number of points at initialization: {num_points}")

        # 基于 KNN 距离估计尺度
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # 初始旋转 (单位四元数)
        rots = torch.zeros((num_points, 4), device="cuda")
        rots[:, 0] = 1

        # 初始密度
        if densities is not None:
            # 使用提供的密度值
            densities_tensor = torch.tensor(densities).float().cuda()
            densities_tensor = torch.clamp(densities_tensor, min=1e-6)
            init_densities = self.density_inverse_activation(densities_tensor.view(-1, 1))
        else:
            # 默认密度
            init_densities = self.density_inverse_activation(
                0.1 * torch.ones((num_points, 1), dtype=torch.float, device="cuda")
            )

        # 设置参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(init_densities.requires_grad_(True))
        self.max_radii2D = torch.zeros((num_points,), device="cuda")

        # 初始化 Neural Renderer
        self._init_neural_renderer()

    def create_from_r2_init(self, init_path: str, spatial_lr_scale: float = 1.0):
        """
        从 R²-Gaussian 初始化文件创建高斯

        Args:
            init_path: 初始化文件路径 (.npy)
            spatial_lr_scale: 空间学习率缩放
        """
        if init_path.endswith('.npy'):
            data = np.load(init_path)
            points = data[:, :3]
            densities = data[:, 3] if data.shape[1] > 3 else None
            self.create_from_pcd(points, densities, spatial_lr_scale)
        else:
            raise ValueError(f"Unsupported init file format: {init_path}")

    # ============ 训练设置 ============

    def get_trainable_params(self) -> List[Dict]:
        """返回可训练参数组"""
        params = [
            {'params': [self._xyz], 'name': 'xyz'},
            {'params': [self._density], 'name': 'density'},
            {'params': [self._scaling], 'name': 'scaling'},
            {'params': [self._rotation], 'name': 'rotation'},
        ]
        return params

    def training_setup(self, opt):
        """设置训练优化器"""
        self.percent_dense = getattr(opt, 'percent_dense', self.config.percent_dense)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Gaussian 参数
        l = [
            {'params': [self._xyz],
             'lr': getattr(opt, 'position_lr_init', self.config.position_lr_init) * self.spatial_lr_scale,
             'name': 'xyz'},
            {'params': [self._density],
             'lr': getattr(opt, 'opacity_lr', self.config.opacity_lr),
             'name': 'density'},
            {'params': [self._scaling],
             'lr': getattr(opt, 'scaling_lr', self.config.scaling_lr),
             'name': 'scaling'},
            {'params': [self._rotation],
             'lr': getattr(opt, 'rotation_lr', self.config.rotation_lr),
             'name': 'rotation'},
        ]

        # Neural Renderer 参数
        if self.neural_renderer is not None:
            l.extend(self.neural_renderer.get_params(
                lr_encoder=getattr(opt, 'neural_encoder_lr', self.config.neural_encoder_lr),
                lr_network=getattr(opt, 'neural_network_lr', self.config.neural_network_lr),
            ))

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=getattr(opt, 'position_lr_init', self.config.position_lr_init) * self.spatial_lr_scale,
            lr_final=getattr(opt, 'position_lr_final', self.config.position_lr_final) * self.spatial_lr_scale,
            lr_delay_mult=getattr(opt, 'position_lr_delay_mult', self.config.position_lr_delay_mult),
            max_steps=getattr(opt, 'position_lr_max_steps', self.config.position_lr_max_steps),
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
        state = {
            '_xyz': self._xyz,
            '_scaling': self._scaling,
            '_rotation': self._rotation,
            '_density': self._density,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'spatial_lr_scale': self.spatial_lr_scale,
        }
        # Neural Renderer 状态
        if self.neural_renderer is not None:
            state['neural_renderer_state'] = self.neural_renderer.state_dict()
        return state

    def restore(self, state: Dict, opt):
        """恢复模型状态"""
        self._xyz = state['_xyz']
        self._scaling = state['_scaling']
        self._rotation = state['_rotation']
        self._density = state['_density']
        self.max_radii2D = state['max_radii2D']
        self.spatial_lr_scale = state['spatial_lr_scale']

        # 恢复 Neural Renderer
        if 'neural_renderer_state' in state:
            self._init_neural_renderer()
            self.neural_renderer.load_state_dict(state['neural_renderer_state'])

        self.training_setup(opt)
        self.xyz_gradient_accum = state['xyz_gradient_accum']
        self.denom = state['denom']
        if state['optimizer_state']:
            self.optimizer.load_state_dict(state['optimizer_state'])

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
            # 跳过 neural renderer 参数
            if group["name"] in ["neural_encoder", "neural_sigma"]:
                continue

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
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """添加张量到优化器"""
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 跳过 neural renderer 参数
            if group["name"] not in tensors_dict:
                continue

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

    def densification_postfix(self, new_xyz, new_densities, new_scaling, new_rotation, new_max_radii2D):
        """密集化后处理"""
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=0)

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
        new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_max_radii2D = torch.zeros((new_xyz.shape[0],), device="cuda")

        self.densification_postfix(
            new_xyz, new_density, new_scaling, new_rotation, new_max_radii2D
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
        new_density = self._density[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = torch.zeros((new_xyz.shape[0],), device="cuda")

        self.densification_postfix(
            new_xyz, new_density, new_scaling, new_rotation, new_max_radii2D
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """密集化和剪枝"""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_density < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def reset_opacity(self):
        """重置 density"""
        densities_new = self.density_inverse_activation(
            torch.min(self.get_base_density, torch.ones_like(self.get_base_density) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]
