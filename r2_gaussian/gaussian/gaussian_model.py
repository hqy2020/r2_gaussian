#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import torch
from torch import nn
import numpy as np
import pickle
from plyfile import PlyData, PlyElement

sys.path.append("./")

from simple_knn._C import distCUDA2
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.utils.system_utils import mkdir_p
from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)

EPS = 1e-5


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize
        
        # SSS: Student's t distribution activation functions (CONSERVATIVE)
        if self.use_student_t:
            # nu parameter: CONSERVATIVE range [2, 8] for numerical stability
            self.nu_activation = lambda x: torch.sigmoid(x) * (8 - 2) + 2
            self.nu_inverse_activation = lambda x: inverse_sigmoid((x - 2) / (8 - 2))
            # opacity: CONSERVATIVE SCOOPING - mostly positive with limited negative (5-10%)
            # Using sigmoid + offset to ensure most values are positive
            self.opacity_activation = lambda x: torch.sigmoid(x) * 1.2 - 0.1  # Range [-0.1, 1.1]
            self.opacity_inverse_activation = lambda x: inverse_sigmoid((torch.clamp(x, -0.09, 1.09) + 0.1) / 1.2)
        else:
            # Default: same as density for backward compatibility
            self.nu_activation = lambda x: torch.ones_like(x) * float('inf')  # Gaussian limit
            self.opacity_activation = lambda x: torch.sigmoid(x)  # [0,1] range

    def __init__(self, scale_bound=None, use_student_t=False):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        # SSS: Student's t distribution parameter (degrees of freedom)
        self._nu = torch.empty(0)  # degrees of freedom for t-distribution
        # SSS: opacity for scooping (negative components)
        self._opacity = torch.empty(0)  # opacity for positive/negative splatting
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        self.use_student_t = use_student_t  # SSS: flag to enable Student's t
        self.setup_functions()

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self._nu,  # SSS: Student's t degrees of freedom
            self._opacity,  # SSS: Scooping opacity
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
            self.use_student_t,  # SSS: Student's t flag
        )

    def restore(self, model_args, training_args):
        if len(model_args) == 13:  # New SSS format
            (
                self._xyz,
                self._scaling,
                self._rotation,
                self._density,
                self._nu,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.scale_bound,
                self.use_student_t,
            ) = model_args
            print(f"🎓 [SSS-R²] Loaded model with Student's t distribution: {self.use_student_t}")
        else:  # Legacy format
            (
                self._xyz,
                self._scaling,
                self._rotation,
                self._density,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self.scale_bound,
            ) = model_args
            # Initialize SSS parameters for backward compatibility
            self.use_student_t = False
            self._nu = torch.zeros_like(self._density)
            self._opacity = inverse_sigmoid(torch.ones_like(self._density) * 0.5)
            print("📦 [R²] Loaded legacy model - SSS features disabled")
            
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)
    
    @property
    def get_nu(self):
        """SSS: Get degrees of freedom for Student's t distribution"""
        if self.use_student_t:
            return self.nu_activation(self._nu)
        else:
            return torch.ones_like(self._density) * float('inf')  # Gaussian limit
    
    @property
    def get_opacity(self):
        """SSS: Get opacity for scooping (positive/negative splatting)"""
        if self.use_student_t:
            return self.opacity_activation(self._opacity)
        else:
            return torch.sigmoid(self._opacity)  # Default [0,1] range

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        n_points = fused_point_cloud.shape[0]
        
        if self.use_student_t:
            print(f"🎓 [SSS-R²] Initialize {n_points} Student's t distributions with scooping")
        else:
            print(f"📦 [R²] Initialize gaussians from {n_points} estimated points")
            
        fused_density = (
            self.density_inverse_activation(torch.tensor(density)).float().cuda()
        )
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((n_points, 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        
        # SSS: Initialize new parameters  
        if self.use_student_t:
            # ENHANCED Initialize nu with wider range for more expressiveness
            nu_vals = torch.rand(n_points, 1, device="cuda") * 4 + 2  # [2, 6] - good tail thickness range
            nu_init = self.nu_inverse_activation(nu_vals)
            self._nu = nn.Parameter(nu_init.requires_grad_(True))
            
            # ENHANCED Initialize opacity - start positive but allow training to explore
            # Use density-based initialization for better distribution
            opacity_vals = torch.sigmoid(fused_density.clone()) * 0.8 + 0.1  # [0.1, 0.9] - density-guided
            opacity_init = self.opacity_inverse_activation(torch.clamp(opacity_vals, 0.01, 0.99))
            self._opacity = nn.Parameter(opacity_init.requires_grad_(True))
            print(f"   🎓 [SSS Enhanced] Initialized nu ~ [2, 6], opacity density-guided [0.1, 0.9]")
        else:
            # Default initialization for backward compatibility
            self._nu = nn.Parameter(torch.zeros(n_points, 1, device="cuda").requires_grad_(True))
            # Use density as opacity for backward compatibility
            self._opacity = nn.Parameter(fused_density.clone().requires_grad_(True))
            
        self.max_radii2D = torch.zeros((n_points,), device="cuda")

        #! Generate one gaussian for debugging purpose
        if False:
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]
        
        # SSS: Add new parameters to optimizer
        if self.use_student_t:
            l.extend([
                {
                    "params": [self._nu],
                    "lr": getattr(training_args, 'nu_lr_init', 0.001) * self.spatial_lr_scale,
                    "name": "nu",
                },
                {
                    "params": [self._opacity],
                    "lr": getattr(training_args, 'opacity_lr_init', 0.01) * self.spatial_lr_scale,
                    "name": "opacity",
                },
            ])
            print(f"🎓 [SSS-R²] Setup optimizer with Student's t parameters (nu_lr={getattr(training_args, 'nu_lr_init', 0.001)}, opacity_lr={getattr(training_args, 'opacity_lr_init', 0.01)})")
        else:
            # Add opacity for backward compatibility 
            l.append({
                "params": [self._opacity],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "opacity",
            })

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        # We save pickle files to store more information

        mkdir_p(os.path.dirname(path))

        xyz = t2a(self._xyz)
        densities = t2a(self._density)
        scale = t2a(self._scaling)
        rotation = t2a(self._rotation)

        out = {
            "xyz": xyz,
            "density": densities,
            "scale": scale,
            "rotation": rotation,
            "scale_bound": self.scale_bound,
            "version": "SSS-R2-v1.0",  # Version control
            "use_student_t": self.use_student_t,
        }
        
        # SSS: Save new parameters
        if self.use_student_t:
            out.update({
                "nu": t2a(self._nu),
                "opacity": t2a(self._opacity),
            })
            print(f"💾 [SSS-R²] Saved model with Student's t distribution (version: SSS-R2-v1.0)")
        else:
            out["opacity"] = t2a(self._opacity)  # Save for compatibility
            print(f"💾 [R²] Saved legacy model")
            
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Check version and SSS support
        version = data.get("version", "legacy")
        self.use_student_t = data.get("use_student_t", False)
        
        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        
        # SSS: Load new parameters
        if "nu" in data and "opacity" in data and self.use_student_t:
            self._nu = nn.Parameter(
                torch.tensor(data["nu"], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._opacity = nn.Parameter(
                torch.tensor(data["opacity"], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            print(f"🔄 [SSS-R²] Loaded model with Student's t distribution (version: {version})")
        else:
            # Initialize for backward compatibility
            n_points = self._xyz.shape[0]
            self._nu = nn.Parameter(torch.zeros(n_points, 1, device="cuda").requires_grad_(True))
            if "opacity" in data:
                self._opacity = nn.Parameter(
                    torch.tensor(data["opacity"], dtype=torch.float, device="cuda").requires_grad_(True)
                )
            else:
                self._opacity = nn.Parameter(self._density.clone().requires_grad_(True))
            print(f"🔄 [R²] Loaded legacy model (version: {version})")
            
        self.scale_bound = data.get("scale_bound")
        self.setup_functions()  # Reset activation functions

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # SSS: Update SSS parameters if using Student's t
        if self.use_student_t:
            if "nu" in optimizable_tensors:
                self._nu = optimizable_tensors["nu"]
            if "opacity" in optimizable_tensors:
                self._opacity = optimizable_tensors["opacity"]
        else:
            # For non-SSS models, handle opacity for compatibility
            if "opacity" in optimizable_tensors:
                self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
        new_nu=None,
        new_opacity=None,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        
        # SSS: Add new parameters if provided
        if self.use_student_t:
            if new_nu is not None:
                d["nu"] = new_nu
            if new_opacity is not None:
                d["opacity"] = new_opacity
        else:
            # For non-SSS models, always add opacity for compatibility
            if new_opacity is not None:
                d["opacity"] = new_opacity

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # SSS: Update SSS parameters if using Student's t
        if self.use_student_t:
            if "nu" in optimizable_tensors:
                self._nu = optimizable_tensors["nu"]
            if "opacity" in optimizable_tensors:
                self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        # 在CPU上计算矩阵乘法以避免CUBLAS错误
        rots_cpu = rots.cpu()
        samples_cpu = samples.unsqueeze(-1).cpu()
        new_xyz = torch.bmm(rots_cpu, samples_cpu).squeeze(-1).cuda() + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
        
        # SSS: Handle new parameters for densification
        new_nu = None
        new_opacity = None
        if self.use_student_t:
            new_nu = self._nu[selected_pts_mask].repeat(N, 1)  # Keep same nu
            new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)  # Keep same opacity
        else:
            # For non-SSS models, use density for opacity compatibility
            new_opacity = new_density

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_nu,
            new_opacity,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities
        
        # SSS: Handle new parameters for cloning
        new_nu = None
        new_opacity = None
        if self.use_student_t:
            new_nu = self._nu[selected_pts_mask]  # Clone same nu
            new_opacity = self._opacity[selected_pts_mask]  # Clone same opacity
        else:
            # For non-SSS models, use density for opacity compatibility
            new_opacity = new_densities

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
            new_nu,
            new_opacity,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = (self.get_density < min_density).squeeze()
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def density_decay(self, factor=0.99, unique_gidx=None):
        """
        对高斯点的密度进行衰减，相当于opacity decay功能
        Args:
            factor: 衰减因子，默认0.99
            unique_gidx: 特定高斯索引（暂时未使用）
        """
        density = self.get_density
        density = density * factor
        self._density.data = self.density_inverse_activation(density)
