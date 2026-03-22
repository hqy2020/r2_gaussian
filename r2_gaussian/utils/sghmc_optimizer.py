#
# SGHMC (Stochastic Gradient Hamiltonian Monte Carlo) Optimizer
# Implementation based on "3D Student Splatting and Scooping" paper
#

import torch
import torch.nn as nn
import math

class SGHMCOptimizer:
    """
    Stochastic Gradient Hamiltonian Monte Carlo optimizer for SSS
    
    This optimizer implements the SGHMC algorithm from the SSS paper,
    with adaptive friction and noise scheduling for better parameter exploration.
    """
    
    def __init__(self, params, lr=0.01, friction=0.1, mass=1.0, burnin_steps=1000):
        """
        Initialize SGHMC optimizer
        
        Args:
            params: parameters to optimize (specifically for xyz/mu)
            lr: learning rate (epsilon in paper)
            friction: friction coefficient (C in paper) 
            mass: mass matrix parameter (M in paper)
            burnin_steps: number of burn-in steps for exploration
        """
        self.param_groups = []
        if isinstance(params, torch.Tensor):
            params = [params]
        
        self.param_groups.append({
            'params': params,
            'lr': lr,
            'friction': friction,
            'mass': mass,
            'burnin_steps': burnin_steps
        })
        
        # Initialize momentum for each parameter
        self.state = {}
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    self.state[param] = {
                        'momentum': torch.zeros_like(param.data),
                        'step': 0
                    }
                    
        print(f"🎯 [SGHMC] Initialized with lr={lr}, friction={friction}, mass={mass}, burnin={burnin_steps}")
    
    def step(self, opacity_values=None):
        """
        Perform one SGHMC step
        
        Args:
            opacity_values: opacity values for adaptive friction/noise
        """
        for group in self.param_groups:
            lr = group['lr']
            friction = group['friction']
            mass = group['mass']
            burnin_steps = group['burnin_steps']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                state = self.state[param]
                momentum = state['momentum']
                step = state['step']
                
                # Gradient
                grad = param.grad.data
                
                # Adaptive friction and noise based on opacity (Eq. 10 from paper)
                if opacity_values is not None and opacity_values.shape[0] == param.shape[0]:
                    # Sigmoid switch: activate friction/noise for low opacity components
                    k = 100
                    t = 0.995
                    sigmoid_switch = torch.sigmoid(-k * (torch.abs(opacity_values) - t))
                    
                    # Ensure sigmoid_switch matches parameter shape
                    if sigmoid_switch.dim() == 1 and param.dim() > 1:
                        sigmoid_switch = sigmoid_switch.unsqueeze(-1)
                    
                    # Adaptive friction and noise
                    adaptive_friction = sigmoid_switch * friction
                    adaptive_noise_std = torch.sqrt(2 * lr**(3/2) * adaptive_friction)
                else:
                    # Default behavior
                    adaptive_friction = friction
                    adaptive_noise_std = math.sqrt(2 * lr**(3/2) * friction)
                
                # SGHMC update equations (Eq. 9 and 10)
                if step < burnin_steps:
                    # Burn-in phase: no friction for exploration
                    # Keep anisotropy by multiplying noise with covariance approximation
                    noise_scale = torch.std(param.data, dim=0, keepdim=True) + 1e-6
                    if isinstance(adaptive_noise_std, torch.Tensor):
                        noise = torch.randn_like(param.data) * adaptive_noise_std * noise_scale
                    else:
                        noise = torch.randn_like(param.data) * adaptive_noise_std * noise_scale
                    
                    # Update momentum (no friction term during burn-in)
                    if isinstance(adaptive_friction, torch.Tensor):
                        friction_term = lr * adaptive_friction
                    else:
                        friction_term = lr * adaptive_friction
                    momentum.mul_(1 - friction_term).add_(grad, alpha=-lr).add_(noise)
                    
                    # Update parameters
                    param.data.add_(momentum, alpha=lr)
                else:
                    # Normal phase: with friction for convergence
                    if isinstance(adaptive_noise_std, torch.Tensor):
                        noise = torch.randn_like(param.data) * adaptive_noise_std
                    else:
                        noise = torch.randn_like(param.data) * adaptive_noise_std
                    
                    # Update momentum (with friction)
                    if isinstance(adaptive_friction, torch.Tensor):
                        friction_term = lr * adaptive_friction
                    else:
                        friction_term = lr * adaptive_friction
                    momentum.mul_(1 - friction_term).add_(grad, alpha=-lr).add_(noise)
                    
                    # Update parameters
                    param.data.add_(momentum, alpha=lr)
                
                # Increment step counter
                state['step'] += 1
    
    def zero_grad(self):
        """Clear gradients for all parameters"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()
    
    def get_lr(self):
        """Get current learning rate"""
        return self.param_groups[0]['lr']
    
    def set_lr(self, lr):
        """Set learning rate for all parameter groups"""
        for group in self.param_groups:
            group['lr'] = lr
    
    def state_dict(self):
        """Return state dictionary for saving"""
        return {
            'param_groups': self.param_groups,
            'state': {id(k): v for k, v in self.state.items()}
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary"""
        self.param_groups = state_dict['param_groups']
        # Note: state loading is simplified for now
        print("🔄 [SGHMC] Loaded optimizer state")

class HybridOptimizer:
    """
    Hybrid optimizer combining SGHMC for positions and Adam for other parameters
    Following SSS paper's approach: use SGHMC for mu (position) and Adam for others
    """
    
    def __init__(self, xyz_params, other_param_groups, sghmc_config=None, adam_config=None):
        """
        Initialize hybrid optimizer
        
        Args:
            xyz_params: position parameters for SGHMC
            other_param_groups: other parameter groups for Adam
            sghmc_config: SGHMC configuration dict
            adam_config: Adam configuration dict
        """
        # Default configurations
        sghmc_config = sghmc_config or {'lr': 0.01, 'friction': 0.1, 'mass': 1.0, 'burnin_steps': 1000}
        adam_config = adam_config or {'lr': 0.001, 'eps': 1e-15}
        
        # Initialize optimizers
        self.sghmc = SGHMCOptimizer(xyz_params, **sghmc_config)
        self.adam = torch.optim.Adam(other_param_groups, **adam_config)
        
        print(f"🔥 [HybridOpt] Initialized SGHMC for positions + Adam for others")
        print(f"   SGHMC config: {sghmc_config}")
        print(f"   Adam config: {adam_config}")
    
    def step(self, opacity_values=None):
        """Perform optimization step for both optimizers"""
        # SGHMC step for positions with adaptive friction
        self.sghmc.step(opacity_values)
        # Adam step for other parameters
        self.adam.step()
    
    def zero_grad(self):
        """Clear gradients for all parameters"""
        self.sghmc.zero_grad()
        self.adam.zero_grad()
    
    def update_learning_rates(self, sghmc_lr_scheduler=None, adam_lr_schedulers=None):
        """Update learning rates using schedulers"""
        if sghmc_lr_scheduler:
            self.sghmc.set_lr(sghmc_lr_scheduler())
            
        if adam_lr_schedulers:
            for i, param_group in enumerate(self.adam.param_groups):
                if i < len(adam_lr_schedulers) and adam_lr_schedulers[i] is not None:
                    param_group['lr'] = adam_lr_schedulers[i]()
    
    def state_dict(self):
        """Return combined state dictionary"""
        return {
            'sghmc': self.sghmc.state_dict(),
            'adam': self.adam.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load combined state dictionary"""
        if 'sghmc' in state_dict:
            self.sghmc.load_state_dict(state_dict['sghmc'])
        if 'adam' in state_dict:
            self.adam.load_state_dict(state_dict['adam'])

def create_sss_optimizer(pc, training_args):
    """
    Create SSS-specific optimizer configuration
    
    Args:
        pc: GaussianModel instance
        training_args: training arguments
        
    Returns:
        HybridOptimizer instance or standard Adam optimizer
    """
    if not pc.use_student_t:
        # Standard Adam optimizer for backward compatibility
        print("📦 [R²] Using standard Adam optimizer")
        return None
    
    print("🎓 [SSS-R²] Creating SSS hybrid optimizer (SGHMC + Adam)")
    
    # Position parameters use SGHMC
    xyz_params = [pc._xyz]
    
    # Other parameters use Adam
    other_params = [
        {'params': [pc._density], 'lr': training_args.density_lr_init * pc.spatial_lr_scale, 'name': 'density'},
        {'params': [pc._scaling], 'lr': training_args.scaling_lr_init * pc.spatial_lr_scale, 'name': 'scaling'}, 
        {'params': [pc._rotation], 'lr': training_args.rotation_lr_init * pc.spatial_lr_scale, 'name': 'rotation'},
        {'params': [pc._nu], 'lr': getattr(training_args, 'nu_lr_init', 0.001) * pc.spatial_lr_scale, 'name': 'nu'},
        {'params': [pc._opacity], 'lr': getattr(training_args, 'opacity_lr_init', 0.01) * pc.spatial_lr_scale, 'name': 'opacity'},
    ]
    
    # SGHMC configuration
    sghmc_config = {
        'lr': training_args.position_lr_init * pc.spatial_lr_scale,
        'friction': getattr(training_args, 'sghmc_friction', 0.1),
        'mass': getattr(training_args, 'sghmc_mass', 1.0),
        'burnin_steps': getattr(training_args, 'sghmc_burnin_steps', 1000)
    }
    
    # Adam configuration
    adam_config = {
        'lr': 0.0,  # Individual LRs set in param groups
        'eps': 1e-15
    }
    
    return HybridOptimizer(xyz_params, other_params, sghmc_config, adam_config)