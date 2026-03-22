#
# Student Splatting and Scooping (SSS) utilities
# Based on "3D Student Splatting and Scooping" paper
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def student_t_3d(x, mu, sigma, nu):
    """
    3D Student's t distribution (unnormalized)
    
    Args:
        x: points [N, 3]
        mu: mean [M, 3] or [3]
        sigma: covariance matrix [M, 3, 3] or [3, 3]
        nu: degrees of freedom [M, 1] or scalar
    
    Returns:
        density values [N, M] or [N]
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if mu.dim() == 1:
        mu = mu.unsqueeze(0)
    if sigma.dim() == 2:
        sigma = sigma.unsqueeze(0)
    if nu.dim() == 0:
        nu = nu.unsqueeze(0).unsqueeze(0)
    elif nu.dim() == 1:
        nu = nu.unsqueeze(-1)
        
    # Compute mahalanobis distance
    diff = x.unsqueeze(1) - mu.unsqueeze(0)  # [N, M, 3]
    
    # Compute sigma_inv using torch.inverse (for numerical stability)
    sigma_inv = torch.inverse(sigma + 1e-6 * torch.eye(3, device=sigma.device).unsqueeze(0))
    
    # Mahalanobis distance squared
    h = torch.sum(diff.unsqueeze(-2) @ sigma_inv.unsqueeze(0) * diff.unsqueeze(-2), dim=-1)  # [N, M]
    
    # Student's t distribution (unnormalized)
    # T(x) = [1 + (1/nu) * h]^(-(nu+3)/2)
    power = -(nu.unsqueeze(0) + 3) / 2  # [1, M, 1]
    density = torch.pow(1 + h.unsqueeze(-1) / nu.unsqueeze(0), power).squeeze(-1)  # [N, M]
    
    return density

def student_t_2d_projection(mu_3d, sigma_3d, nu, W, t, J):
    """
    Project 3D Student's t distribution to 2D image plane
    
    Args:
        mu_3d: 3D mean [N, 3]
        sigma_3d: 3D covariance [N, 3, 3]
        nu: degrees of freedom [N, 1]
        W: view transformation matrix
        t: translation vector
        J: jacobian matrix for projective transformation
        
    Returns:
        mu_2d: 2D projected mean [N, 2]
        sigma_2d: 2D projected covariance [N, 2, 2]
    """
    # Transform 3D points
    transformed_3d = mu_3d @ W.T + t.unsqueeze(0)
    
    # Project to 2D (perspective projection)
    z = transformed_3d[:, 2:3]
    mu_2d = transformed_3d[:, :2] / (z + 1e-6)
    
    # Transform covariance
    JW = J @ W  # Combined transformation
    sigma_2d_full = JW.unsqueeze(0) @ sigma_3d @ JW.unsqueeze(0).transpose(-1, -2)  
    sigma_2d = sigma_2d_full[:, :2, :2]  # Take 2x2 submatrix
    
    return mu_2d, sigma_2d

def student_t_2d(u, mu_2d, sigma_2d, nu):
    """
    2D Student's t distribution after projection
    
    Args:
        u: 2D pixel coordinates [H, W, 2] or [N, 2]
        mu_2d: 2D mean [M, 2]
        sigma_2d: 2D covariance [M, 2, 2] 
        nu: degrees of freedom [M, 1]
        
    Returns:
        density values [H, W, M] or [N, M]
    """
    original_shape = u.shape[:-1]
    u_flat = u.reshape(-1, 2)  # [N, 2]
    
    if mu_2d.dim() == 1:
        mu_2d = mu_2d.unsqueeze(0)
    if sigma_2d.dim() == 2:
        sigma_2d = sigma_2d.unsqueeze(0)
    if nu.dim() == 0:
        nu = nu.unsqueeze(0).unsqueeze(0)
    elif nu.dim() == 1:
        nu = nu.unsqueeze(-1)
        
    # Compute mahalanobis distance
    diff = u_flat.unsqueeze(1) - mu_2d.unsqueeze(0)  # [N, M, 2]
    
    # Inverse covariance with regularization
    sigma_inv = torch.inverse(sigma_2d + 1e-6 * torch.eye(2, device=sigma_2d.device).unsqueeze(0))
    
    # Mahalanobis distance squared
    h = torch.sum(diff.unsqueeze(-2) @ sigma_inv.unsqueeze(0) * diff.unsqueeze(-2), dim=-1)  # [N, M]
    
    # 2D Student's t distribution
    # T^2D(u) = [1 + (1/nu) * h]^(-(nu+2)/2)
    power = -(nu.unsqueeze(0) + 2) / 2  # [1, M, 1]
    density = torch.pow(1 + h.unsqueeze(-1) / nu.unsqueeze(0), power).squeeze(-1)  # [N, M]
    
    # Reshape back to original
    density = density.reshape(*original_shape, density.shape[-1])
    
    return density

def scooping_blend(densities, opacities, colors):
    """
    Blend colors using positive/negative scooping
    
    Args:
        densities: Student's t densities [N, M]  
        opacities: signed opacities in [-1, 1] [M]
        colors: colors [M, C]
        
    Returns:
        blended_color: final color [N, C]
        weights: blending weights [N, M]
    """
    # Compute alpha values (can be positive or negative)
    alpha = opacities.unsqueeze(0) * densities  # [N, M]
    
    # Compute transmittance (product of (1 - |alpha|))
    abs_alpha = torch.abs(alpha)
    transmittance = torch.cumprod(1 - abs_alpha, dim=1)  # [N, M]
    transmittance = F.pad(transmittance[:, :-1], (1, 0), value=1.0)  # Shift by 1
    
    # Final blending weights (can be negative for scooping)
    weights = alpha * transmittance  # [N, M]
    
    # Blend colors 
    blended_color = torch.sum(weights.unsqueeze(-1) * colors.unsqueeze(0), dim=1)  # [N, C]
    
    return blended_color, weights

def adaptive_component_recycling(low_opacity_mask, high_opacity_components, nu_old, nu_new, 
                                sigma_old, sigma_new, opacity_old, opacity_new, N):
    """
    Recycle low opacity components to high opacity locations
    Following the mathematical constraints in SSS paper
    
    Args:
        low_opacity_mask: mask for components to recycle [M] 
        high_opacity_components: indices of high opacity components
        nu_old, nu_new: degrees of freedom before/after recycling
        sigma_old, sigma_new: covariance matrices before/after
        opacity_old, opacity_new: opacities before/after
        N: number of components to relocate to each high opacity location
        
    Returns:
        Updated covariance matrix following Eq. 12 from SSS paper
    """
    # Ensure distribution consistency: minimize ||C_new - C_old||
    # Implementation of Eq. 12 from SSS paper
    
    # Beta function computation using log-gamma for numerical stability
    def log_beta(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    
    def beta_func(a, b):
        return torch.exp(log_beta(a, b))
    
    # Compute beta function terms
    beta_old = beta_func(torch.tensor(0.5), (nu_old + 2) / 2)
    
    # Compute sum for new configuration  
    sum_term = 0
    for i in range(1, N + 1):
        for k in range(i):
            binom_coeff = math.comb(i-1, k)
            sign = (-1) ** k
            opacity_power = opacity_new ** (k + 1) 
            beta_new = beta_func(torch.tensor(0.5), ((k + 1) * (nu_new + 3) - 1) / 2)
            sum_term += binom_coeff * sign * opacity_power * beta_new
    
    # Compute new covariance following Eq. 12
    ratio = beta_old / sum_term  
    covariance_scale = (opacity_old ** 2) * (nu_old / nu_new) * (ratio ** 2)
    sigma_new_computed = covariance_scale * sigma_old
    
    return sigma_new_computed

class SSS_Loss(nn.Module):
    """
    SSS-specific loss function combining L1, D-SSIM and regularization
    """
    def __init__(self, lambda_dssim=0.2, lambda_opacity=0.01, lambda_scale=0.01):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.lambda_opacity = lambda_opacity
        self.lambda_scale = lambda_scale
        
    def forward(self, image_pred, image_gt, opacities, scale_eigenvals):
        # L1 loss
        l1_loss = F.l1_loss(image_pred, image_gt)
        
        # D-SSIM loss (placeholder - implement according to paper)
        dssim_loss = 1.0 - F.cosine_similarity(
            image_pred.flatten(), image_gt.flatten(), dim=0
        )
        
        # Opacity regularization (L1 on absolute values)
        opacity_reg = torch.mean(torch.abs(opacities))
        
        # Scale regularization (encourage spiky components)
        scale_reg = torch.mean(torch.sqrt(scale_eigenvals))
        
        total_loss = (
            (1 - self.lambda_dssim) * l1_loss + 
            self.lambda_dssim * dssim_loss +
            self.lambda_opacity * opacity_reg +
            self.lambda_scale * scale_reg
        )
        
        return total_loss, {
            'l1': l1_loss.item(),
            'dssim': dssim_loss.item(), 
            'opacity_reg': opacity_reg.item(),
            'scale_reg': scale_reg.item()
        }