"""
单目深度估计模块 - FSGS风格深度监督
支持DPT和MiDaS深度估计器，实现Pearson相关性深度损失

参考论文: FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting
核心功能: 单目深度估计 + 尺度不变深度损失
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import warnings

# 尝试导入深度估计库
try:
    import transformers
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    HAS_DPT = True
except ImportError:
    HAS_DPT = False
    print("Warning: transformers not available, DPT depth estimation disabled")

try:
    import midas
    from midas.model_loader import default_models, load_model
    HAS_MIDAS = True
except ImportError:
    HAS_MIDAS = False
    print("Warning: MiDaS not available, using fallback depth estimation")


class MonocularDepthEstimator:
    """
    单目深度估计器 - 支持多种预训练模型
    实现FSGS论文中的深度监督机制
    """
    
    def __init__(self, 
                 model_type: str = "dpt_large",
                 device: str = "cuda",
                 enable_depth_estimation: bool = True):
        """
        初始化深度估计器
        
        Args:
            model_type: 深度估计模型类型
                       - "dpt_large": Dense Prediction Transformer Large (推荐)
                       - "dpt_hybrid": DPT Hybrid
                       - "midas_small": MiDaS Small
                       - "midas_large": MiDaS Large
                       - "disabled": 禁用深度估计 (向下兼容)
            device: 计算设备
            enable_depth_estimation: 是否启用深度估计 (向下兼容开关)
        """
        self.model_type = model_type
        self.device = device
        self.enabled = enable_depth_estimation and model_type != "disabled"
        
        # 向下兼容检查
        if not self.enabled:
            print("📦 [Compatibility] Depth estimation disabled - using legacy pseudo-label mode")
            self.model = None
            self.processor = None
            return
            
        self.model = None
        self.processor = None
        
        # 初始化选定的模型
        if self.enabled:
            self._initialize_model()
    
    def _initialize_model(self):
        """初始化深度估计模型"""
        try:
            if self.model_type.startswith("dpt") and HAS_DPT:
                self._initialize_dpt_model()
            elif self.model_type.startswith("midas") and HAS_MIDAS:
                self._initialize_midas_model()
            else:
                # 备用方案: 禁用深度估计
                print(f"⚠️  Model type {self.model_type} not available, disabling depth estimation")
                self.enabled = False
                
        except Exception as e:
            print(f"⚠️  Failed to initialize {self.model_type}: {e}")
            print("📦 Falling back to legacy mode without depth estimation")
            self.enabled = False
    
    def _initialize_dpt_model(self):
        """初始化DPT模型"""
        if self.model_type == "dpt_large":
            model_name = "Intel/dpt-large"
        elif self.model_type == "dpt_hybrid":
            model_name = "Intel/dpt-hybrid-midas"
        else:
            raise ValueError(f"Unknown DPT model: {self.model_type}")
        
        print(f"🎯 Loading DPT model: {model_name}")
        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("✅ DPT model loaded successfully")
    
    def _initialize_midas_model(self):
        """初始化MiDaS模型"""
        if self.model_type == "midas_small":
            model_type = "MiDaS_small"
        elif self.model_type == "midas_large":
            model_type = "MiDaS"
        else:
            raise ValueError(f"Unknown MiDaS model: {self.model_type}")
        
        print(f"🎯 Loading MiDaS model: {model_type}")
        self.model, self.processor, _ = load_model(
            device=torch.device(self.device),
            model_type=model_type,
            optimize=True
        )
        print("✅ MiDaS model loaded successfully")
    
    def estimate_depth(self, image: torch.Tensor, 
                      normalize: bool = True) -> Optional[torch.Tensor]:
        """
        估计图像深度
        
        Args:
            image: 输入图像 [C, H, W] 或 [B, C, H, W]，值域[0,1]
            normalize: 是否归一化深度值
        
        Returns:
            深度图 [H, W] 或 [B, H, W]，如果禁用则返回None
        """
        if not self.enabled or self.model is None:
            return None
        
        try:
            with torch.no_grad():
                if self.model_type.startswith("dpt"):
                    return self._estimate_depth_dpt(image, normalize)
                elif self.model_type.startswith("midas"):
                    return self._estimate_depth_midas(image, normalize)
                else:
                    return None
                    
        except Exception as e:
            warnings.warn(f"Depth estimation failed: {e}", RuntimeWarning)
            return None
    
    def _estimate_depth_dpt(self, image: torch.Tensor, normalize: bool) -> torch.Tensor:
        """DPT深度估计"""
        # 转换为PIL格式进行预处理
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, C, H, W]
        
        # 转换到CPU进行预处理
        image_np = image.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
        image_np = (image_np * 255).astype(np.uint8)
        
        batch_size = image_np.shape[0]
        depth_maps = []
        
        for i in range(batch_size):
            # 预处理单张图像
            inputs = self.processor(images=image_np[i], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 深度预测
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # 插值到原始分辨率
            original_shape = image.shape[-2:]  # [H, W]
            depth = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=original_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [1, H, W] -> [H, W]
            
            depth_maps.append(depth.squeeze(0))  # [H, W]
        
        depth_tensor = torch.stack(depth_maps, dim=0)  # [B, H, W]
        
        # 归一化
        if normalize:
            depth_tensor = self._normalize_depth(depth_tensor)
        
        return depth_tensor.squeeze(0) if batch_size == 1 else depth_tensor
    
    def _estimate_depth_midas(self, image: torch.Tensor, normalize: bool) -> torch.Tensor:
        """MiDaS深度估计"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # MiDaS预处理
        input_batch = []
        for i in range(image.shape[0]):
            img = image[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            img = (img * 255).astype(np.uint8)
            processed = self.processor(img)
            input_batch.append(torch.from_numpy(processed).to(self.device))
        
        input_tensor = torch.stack(input_batch, dim=0)
        
        # 深度预测
        depth_batch = self.model(input_tensor)
        
        # 归一化
        if normalize:
            depth_batch = self._normalize_depth(depth_batch)
        
        return depth_batch.squeeze(0) if depth_batch.shape[0] == 1 else depth_batch
    
    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """归一化深度值"""
        # 对每个样本分别归一化
        if depth.dim() == 3:  # [B, H, W]
            normalized_depth = []
            for i in range(depth.shape[0]):
                d = depth[i]
                d_min, d_max = d.min(), d.max()
                if d_max > d_min:
                    d_norm = (d - d_min) / (d_max - d_min)
                else:
                    d_norm = torch.zeros_like(d)
                normalized_depth.append(d_norm)
            return torch.stack(normalized_depth, dim=0)
        else:  # [H, W]
            d_min, d_max = depth.min(), depth.max()
            if d_max > d_min:
                return (depth - d_min) / (d_max - d_min)
            else:
                return torch.zeros_like(depth)
    
    def compute_pearson_loss(self, 
                           rendered_depth: torch.Tensor,
                           estimated_depth: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算Pearson相关性深度损失 (FSGS核心损失函数)
        
        Args:
            rendered_depth: 渲染深度图 [H, W]
            estimated_depth: 估计深度图 [H, W] 
            mask: 可选遮罩 [H, W]
        
        Returns:
            Pearson相关性损失 (越小越好)
        """
        if not self.enabled or estimated_depth is None:
            return torch.tensor(0.0, device=rendered_depth.device)
        
        # 确保深度图在同一设备上
        if estimated_depth.device != rendered_depth.device:
            estimated_depth = estimated_depth.to(rendered_depth.device)
        
        # 应用遮罩
        if mask is not None:
            rendered_flat = rendered_depth[mask > 0.5]
            estimated_flat = estimated_depth[mask > 0.5]
        else:
            rendered_flat = rendered_depth.reshape(-1)
            estimated_flat = estimated_depth.reshape(-1)
        
        # 过滤无效值
        valid_mask = torch.isfinite(rendered_flat) & torch.isfinite(estimated_flat)
        rendered_flat = rendered_flat[valid_mask]
        estimated_flat = estimated_flat[valid_mask]
        
        if len(rendered_flat) < 10:  # 需要足够的有效像素
            return torch.tensor(0.0, device=rendered_depth.device)
        
        # 计算Pearson相关系数
        corr = self._pearson_correlation(rendered_flat, estimated_flat)
        
        # FSGS损失: 1 - |correlation|，使损失最小化对应相关性最大化
        loss = 1.0 - torch.abs(corr)
        
        return loss
    
    def _pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算Pearson相关系数"""
        # 计算均值
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        
        # 计算协方差和方差
        cov_xy = torch.mean((x - mean_x) * (y - mean_y))
        var_x = torch.mean((x - mean_x) ** 2)
        var_y = torch.mean((y - mean_y) ** 2)
        
        # 计算相关系数
        correlation = cov_xy / (torch.sqrt(var_x * var_y) + 1e-8)
        
        return correlation
    
    def compute_scale_invariant_loss(self,
                                   rendered_depth: torch.Tensor,
                                   estimated_depth: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算尺度不变深度损失 (备选损失函数)
        
        Args:
            rendered_depth: 渲染深度图
            estimated_depth: 估计深度图
            mask: 可选遮罩
        
        Returns:
            尺度不变深度损失
        """
        if not self.enabled or estimated_depth is None:
            return torch.tensor(0.0, device=rendered_depth.device)
        
        # 对数深度差异
        log_rendered = torch.log(rendered_depth + 1e-6)
        log_estimated = torch.log(estimated_depth + 1e-6)
        
        if mask is not None:
            log_diff = (log_rendered - log_estimated)[mask > 0.5]
        else:
            log_diff = log_rendered - log_estimated
        
        # 尺度不变损失
        loss = torch.mean(log_diff ** 2) - 0.5 * (torch.mean(log_diff) ** 2)
        
        return loss


def create_depth_estimator(model_type: str = "dpt_large", 
                         device: str = "cuda",
                         enable_fsgs_depth: bool = True) -> MonocularDepthEstimator:
    """
    创建深度估计器的便捷函数
    
    Args:
        model_type: 模型类型
        device: 设备
        enable_fsgs_depth: 是否启用FSGS深度功能 (向下兼容开关)
    
    Returns:
        深度估计器实例
    """
    if not enable_fsgs_depth:
        model_type = "disabled"
        print("📦 [Compatibility] FSGS depth estimation disabled by user setting")
    
    return MonocularDepthEstimator(
        model_type=model_type,
        device=device,
        enable_depth_estimation=enable_fsgs_depth
    )