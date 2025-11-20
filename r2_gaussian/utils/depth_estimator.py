"""
DPT (Dense Prediction Transformer) 深度估计器
用于IPSM的单目深度估计
"""

import torch
import torch.nn.functional as F
from typing import Optional


class DPTDepthEstimator:
    """DPT深度估计封装，支持CT图像"""

    def __init__(self, model_type: str = "DPT_Hybrid", device: str = "cuda"):
        """
        Args:
            model_type: DPT模型类型 ("DPT_Hybrid" 或 "DPT_Large")
            device: 运行设备
        """
        self.device = device
        self.model = None
        self.transform = None
        self._load_model(model_type)

    def _load_model(self, model_type: str):
        """加载DPT模型"""
        try:
            # 尝试从torch hub加载
            self.model = torch.hub.load(
                'intel-isl/MiDaS',
                model_type,
                pretrained=True,
                skip_validation=True
            )
            self.model.to(self.device)
            self.model.eval()

            # 加载对应的transform
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            if model_type == "DPT_Large":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.dpt_transform

            print(f"✓ DPT模型加载成功: {model_type}")

        except Exception as e:
            print(f"⚠️  DPT加载失败: {e}")
            print("   将使用占位符模式（返回渲染深度的副本）")
            self.model = None

    @torch.no_grad()
    def estimate(
        self,
        image: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        估计单目深度

        Args:
            image: (C, H, W) 或 (1, C, H, W) 输入图像 [0, 1]
                   可以是灰度CT图像 (1, H, W) 或 RGB (3, H, W)
            normalize: 是否归一化到[0, 1]

        Returns:
            depth: (H, W) 深度图 [0, 1] (如果normalize=True)
        """
        if self.model is None:
            # 占位符：返回零深度图
            if image.dim() == 4:
                H, W = image.shape[-2:]
            else:
                H, W = image.shape[-2:]
            return torch.zeros(H, W, device=self.device)

        # 确保输入是4D tensor
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        # CT图像转RGB (如果是单通道)
        if image.shape[1] == 1:
            image_rgb = image.repeat(1, 3, 1, 1)  # (1, 1, H, W) -> (1, 3, H, W)
        else:
            image_rgb = image

        # 转换为numpy用于MiDaS transform
        import numpy as np
        image_np = image_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        image_np = (image_np * 255).astype(np.uint8)

        # 应用transform
        input_batch = self.transform(image_np).to(self.device)

        # 推理
        prediction = self.model(input_batch)

        # 调整大小到原始尺寸
        H, W = image.shape[-2:]
        depth = F.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode='bicubic',
            align_corners=False
        ).squeeze()

        # 归一化
        if normalize:
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    def estimate_batch(
        self,
        images: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        批量估计深度

        Args:
            images: (B, C, H, W) 批量图像
            normalize: 是否归一化

        Returns:
            depths: (B, H, W) 批量深度图
        """
        depths = []
        for img in images:
            depth = self.estimate(img, normalize=normalize)
            depths.append(depth)
        return torch.stack(depths)


# 全局单例（避免重复加载）
_global_depth_estimator: Optional[DPTDepthEstimator] = None


def get_depth_estimator(device: str = "cuda") -> DPTDepthEstimator:
    """获取全局深度估计器单例"""
    global _global_depth_estimator
    if _global_depth_estimator is None:
        _global_depth_estimator = DPTDepthEstimator(device=device)
    return _global_depth_estimator
