"""
FSGS 深度估计器
使用 MiDaS 预训练模型估计单目深度

参考 FSGS 论文: https://github.com/VITA-Group/FSGS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

# 尝试导入 MiDaS
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("⚠️ timm 未安装，MiDaS 深度估计不可用。安装命令: pip install timm")


class MiDaSDepthEstimator(nn.Module):
    """
    MiDaS 深度估计器

    使用预训练的 DPT (Dense Prediction Transformer) 模型估计相对深度

    Args:
        model_type: 模型类型
            - "dpt_large": DPT-Large (最高质量，较慢)
            - "dpt_hybrid": DPT-Hybrid (平衡)
            - "midas_small": MiDaS v2.1 small (最快)
        device: 设备 (cuda/cpu)

    Example:
        >>> estimator = MiDaSDepthEstimator("dpt_large", device="cuda")
        >>> image = torch.randn(1, 3, 256, 256).cuda()  # RGB 图像
        >>> depth = estimator(image)  # 相对深度图
        >>> print(depth.shape)  # (1, 1, 256, 256)
    """

    # MiDaS 模型映射
    MODEL_MAP = {
        "dpt_large": "intel/dpt-large",
        "dpt_hybrid": "intel/dpt-hybrid-midas",
        "midas_small": "intel/midas-small",
    }

    def __init__(
        self,
        model_type: Literal["dpt_large", "dpt_hybrid", "midas_small"] = "dpt_large",
        device: str = "cuda"
    ):
        super().__init__()

        if not HAS_TIMM:
            raise ImportError("需要安装 timm: pip install timm")

        self.model_type = model_type
        self.device = device

        # 加载预训练模型
        print(f"📦 加载 MiDaS 模型: {model_type}")

        # torch.hub 模型名称映射
        HUB_MODEL_NAMES = {
            "dpt_large": "DPT_Large",
            "dpt_hybrid": "DPT_Hybrid",
            "midas_small": "MiDaS_small",
        }

        try:
            # 使用 torch.hub 加载 MiDaS
            hub_name = HUB_MODEL_NAMES.get(model_type, "DPT_Large")
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                hub_name,
                pretrained=True,
                trust_repo=True
            )
        except Exception as e:
            print(f"⚠️ torch.hub 加载失败: {e}")
            print("尝试使用 transformers 加载...")

            try:
                from transformers import DPTForDepthEstimation, DPTImageProcessor

                model_name = self.MODEL_MAP.get(model_type, "intel/dpt-large")
                self.model = DPTForDepthEstimation.from_pretrained(model_name)
                self.processor = DPTImageProcessor.from_pretrained(model_name)
                self._use_transformers = True
            except Exception as e2:
                raise RuntimeError(f"无法加载 MiDaS 模型: {e}, {e2}")
        else:
            self._use_transformers = False

        self.model = self.model.to(device)
        self.model.eval()

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

        # MiDaS 输入归一化参数
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        print(f"✅ MiDaS 模型加载完成")

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: (B, C, H, W) 输入图像，范围 [0, 1] 或 [-1, 1]

        Returns:
            normalized: (B, 3, 384, 384) 归一化后的图像
        """
        # 确保是 3 通道
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # 归一化到 [0, 1]
        if image.min() < 0:
            image = (image + 1) / 2

        # 调整大小到 MiDaS 输入尺寸
        image = F.interpolate(image, size=(384, 384), mode="bilinear", align_corners=False)

        # ImageNet 归一化
        image = (image - self.mean.to(image.device)) / self.std.to(image.device)

        return image

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        估计深度

        Args:
            image: (B, C, H, W) 输入图像

        Returns:
            depth: (B, 1, H, W) 相对深度图（值越大越远）
        """
        original_size = image.shape[2:]

        # 预处理
        x = self.preprocess(image)

        # MiDaS 推理
        if self._use_transformers:
            outputs = self.model(x)
            depth = outputs.predicted_depth.unsqueeze(1)
        else:
            depth = self.model(x)
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)

        # 调整回原始尺寸
        depth = F.interpolate(depth, size=original_size, mode="bilinear", align_corners=False)

        # 归一化到 [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算 Pearson 相关系数

    用于深度一致性损失：depth_loss = 1 - pearson_corrcoef(rendered_depth, midas_depth)

    Args:
        x: (N,) 或 (B, N) 张量
        y: (N,) 或 (B, N) 张量，形状与 x 相同

    Returns:
        corr: () 或 (B,) 相关系数，范围 [-1, 1]
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # 中心化
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)

    # 计算相关系数
    numerator = (x_centered * y_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (x_centered ** 2).sum(dim=-1) * (y_centered ** 2).sum(dim=-1)
    ) + 1e-8

    corr = numerator / denominator

    if squeeze_output:
        corr = corr.squeeze(0)

    return corr


def compute_depth_loss(
    rendered_depth: torch.Tensor,
    midas_depth: torch.Tensor,
    loss_type: Literal["pearson", "l1", "l2"] = "pearson"
) -> torch.Tensor:
    """
    计算深度一致性损失

    FSGS 论文使用 Pearson 相关系数作为深度损失：
    depth_loss = (1 - pearson_corrcoef(rendered_depth, -midas_depth)).mean()

    注意：MiDaS 输出的是逆深度（近处值大），所以需要取负号

    Args:
        rendered_depth: (B, 1, H, W) 渲染的深度图
        midas_depth: (B, 1, H, W) MiDaS 估计的深度图
        loss_type: 损失类型
            - "pearson": Pearson 相关系数 (FSGS 默认)
            - "l1": L1 损失
            - "l2": L2 损失

    Returns:
        loss: () 标量损失
    """
    # 展平
    rendered_flat = rendered_depth.view(rendered_depth.shape[0], -1)
    midas_flat = midas_depth.view(midas_depth.shape[0], -1)

    if loss_type == "pearson":
        # Pearson 相关系数损失
        # 注意：MiDaS 输出逆深度，所以取负号使其与 rendered_depth 方向一致
        corr = pearson_corrcoef(rendered_flat, -midas_flat)
        loss = (1 - corr).mean()
    elif loss_type == "l1":
        # 归一化后的 L1 损失
        rendered_norm = (rendered_flat - rendered_flat.mean(dim=-1, keepdim=True)) / (rendered_flat.std(dim=-1, keepdim=True) + 1e-8)
        midas_norm = (midas_flat - midas_flat.mean(dim=-1, keepdim=True)) / (midas_flat.std(dim=-1, keepdim=True) + 1e-8)
        loss = F.l1_loss(rendered_norm, -midas_norm)
    elif loss_type == "l2":
        # 归一化后的 L2 损失
        rendered_norm = (rendered_flat - rendered_flat.mean(dim=-1, keepdim=True)) / (rendered_flat.std(dim=-1, keepdim=True) + 1e-8)
        midas_norm = (midas_flat - midas_flat.mean(dim=-1, keepdim=True)) / (midas_flat.std(dim=-1, keepdim=True) + 1e-8)
        loss = F.mse_loss(rendered_norm, -midas_norm)
    else:
        raise ValueError(f"未知的损失类型: {loss_type}")

    return loss


# 全局深度估计器实例（惰性初始化）
_global_depth_estimator: Optional[MiDaSDepthEstimator] = None


def get_depth_estimator(
    model_type: str = "dpt_hybrid",
    device: str = "cuda"
) -> MiDaSDepthEstimator:
    """
    获取全局深度估计器实例（单例模式）

    Args:
        model_type: 模型类型
        device: 设备

    Returns:
        estimator: MiDaS 深度估计器
    """
    global _global_depth_estimator

    if _global_depth_estimator is None:
        _global_depth_estimator = MiDaSDepthEstimator(model_type, device)

    return _global_depth_estimator


def estimate_depth(
    image: torch.Tensor,
    model_type: str = "dpt_hybrid",
    device: str = "cuda"
) -> torch.Tensor:
    """
    便捷函数：估计图像深度

    Args:
        image: (B, C, H, W) 输入图像
        model_type: 模型类型
        device: 设备

    Returns:
        depth: (B, 1, H, W) 深度图

    Example:
        >>> image = torch.randn(1, 1, 256, 256).cuda()  # X-ray 投影
        >>> depth = estimate_depth(image)
        >>> print(depth.shape)  # (1, 1, 256, 256)
    """
    estimator = get_depth_estimator(model_type, device)
    return estimator(image)
