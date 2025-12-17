#
# Baseline method registry and abstract interfaces
#

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn


class BaseModel(ABC):
    """所有 baseline 模型的抽象基类"""

    @abstractmethod
    def get_trainable_params(self) -> List[Dict]:
        """返回可训练参数组（用于优化器）"""
        pass

    @abstractmethod
    def training_setup(self, opt):
        """设置训练优化器和调度器"""
        pass

    @abstractmethod
    def capture(self) -> Dict:
        """捕获模型状态用于保存"""
        pass

    @abstractmethod
    def restore(self, state: Dict, opt):
        """从保存状态恢复模型"""
        pass

    @abstractmethod
    def get_num_points(self) -> int:
        """返回模型点/参数数量"""
        pass


class GaussianBaseModel(BaseModel):
    """3DGS 系列模型基类（R2-Gaussian, X-Gaussian）"""

    @abstractmethod
    def get_xyz(self) -> torch.Tensor:
        """获取高斯中心位置"""
        pass

    @abstractmethod
    def get_scaling(self) -> torch.Tensor:
        """获取尺度参数"""
        pass

    @abstractmethod
    def get_rotation(self) -> torch.Tensor:
        """获取旋转参数"""
        pass

    @abstractmethod
    def densify_and_prune(self, *args, **kwargs):
        """密度控制（自适应）"""
        pass

    @abstractmethod
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """添加密度化统计信息"""
        pass


class NeRFBaseModel(BaseModel):
    """NeRF 系列模型基类（NAF, TensoRF, SAX-NeRF）"""

    @property
    @abstractmethod
    def network(self) -> nn.Module:
        """返回主网络"""
        pass

    @property
    def network_fine(self) -> Optional[nn.Module]:
        """返回精细网络（可选）"""
        return None

    @abstractmethod
    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """查询 3D 点的密度值

        Args:
            pts: [N, 3] 3D 坐标点

        Returns:
            densities: [N, 1] 密度值
        """
        pass


class BaseRenderer(ABC):
    """渲染器抽象基类"""

    @abstractmethod
    def render(
        self,
        viewpoint,
        model: BaseModel,
        pipe,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        渲染投影图像

        Args:
            viewpoint: 相机视角
            model: 模型实例
            pipe: 管线参数

        Returns:
            dict with keys:
                - 'render': 渲染图像 [1, H, W] 或 [C, H, W]
                - 'viewspace_points': 用于梯度计算（3DGS）
                - 'visibility_filter': 可见性掩码
                - 'radii': 2D 半径
        """
        pass


# 方法注册表
METHOD_REGISTRY = {
    'r2_gaussian': {
        'type': 'gaussian',
        'description': 'R2-Gaussian (本项目默认方法)',
    },
    'xgaussian': {
        'type': 'gaussian',
        'description': 'X-Gaussian baseline',
        'module': 'r2_gaussian.baselines.xgaussian',
    },
    'naf': {
        'type': 'nerf',
        'description': 'Neural Attenuation Fields',
        'module': 'r2_gaussian.baselines.naf',
    },
    'tensorf': {
        'type': 'nerf',
        'description': 'TensoRF',
        'module': 'r2_gaussian.baselines.tensorf',
    },
    'saxnerf': {
        'type': 'nerf',
        'description': 'SAX-NeRF with Lineformer',
        'module': 'r2_gaussian.baselines.saxnerf',
    },
    'corgs': {
        'type': 'gaussian',
        'description': 'CoR-GS: Co-Regularization Gaussian Splatting',
        'module': 'r2_gaussian.baselines.corgs',
    },
    'dngaussian': {
        'type': 'gaussian',
        'description': 'DNGaussian: Sparse-View 3DGS with Depth Normalization (CVPR 2024)',
        'module': 'r2_gaussian.baselines.dngaussian',
    },
    'fsgs': {
        'type': 'gaussian',
        'description': 'FSGS: Few-Shot Gaussian Splatting with proximity densification (ECCV 2024)',
        'module': 'r2_gaussian.baselines.fsgs',
    },
}


def get_method_config(method: str) -> Dict[str, Any]:
    """获取方法配置

    Args:
        method: 方法名称

    Returns:
        方法配置字典
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[method]


def get_method_type(method: str) -> str:
    """获取方法类型（gaussian 或 nerf）"""
    return get_method_config(method)['type']


def is_gaussian_method(method: str) -> bool:
    """判断是否为 3DGS 类型方法"""
    return get_method_type(method) == 'gaussian'


def is_nerf_method(method: str) -> bool:
    """判断是否为 NeRF 类型方法"""
    return get_method_type(method) == 'nerf'
