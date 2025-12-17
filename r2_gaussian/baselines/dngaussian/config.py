#
# DNGaussian 配置参数
#
# 基于 CVPR 2024 DNGaussian，适配 CT 重建场景
#


class DNGaussianConfig:
    """DNGaussian 默认配置"""

    # ============ GridRenderer (Neural Renderer) 参数 ============
    # Hash Grid 编码器参数
    num_levels: int = 16           # 分辨率级数
    level_dim: int = 2             # 每级特征维度
    base_resolution: int = 16      # 基础分辨率
    log2_hashmap_size: int = 19    # 哈希表大小 (2^19)
    desired_resolution: int = 512  # 目标分辨率

    # Sigma Network (MLP) 参数
    hidden_dim: int = 64           # 隐藏层维度
    num_layers: int = 3            # 层数
    geo_feat_dim: int = 64         # 几何特征维度

    # ============ 深度正则化参数 ============
    # 深度损失启动时机
    hard_depth_start: int = 500    # Hard depth loss 开始迭代
    soft_depth_start: int = 1000   # Soft depth loss 开始迭代
    depth_end_iter: int = 15000    # 深度正则化结束迭代

    # Patch Norm Loss 参数
    patch_size_min: int = 5        # 最小 patch 大小
    patch_size_max: int = 17       # 最大 patch 大小
    error_tolerance: float = 0.01  # 误差容忍度 (margin)

    # 深度损失权重
    lambda_depth_local: float = 0.1   # 局部深度损失权重
    lambda_depth_global: float = 1.0  # 全局深度损失权重
    lambda_depth_hard: float = 0.5    # Hard depth loss 权重
    lambda_depth_soft: float = 0.5    # Soft depth loss 权重

    # ============ 训练参数（继承自 X-Gaussian）============
    # 球谐阶数（CT 场景不需要，但保留兼容性）
    sh_degree: int = 0

    # 学习率
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # Neural Renderer 学习率
    neural_encoder_lr: float = 0.01   # Hash Grid 编码器学习率
    neural_network_lr: float = 0.001  # MLP 学习率

    # 密集化参数
    percent_dense: float = 0.01
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005

    # DSSIM 损失权重
    lambda_dssim: float = 0.2

    # ============ CT 场景特定参数 ============
    # 投影深度计算方法
    depth_prior_method: str = 'inverse'  # 'inverse' 或 'log'

    # opacity 调制方式
    modulation_mode: str = 'multiplicative'  # 'multiplicative' 或 'additive'
    modulation_strength: float = 0.5  # 调制强度


def get_dngaussian_config():
    """获取 DNGaussian 默认配置"""
    return DNGaussianConfig()
