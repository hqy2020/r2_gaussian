"""
SSS (Student Splatting and Scooping) 辅助函数
用于 R²-Gaussian baseline 的 PyTorch 层面 Student-t 近似实现

生成日期: 2025-11-17
作者: PyTorch/CUDA 编程专家
"""

import torch
import torch.nn.functional as F


def inverse_tanh(x):
    """
    计算 tanh 的反函数: artanh(x) = 0.5 * log((1+x)/(1-x))

    Args:
        x: 输入张量, 范围 (-1, 1)

    Returns:
        y: 输出张量, 范围 (-∞, +∞)

    Notes:
        - 当 x → ±1 时,log 趋于无穷,使用 clamp 防止数值溢出
        - 添加 eps 避免除零
    """
    x_clamped = torch.clamp(x, -0.999, 0.999)
    eps = 1e-8
    return 0.5 * torch.log((1 + x_clamped) / (1 - x_clamped + eps))


def compute_student_t_radius_multiplier(nu):
    """
    根据 ν 计算 Student-t 的有效半径放大因子

    参考 SSS 论文的经验公式 (forward.cu Line 242-286):
        - ν=1: 63.657 (极端长尾)
        - ν=2: 9.925
        - ν=3: 5.841
        - ν=8: 3.055
        - ν→∞: 3.0 (高斯极限)

    本实现采用简化的线性插值:
        - ν ∈ [2, 8] → multiplier ∈ [5.0, 3.0]

    Args:
        nu: 自由度张量, shape (N, 1), 范围 [2, 8]

    Returns:
        multiplier: 半径放大因子, shape (N, 1), 范围 [3.0, 10.0]
    """
    # 线性插值: nu=2 → 5.0x, nu=8 → 3.0x
    multiplier = 5.0 - (nu - 2) * (2.0 / 6.0)  # [3.0, 5.0]
    # 裁剪防止异常值
    return torch.clamp(multiplier, 3.0, 10.0)


def compute_depth_smoothness(depth_map):
    """
    计算深度图的平滑度损失 (Sobel 梯度的 L1 norm)

    用于 Student-t 深度监督: 长尾分布应该产生更平滑的深度图

    Args:
        depth_map: 深度图张量
            - shape (H, W) 或 (1, H, W)

    Returns:
        smoothness_loss: 标量损失值
    """
    if depth_map.ndim == 2:
        depth_map = depth_map.unsqueeze(0)  # (H, W) → (1, H, W)

    # Sobel 算子
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=depth_map.dtype,
        device=depth_map.device
    )
    sobel_y = sobel_x.t()

    # 添加 batch 和 channel 维度: (1, H, W) → (1, 1, H, W)
    depth_4d = depth_map.unsqueeze(0)
    sobel_x_4d = sobel_x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    sobel_y_4d = sobel_y.unsqueeze(0).unsqueeze(0)

    # 卷积计算梯度
    grad_x = F.conv2d(depth_4d, sobel_x_4d, padding=1)  # (1, 1, H, W)
    grad_y = F.conv2d(depth_4d, sobel_y_4d, padding=1)

    # 梯度幅值
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    # 返回平均梯度 (越小越平滑)
    return grad_magnitude.mean()


# 单元测试 (仅在直接运行时执行)
if __name__ == "__main__":
    print("Testing sss_helpers.py...")

    # Test inverse_tanh
    x = torch.linspace(-0.99, 0.99, 100)
    y = torch.tanh(inverse_tanh(x))
    assert torch.allclose(x, y, atol=1e-3), "inverse_tanh failed"
    print("✅ inverse_tanh passed")

    # Test compute_student_t_radius_multiplier
    nu = torch.tensor([[2.0], [5.0], [8.0]])
    mult = compute_student_t_radius_multiplier(nu)
    assert mult[0] > mult[2], "Radius multiplier should decrease with nu"
    print(f"✅ radius_multiplier passed: nu=2→{mult[0].item():.2f}x, nu=8→{mult[2].item():.2f}x")

    # Test compute_depth_smoothness
    depth = torch.randn(64, 64).cuda() if torch.cuda.is_available() else torch.randn(64, 64)
    loss = compute_depth_smoothness(depth)
    assert loss > 0, "Smoothness loss should be positive"
    print(f"✅ depth_smoothness passed: loss={loss.item():.4f}")

    print("All tests passed!")
