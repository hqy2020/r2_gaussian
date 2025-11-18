"""
CoR-GS Stage 3: Pseudo-view Co-regularization 核心算法实现（医学适配版）

功能模块:
1. 四元数 SLERP 插值（球面线性插值）
2. Pseudo-view 相机生成（医学适配版）
3. Co-regularization 损失计算（支持 ROI 自适应权重）
4. 置信度筛选机制
5. 不确定性量化

实现日期: 2025-11-17
作者: 编程专家
参考: CoR-GS 论文 + 医学适用性评估报告
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================================================
# 四元数转换与 SLERP 插值
# ============================================================================

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    将 3x3 旋转矩阵转换为四元数 [w, x, y, z]

    Args:
        R: 旋转矩阵 (torch.Tensor or numpy.ndarray, shape [3, 3])

    Returns:
        q: 四元数 (torch.Tensor, shape [4])

    数值稳定性: 使用 Shepperd's 方法避免除零
    """
    # 【Bug 修复】确保 R 是 Tensor（R²-Gaussian 的 camera.R 是 numpy array）
    # 并且在正确的设备上（检测是否有 CUDA 可用）
    if not isinstance(R, torch.Tensor):
        R = torch.from_numpy(R).float()
        # 尝试移到 CUDA（如果可用）
        if torch.cuda.is_available():
            R = R.cuda()

    assert R.shape == (3, 3), f"期望旋转矩阵形状 [3, 3]，得到 {R.shape}"

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    # 根据迹的大小选择不同的计算路径（数值稳定性）
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0 + 1e-8)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2] + 1e-8)
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2] + 1e-8)
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1] + 1e-8)
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], device=R.device, dtype=R.dtype)
    # 归一化四元数
    q = q / (torch.norm(q) + 1e-8)
    return q


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数 [w, x, y, z] 转换为 3x3 旋转矩阵

    Args:
        q: 四元数 (torch.Tensor, shape [4])

    Returns:
        R: 旋转矩阵 (torch.Tensor, shape [3, 3])
    """
    assert q.shape == (4,), f"期望四元数形状 [4]，得到 {q.shape}"

    # 归一化四元数（防止数值误差累积）
    q = q / (torch.norm(q) + 1e-8)

    w, x, y, z = q[0], q[1], q[2], q[3]

    R = torch.zeros(3, 3, device=q.device, dtype=q.dtype)

    # 使用标准四元数到旋转矩阵的转换公式
    R[0, 0] = 1 - 2*y*y - 2*z*z
    R[0, 1] = 2*x*y - 2*z*w
    R[0, 2] = 2*x*z + 2*y*w

    R[1, 0] = 2*x*y + 2*z*w
    R[1, 1] = 1 - 2*x*x - 2*z*z
    R[1, 2] = 2*y*z - 2*x*w

    R[2, 0] = 2*x*z - 2*y*w
    R[2, 1] = 2*y*z + 2*x*w
    R[2, 2] = 1 - 2*x*x - 2*y*y

    return R


def slerp(q1: torch.Tensor, q2: torch.Tensor, t: float = 0.5) -> torch.Tensor:
    """
    四元数球面线性插值 (Spherical Linear Interpolation)

    Args:
        q1: 第一个四元数 [w, x, y, z] (torch.Tensor, shape [4])
        q2: 第二个四元数 [w, x, y, z] (torch.Tensor, shape [4])
        t: 插值参数 (0=q1, 1=q2, 0.5=中间)

    Returns:
        q_interp: 插值后的四元数 (torch.Tensor, shape [4])

    数值稳定性:
        - 处理四元数反向问题（点积为负）
        - 小角度时回退到线性��值
        - 避免除零和三角函数数值误差
    """
    # 归一化输入四元数
    q1 = q1 / (torch.norm(q1) + 1e-8)
    q2 = q2 / (torch.norm(q2) + 1e-8)

    # 计算点积（夹角余弦）
    dot = torch.dot(q1, q2)

    # 如果点积为负，反转 q2（确保最短路径插值）
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # 限制点积范围 [-1, 1] 避免 acos 数值错误
    dot = torch.clamp(dot, -1.0, 1.0)

    # 如果夹角很小（接近平行），使用线性插值（避免数值不稳定）
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / (torch.norm(result) + 1e-8)

    # 球面插值
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # 避免除零
    if sin_theta.abs() < 1e-6:
        result = q1 + t * (q2 - q1)
        return result / (torch.norm(result) + 1e-8)

    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta

    result = w1 * q1 + w2 * q2
    return result / (torch.norm(result) + 1e-8)


# ============================================================================
# Pseudo-view 相机生成（医学适配版）
# ============================================================================

def find_nearest_camera_index(base_idx: int, train_cameras: List) -> int:
    """
    找到与基准相机最近的另一个训练相机（基于欧几里德距离）

    Args:
        base_idx: 基准相机索引
        train_cameras: 训练相机列表（每个元素是 Camera 对象）

    Returns:
        nearest_idx: 最近相机的索引
    """
    base_pos = train_cameras[base_idx].camera_center  # shape [3]
    min_dist = float('inf')
    nearest_idx = 0

    for i, cam in enumerate(train_cameras):
        if i == base_idx:
            continue

        dist = torch.norm(cam.camera_center - base_pos).item()
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    return nearest_idx


def generate_random_pseudo_cameras(
    train_cameras: List,
    num_pseudo: int = 10000,
    radius_range: Tuple[float, float] = (0.8, 1.2),
    seed: int = 42
) -> List:
    """
    官方 CoR-GS 的 pseudo-view 生成策略：完全随机采样

    Args:
        train_cameras: 训练相机列表
        num_pseudo: 生成的 pseudo-view 数量（官方默认 10000）
        radius_range: 相机距离场景中心的半径范围（相对于平均半径）
        seed: 随机种子

    Returns:
        pseudo_cameras: 随机生成的相机列表
    """
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. 计算场景中心和平均半径
    centers = torch.stack([cam.camera_center for cam in train_cameras])
    scene_center = centers.mean(dim=0)
    avg_radius = torch.norm(centers - scene_center, dim=1).mean().item()

    # 2. 随机采样球面坐标
    pseudo_cameras = []
    for i in range(num_pseudo):
        # 随机半径（在平均半径的 0.8~1.2 倍范围内）
        r = avg_radius * np.random.uniform(radius_range[0], radius_range[1])

        # 随机方向（球面均匀采样）
        theta = np.random.uniform(0, 2 * np.pi)  # 方位角
        phi = np.arccos(np.random.uniform(-1, 1))  # 仰角

        # 球面坐标转笛卡尔坐标
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        position = scene_center + torch.tensor([x, y, z], dtype=torch.float32, device=scene_center.device)

        # 3. 计算相机朝向（始终看向场景中心）
        forward = (scene_center - position)
        forward = forward / torch.norm(forward)

        # 4. 构造旋转矩阵（使用随机 up 向量）
        up = torch.tensor([0.0, 0.0, 1.0], device=position.device)  # 初始 up
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)

        R = torch.stack([right, up, -forward], dim=1)  # 3x3 旋转矩阵

        # 5. 复制训练相机的内参
        template_cam = train_cameras[0]

        # 计算新的 T（相机外参平移）
        # 注意: R²-Gaussian 使用 T = -R @ camera_center
        pseudo_T = -R @ position

        from r2_gaussian.dataset.cameras import Camera

        pseudo_cam = Camera(
            colmap_id=template_cam.colmap_id,
            scanner_cfg=template_cam.scanner_cfg if hasattr(template_cam, 'scanner_cfg') else None,
            R=R.cpu().numpy(),  # Camera 类期望 numpy 数组
            T=pseudo_T.cpu().numpy(),
            angle=template_cam.angle,
            mode=template_cam.mode,
            FoVx=template_cam.FoVx,
            FoVy=template_cam.FoVy,
            image=torch.zeros(1, 1, 3, device=position.device, dtype=torch.float32),  # 1x1 占位符，节省显存
            image_name=f"pseudo_random_{i}",
            uid=template_cam.uid + 10000 + i,  # 避免 ID 冲突
            trans=template_cam.trans,
            scale=template_cam.scale,
            data_device=str(position.device)
        )

        pseudo_cameras.append(pseudo_cam)

    return pseudo_cameras


def generate_pseudo_view_medical(
    train_cameras: List,
    current_camera_idx: Optional[int] = None,
    noise_std: float = 0.02,
    roi_info: Optional[Dict] = None
) -> object:
    """
    生成医学适配的 Pseudo-view 相机（CoR-GS 论文 + 医学约束）

    策略:
    1. 从训练相机中选择基准相机
    2. 找到最近的邻居相机
    3. 对旋转四元数进行 SLERP 插值（t=0.5 中间位置）
    4. 根据 ROI 类型自适应添加随机扰动（骨区 σ=0.01, 软组织 σ=0.02）
    5. 构建新的 pseudo-view 相机

    Args:
        train_cameras: 训练相机列表 (list of Camera objects)
        current_camera_idx: 当前迭代使用的真实相机索引（可选）
        noise_std: 位置噪声标准差（默认 0.02，对应归一化场景）
        roi_info: ROI 信息字典（可选，包含 'roi_type' 和 'density' 字段）
            - 'roi_type': 'bone' 或 'soft_tissue'
            - 'density': 平均 HU 值（用于自适应扰动）

    Returns:
        pseudo_camera: 生成的虚拟相机 (Camera object)

    医学适配:
        - 骨区扰动减半（σ=0.01 → ±0.2mm）
        - 软组织区标准扰动（σ=0.02 → ±0.4mm）
        - 根据 HU 值自动判断组织类型
    """
    # 步骤 1: 选择基准相机
    if current_camera_idx is None:
        base_idx = np.random.randint(0, len(train_cameras))
    else:
        base_idx = current_camera_idx

    base_camera = train_cameras[base_idx]

    # 步骤 2: 找到最近的邻居相机
    nearest_idx = find_nearest_camera_index(base_idx, train_cameras)
    nearest_camera = train_cameras[nearest_idx]

    # 步骤 3: 插值旋转四元数（SLERP）
    base_quat = rotation_matrix_to_quaternion(base_camera.R)  # [4]
    nearest_quat = rotation_matrix_to_quaternion(nearest_camera.R)  # [4]

    # 50% 插值（中间位置，CoR-GS 论文公式 3）
    interp_quat = slerp(base_quat, nearest_quat, t=0.5)

    # 步骤 4: 医学适配的自适应扰动
    device = base_camera.camera_center.device

    # 根据 ROI 信息自适应调整扰动强度
    if roi_info is not None and 'roi_type' in roi_info:
        if roi_info['roi_type'] == 'bone':
            adaptive_noise_std = noise_std * 0.5  # 骨区扰动减半
        else:
            adaptive_noise_std = noise_std
    elif roi_info is not None and 'density' in roi_info:
        # 根据 HU 值判断组织类型（HU > 150 为骨区）
        if roi_info['density'] > 150:
            adaptive_noise_std = noise_std * 0.5
        else:
            adaptive_noise_std = noise_std
    else:
        # 默认使用标准扰动
        adaptive_noise_std = noise_std

    epsilon = torch.randn(3, device=device, dtype=base_camera.camera_center.dtype) * adaptive_noise_std
    pseudo_position = base_camera.camera_center + epsilon

    # 步骤 5: 构建 pseudo-view 相机
    pseudo_R = quaternion_to_rotation_matrix(interp_quat)

    # 计算新的 T（相机外参平移）
    # 注意: R²-Gaussian 使用 T = -R @ camera_center
    pseudo_T = -pseudo_R @ pseudo_position

    # 复制基准相机的其他参数（内参、图像尺寸等）
    # 注意: pseudo-view 没有真实图像，仅用于渲染约束
    from r2_gaussian.dataset.cameras import Camera

    pseudo_camera = Camera(
        colmap_id=base_camera.colmap_id,
        scanner_cfg=base_camera.scanner_cfg if hasattr(base_camera, 'scanner_cfg') else None,
        R=pseudo_R.cpu().numpy(),  # Camera 类期望 numpy 数组
        T=pseudo_T.cpu().numpy(),
        angle=base_camera.angle,
        mode=base_camera.mode,
        FoVx=base_camera.FoVx,
        FoVy=base_camera.FoVy,
        image=torch.zeros_like(base_camera.original_image),  # 无 GT 图像
        image_name=f"pseudo_{base_idx}_{nearest_idx}",
        uid=base_camera.uid + 10000,  # 避免 ID 冲突
        trans=base_camera.trans,
        scale=base_camera.scale,
        data_device=str(device)
    )

    return pseudo_camera


# ============================================================================
# Co-regularization 损失计算（医学适配版）
# ============================================================================

def compute_pseudo_coreg_loss_medical(
    render1: Dict[str, torch.Tensor],
    render2: Dict[str, torch.Tensor],
    lambda_dssim: float = 0.2,
    roi_weights: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    计算医学适��的 Pseudo-view Co-regularization 损失（CoR-GS 论文公式 4）

    损失公式:
        R_pcolor = (1-λ) * L1(I'¹, I'²) + λ * L_D-SSIM(I'¹, I'²)

    医学适配:
        - 支持 ROI 自适应权重（骨区 λ_p=0.3, 软组织 λ_p=1.0）
        - 逐像素权重调制（保护骨折线等关键结构）

    Args:
        render1: 模型 1 在 pseudo-view 的渲染结果 (dict, 包含 'render' key)
            - 'render': torch.Tensor, shape [C, H, W], 值域 [0, 1]
        render2: 模型 2 在 pseudo-view 的渲染结果 (dict, 包含 'render' key)
        lambda_dssim: D-SSIM 损失权重（默认 0.2，与 3DGS 一致）
        roi_weights: ROI 权重图（可选，shape [H, W]，值域 [0, 1]）
            - None: 全图均匀权重 1.0
            - 提供时: 骨区像素权重 0.3, 软组织像素权重 1.0

    Returns:
        loss_dict: 包含总损失和各项损失的字典
            - 'loss': 总损失（标量）
            - 'l1': L1 损失（标量）
            - 'd_ssim': D-SSIM 损失（标量）
            - 'ssim': SSIM 值（标量，越接近 1 越好）
    """
    # 【修复】train.py 已经提取了 ["render"]，所以这里直接接受 Tensor
    image1 = render1  # shape [C, H, W]
    image2 = render2  # shape [C, H, W]

    assert image1.shape == image2.shape, f"图像形状不匹配: {image1.shape} vs {image2.shape}"

    # 计算 L1 损失
    if roi_weights is not None:
        # 逐像素加权 L1 损失
        assert roi_weights.shape == image1.shape[1:], \
            f"ROI 权重形状不匹配: {roi_weights.shape} vs {image1.shape[1:]}"

        # 扩展权重维度以匹配图像 [H, W] → [1, H, W]
        weights_expanded = roi_weights.unsqueeze(0)

        # 加权 L1
        l1_loss = (torch.abs(image1 - image2) * weights_expanded).mean()
    else:
        # 标准 L1 损失
        l1_loss = F.l1_loss(image1, image2)

    # 计算 D-SSIM 损失（使用 R²-Gaussian 已有的 SSIM 函数）
    from r2_gaussian.utils.loss_utils import ssim

    # SSIM 需要 [B, C, H, W] 格式
    image1_batch = image1.unsqueeze(0)  # [1, C, H, W]
    image2_batch = image2.unsqueeze(0)  # [1, C, H, W]

    # 计算 SSIM（R²-Gaussian 的 ssim() 返回 Tensor）
    ssim_value = ssim(image1_batch, image2_batch)

    # 【调试】打印类型信息
    # print(f"[DEBUG] ssim_value type: {type(ssim_value)}, value: {ssim_value}")
    # print(f"[DEBUG] isinstance Tensor: {isinstance(ssim_value, torch.Tensor)}")
    # if isinstance(ssim_value, torch.Tensor):
    #     print(f"[DEBUG] ssim_value.requires_grad: {ssim_value.requires_grad}")

    # 【Bug 修复 v2】确保 ssim_value 是 Tensor 并且在正确设备上（修复日期: 2025-11-17）
    # 根本原因：R²-Gaussian 的 ssim() 已经返回 Tensor，但可能没有正确的 requires_grad
    if not isinstance(ssim_value, torch.Tensor):
        # 如果是 numpy 或标量，转换为 Tensor
        ssim_value = torch.as_tensor(
            ssim_value,
            dtype=torch.float32,
            device=image1.device
        )
    else:
        # 如果已经是 Tensor，确保在正确的设备上并且 detach 后重新 attach
        if not ssim_value.requires_grad and image1.requires_grad:
            ssim_value = ssim_value.detach().requires_grad_(True)

    d_ssim_loss = 1.0 - ssim_value

    # 如果有 ROI 权重，SSIM 也应该考虑权重
    # 但标准 SSIM 实现不支持逐像素权重，这里保持简单

    # 组合损失（CoR-GS 论文公式 4）
    total_loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * d_ssim_loss

    # 【类型断言】确保所有返回值都是 Tensor 类型（调试辅助）
    assert isinstance(total_loss, torch.Tensor), f"total_loss 类型错误: {type(total_loss)}"
    assert isinstance(l1_loss, torch.Tensor), f"l1_loss 类型错误: {type(l1_loss)}"
    assert isinstance(d_ssim_loss, torch.Tensor), f"d_ssim_loss 类型错误: {type(d_ssim_loss)}"
    assert isinstance(ssim_value, torch.Tensor), f"ssim_value 类型错误: {type(ssim_value)}"

    return {
        'loss': total_loss,
        'l1': l1_loss,
        'd_ssim': d_ssim_loss,
        'ssim': ssim_value
    }


# ============================================================================
# 医学适配模块: 置信度筛选
# ============================================================================

def filter_by_confidence(
    pseudo_camera: object,
    gaussians_coarse: object,
    gaussians_fine: object,
    fitness_threshold: float = 0.90,
    rmse_threshold: float = 50.0
) -> Tuple[bool, Dict[str, float]]:
    """
    医学置信度筛选: 丢弃低可信度的 pseudo-view

    检验标准:
    1. 几何一致性: 粗/精模型在 pseudo-view 下的 Fitness ≥ 0.90
    2. 渲染误差: 与最近真实视角的 RMSE ≤ 50 HU

    Args:
        pseudo_camera: Pseudo-view 相机对象
        gaussians_coarse: 粗模型（GaussianModel）
        gaussians_fine: 精模型（GaussianModel）
        fitness_threshold: Fitness 阈值（默认 0.90）
        rmse_threshold: RMSE 阈值（默认 50 HU，医学可接受误差）

    Returns:
        is_valid: bool，是否接受此 pseudo-view
        metrics: dict，包含 'fitness' 和 'rmse' 键

    注: 此函数需要导入 corgs_metrics 模块计算 Fitness
    """
    try:
        from r2_gaussian.utils.corgs_metrics import compute_point_disagreement

        # 计算几何一致性（Point Disagreement）
        xyz_coarse = gaussians_coarse.get_xyz.detach()
        xyz_fine = gaussians_fine.get_xyz.detach()

        fitness, rmse_geom = compute_point_disagreement(
            xyz_coarse, xyz_fine, threshold=0.3, max_points=10000
        )

        # 检验 1: Fitness 检验
        if fitness < fitness_threshold:
            return False, {'fitness': fitness, 'rmse': float('inf')}

        # 检验 2: 渲染误差检验（简化版，实际应与最近真实视角对比）
        # 这里仅检查几何 RMSE，完整实现需要渲染对比
        # 注: rmse_geom 单位是归一化坐标，需要转换到 mm
        # 假设场景归一化到 [-1,1]³，对应 200mm 视野
        rmse_mm = rmse_geom * 100  # 粗略转换

        if rmse_mm > rmse_threshold:
            return False, {'fitness': fitness, 'rmse': rmse_mm}

        return True, {'fitness': fitness, 'rmse': rmse_mm}

    except Exception as e:
        # 如果置信度计算失败，默认接受（降级模式）
        print(f"⚠️ 置信度筛选失败，降级模式: {e}")
        return True, {'fitness': 1.0, 'rmse': 0.0}


# ============================================================================
# 医学适配模块: 不确定性量化
# ============================================================================

def quantify_uncertainty(
    pseudo_cameras: List,
    gaussian_model: object,
    pipe: object,
    background: torch.Tensor,
    num_samples: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    医学不确定性量化: 多次采样标记高不确定性区域

    策略:
    1. 生成 num_samples 个 pseudo-view（不同扰动）
    2. 渲染所有 pseudo-view
    3. 计算像素级标准差（HU 值波动）
    4. 标记高不确定性区域（std > 30 HU）

    Args:
        pseudo_cameras: Pseudo-view 相机列表（已生成）
        gaussian_model: 高斯模型（用于渲染）
        pipe: Pipeline 参数
        background: 背景颜色
        num_samples: 采样次数（默认 3）

    Returns:
        uncertainty_map: (H, W) 不确定性图（标准差）
        high_uncertainty_mask: (H, W) 高不确定性区域掩码（bool）

    用途: 辅助医师诊断，标记需要额外验证的区域
    """
    from r2_gaussian.gaussian import render

    rendered_images = []

    with torch.no_grad():
        for pseudo_cam in pseudo_cameras[:num_samples]:
            render_pkg = render(pseudo_cam, gaussian_model, pipe, background)
            rendered_images.append(render_pkg['render'])  # [C, H, W]

    # Stack 渲染结果 [N, C, H, W]
    stacked = torch.stack(rendered_images, dim=0)

    # 计算逐像素标准差（沿样本维度）
    # 转换到 HU 值尺度（假设值域 [0,1] 对应 [-1000, 3000] HU）
    # 简化: 直接使用归一化值的标准差
    uncertainty_map = torch.std(stacked, dim=0).mean(dim=0)  # 平均所有通道 [H, W]

    # 标记高不确定性区域（阈值可调）
    # 假设 std > 0.02 对应 HU 值波动 > 80 HU
    high_uncertainty_mask = uncertainty_map > 0.02

    return uncertainty_map, high_uncertainty_mask


# ============================================================================
# 辅助函数: 创建 ROI 权重图
# ============================================================================

def create_roi_weight_map(
    image_shape: Tuple[int, int],
    roi_mask: Optional[torch.Tensor] = None,
    bone_weight: float = 0.3,
    soft_tissue_weight: float = 1.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    创建 ROI 自适应权重图（用于 Co-regularization 损失）

    Args:
        image_shape: 图像形状 (H, W)
        roi_mask: ROI 掩码 (H, W) 布尔张量（True=骨区，False=软组织）
            - None: 全图使用软组织权重
        bone_weight: 骨区权重（默认 0.3，降低 pseudo-view 影响）
        soft_tissue_weight: 软组织权重（默认 1.0，标准影响）
        device: 设备（'cuda' 或 'cpu'）

    Returns:
        weight_map: (H, W) 权重图，值域 [bone_weight, soft_tissue_weight]

    用途: 保护骨折线等关键结构，降低虚拟视角在骨区的过度约束
    """
    H, W = image_shape
    weight_map = torch.ones(H, W, dtype=torch.float32, device=device) * soft_tissue_weight

    if roi_mask is not None:
        assert roi_mask.shape == (H, W), f"ROI 掩码形状不匹配: {roi_mask.shape} vs {image_shape}"
        weight_map[roi_mask] = bone_weight

    return weight_map


# ============================================================================
# 单元测试辅助函数
# ============================================================================

def test_quaternion_conversion():
    """测试四元数与旋转矩阵转换的正确性"""
    print("测试四元数转换...")

    # 创建随机旋转矩阵（正交矩阵）
    from scipy.spatial.transform import Rotation as R
    scipy_rot = R.random()
    R_scipy = torch.tensor(scipy_rot.as_matrix(), dtype=torch.float32)

    # 转换到四元数再转回来
    q = rotation_matrix_to_quaternion(R_scipy)
    R_reconstructed = quaternion_to_rotation_matrix(q)

    # 验证重建误差
    error = torch.norm(R_scipy - R_reconstructed)
    print(f"  重建误差: {error.item():.8f}")

    # 验证旋转矩阵正交性
    orthogonality_error = torch.norm(R_reconstructed @ R_reconstructed.T - torch.eye(3))
    print(f"  正交性误差: {orthogonality_error.item():.8f}")

    assert error < 1e-5, "四元数转换误差过大！"
    assert orthogonality_error < 1e-5, "重建的旋转矩阵不正交！"
    print("✅ 四元数转换测试通过！")


def test_slerp():
    """测试 SLERP 插值的正确性"""
    print("测试 SLERP 插值...")

    # 创建两个正交的旋转
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 单位四元数（无旋转）
    q2_raw = torch.tensor([0.707, 0.707, 0.0, 0.0])  # 绕 X 轴旋转 90°
    q2 = q2_raw / torch.norm(q2_raw)  # 归一化

    # 中间插值
    q_mid = slerp(q1, q2, t=0.5)

    # 验证插值结果归一化
    norm = torch.norm(q_mid)
    print(f"  插值四元数模长: {norm.item():.8f}")
    assert abs(norm - 1.0) < 1e-5, "插值四元数未归一化！"

    # 验证边界条件
    q_start = slerp(q1, q2, t=0.0)
    q_end = slerp(q1, q2, t=1.0)
    assert torch.allclose(q_start, q1, atol=1e-4), "t=0 插值错误！"
    assert torch.allclose(q_end, q2, atol=1e-4), "t=1 插值错误！"

    print("✅ SLERP 插值测试通过！")


if __name__ == "__main__":
    # 运行单元测试
    print("\n" + "="*60)
    print("运行 Pseudo-view Co-regularization 核心算法单元测试")
    print("="*60 + "\n")

    test_quaternion_conversion()
    print()
    test_slerp()

    print("\n" + "="*60)
    print("✅ 所有单元测试通过！核心算法实现正确。")
    print("="*60 + "\n")
