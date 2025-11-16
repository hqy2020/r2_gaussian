"""
FSGS风格伪视角生成工具
基于FSGS论文实现智能伪视角生成算法

参考论文: FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting
核心思想: 通过欧几里得空间插值生成几何一致的伪视角
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import random
from r2_gaussian.dataset.cameras import Camera


class FSGSPseudoViewGenerator:
    """FSGS风格伪视角生成器"""
    
    def __init__(self, noise_std: float = 0.05, min_distance_threshold: float = 0.1):
        """
        初始化伪视角生成器
        
        Args:
            noise_std: 3DoF位置噪声标准差 (参考FSGS论文 Eq.5)
            min_distance_threshold: 最小相机间距阈值
        """
        self.noise_std = noise_std
        self.min_distance_threshold = min_distance_threshold
    
    def find_closest_camera_pairs(self, train_cameras: dict) -> List[Tuple[Camera, Camera]]:
        """
        找到训练相机中最近的相机对
        
        Args:
            train_cameras: 训练相机字典 {scale: [cameras]}
        
        Returns:
            相机对列表
        """
        camera_pairs = []
        cameras_list = []
        
        # 收集所有相机 (兼容字典和列表格式)
        if isinstance(train_cameras, dict):
            for scale, cameras in train_cameras.items():
                cameras_list.extend(cameras)
        elif isinstance(train_cameras, list):
            cameras_list = train_cameras
        else:
            print(f"Warning: Unknown train_cameras type: {type(train_cameras)}")
            return []
        
        # 计算所有相机对的欧几里得距离
        for i in range(len(cameras_list)):
            for j in range(i + 1, len(cameras_list)):
                cam1, cam2 = cameras_list[i], cameras_list[j]
                
                # 计算相机中心的欧几里得距离
                center1 = cam1.camera_center.cpu().numpy()
                center2 = cam2.camera_center.cpu().numpy()
                distance = np.linalg.norm(center1 - center2)
                
                # 过滤距离过近的相机对
                if distance > self.min_distance_threshold:
                    camera_pairs.append((cam1, cam2, distance))
        
        # 按距离排序，选择最近的相机对
        camera_pairs.sort(key=lambda x: x[2])
        return [(pair[0], pair[1]) for pair in camera_pairs]
    
    def interpolate_camera_poses(self, cam1: Camera, cam2: Camera, 
                                noise_std: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        在两个相机之间进行欧几里得空间插值
        
        Args:
            cam1, cam2: 输入相机对
            noise_std: 噪声标准差，如果为None则使用默认值
        
        Returns:
            (position, quaternion): 插值后的位置和四元数
        """
        if noise_std is None:
            noise_std = self.noise_std
        
        # 获取相机中心位置
        center1 = cam1.camera_center.cpu().numpy()[:3]
        center2 = cam2.camera_center.cpu().numpy()[:3]
        
        # 计算平均位置 (FSGS Eq.5 中的 t)
        avg_position = (center1 + center2) / 2.0
        
        # 添加3DoF高斯噪声 (FSGS Eq.5: ε ~ N(0, δ))
        noise = np.random.normal(0, noise_std, 3)
        noisy_position = avg_position + noise
        
        # 计算旋转四元数的平均值
        # 从world_view_transform提取旋转部分
        R1 = cam1.world_view_transform[:3, :3].cpu().numpy()
        R2 = cam2.world_view_transform[:3, :3].cpu().numpy()
        
        # 将旋转矩阵转换为四元数并平均
        q1 = self._rotation_matrix_to_quaternion(R1)
        q2 = self._rotation_matrix_to_quaternion(R2)
        avg_quaternion = self._slerp_quaternion(q1, q2, t=0.5)
        
        return noisy_position, avg_quaternion
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵转四元数"""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])
    
    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """四元数球面线性插值 (SLERP)"""
        # 确保四元数归一化
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算点积
        dot = np.dot(q1, q2)
        
        # 如果点积为负，翻转一个四元数以选择较短路径
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # 如果四元数几乎相同，使用线性插值
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # 球面线性插值
        theta_0 = np.arccos(abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        qx, qy, qz, qw = q
        
        # 计算旋转矩阵元素
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def create_pseudo_camera(self, cam1: Camera, cam2: Camera, 
                           uid: int, device: str = "cuda") -> Camera:
        """
        创建插值后的伪相机 - 使用简化的PseudoCamera类
        
        Args:
            cam1, cam2: 输入相机对
            uid: 相机唯一标识符
            device: 设备类型
        
        Returns:
            伪相机对象
        """
        # 执行插值
        position, quaternion = self.interpolate_camera_poses(cam1, cam2)
        
        # 构建变换矩阵
        R = self._quaternion_to_rotation_matrix(quaternion)
        
        # 使用PseudoCamera类来避免参数问题
        from r2_gaussian.dataset.cameras import PseudoCamera
        
        pseudo_camera = PseudoCamera(
            R=R,
            T=position,
            FoVx=cam1.FoVx,
            FoVy=cam1.FoVy,
            width=cam1.image_width,
            height=cam1.image_height,
        )
        
        return pseudo_camera
    
    def generate_pseudo_cameras(self, train_cameras: dict, num_views: int = 10,
                              device: str = "cuda") -> List[Camera]:
        """
        生成FSGS风格的伪相机阵列
        
        Args:
            train_cameras: 训练相机字典
            num_views: 生成的伪视角数量
            device: 设备类型
        
        Returns:
            伪相机列表
        """
        # 找到最近的相机对
        camera_pairs = self.find_closest_camera_pairs(train_cameras)
        
        if len(camera_pairs) == 0:
            print("Warning: No valid camera pairs found for pseudo view generation")
            return []
        
        pseudo_cameras = []
        
        # 生成指定数量的伪视角
        for i in range(num_views):
            # 随机选择相机对 (也可以按距离顺序选择)
            if len(camera_pairs) > 0:
                cam1, cam2 = random.choice(camera_pairs)
                
                try:
                    pseudo_cam = self.create_pseudo_camera(cam1, cam2, uid=10000 + i, device=device)
                    pseudo_cameras.append(pseudo_cam)
                except Exception as e:
                    print(f"Warning: Failed to create pseudo camera {i}: {e}")
                    continue
        
        print(f"Successfully generated {len(pseudo_cameras)} FSGS-style pseudo cameras")
        return pseudo_cameras


def validate_pseudo_camera(pseudo_cam: Camera, train_cameras: dict, 
                         max_distance_ratio: float = 2.0) -> bool:
    """
    验证伪相机的几何合理性
    
    Args:
        pseudo_cam: 伪相机
        train_cameras: 训练相机集合
        max_distance_ratio: 最大距离比例阈值
    
    Returns:
        是否几何合理
    """
    pseudo_center = pseudo_cam.camera_center.cpu().numpy()
    
    # 计算到所有训练相机的距离
    min_dist = float('inf')
    max_dist = 0.0
    
    # 兼容字典和列表格式
    if isinstance(train_cameras, dict):
        for scale, cameras in train_cameras.items():
            for cam in cameras:
                cam_center = cam.camera_center.cpu().numpy()
                dist = np.linalg.norm(pseudo_center - cam_center)
                min_dist = min(min_dist, dist)
                max_dist = max(max_dist, dist)
    elif isinstance(train_cameras, list):
        for cam in train_cameras:
            cam_center = cam.camera_center.cpu().numpy()
            dist = np.linalg.norm(pseudo_center - cam_center)
            min_dist = min(min_dist, dist)
            max_dist = max(max_dist, dist)
    
    # 检查距离比例是否合理
    if min_dist > 0 and max_dist / min_dist > max_distance_ratio:
        return False
    
    return True


# FSGS伪标签工具函数
def create_fsgs_pseudo_cameras(scene, num_additional_views: int = 10,
                              noise_std: float = 0.05, device: str = "cuda") -> List[Camera]:
    """
    为场景创建FSGS风格伪相机的便捷函数
    
    Args:
        scene: Scene对象
        num_additional_views: 伪视角数量
        noise_std: 噪声标准差
        device: 设备类型
    
    Returns:
        伪相机列表
    """
    generator = FSGSPseudoViewGenerator(noise_std=noise_std)
    return generator.generate_pseudo_cameras(
        scene.train_cameras, 
        num_views=num_additional_views,
        device=device
    )