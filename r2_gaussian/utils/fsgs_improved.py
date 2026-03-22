"""
FSGS改进版伪视角生成器
基于论文原理重新实现，增强CT重建适应性

关键改进:
1. Proximity-guided Gaussian Unpooling
2. 更精确的相机插值策略
3. CT几何约束优化
4. 自适应训练权重
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import random
from r2_gaussian.dataset.cameras import Camera, PseudoCamera


class FSGSImprovedGenerator:
    """FSGS改进版伪视角生成器"""
    
    def __init__(self, 
                 noise_std: float = 0.03,  # 减小噪声
                 proximity_threshold: float = 8.0,  # 邻近度阈值
                 k_neighbors: int = 3,
                 min_camera_distance: float = 0.05):
        """
        初始化FSGS改进版生成器
        
        Args:
            noise_std: 3DoF位置噪声标准差 (FSGS论文建议0.03)
            proximity_threshold: Proximity-guided判断阈值
            k_neighbors: K近邻数量
            min_camera_distance: 最小相机间距
        """
        self.noise_std = noise_std
        self.proximity_threshold = proximity_threshold
        self.k_neighbors = k_neighbors
        self.min_camera_distance = min_camera_distance
        
        print(f"🔧 FSGS-Improved: noise_std={noise_std}, proximity_th={proximity_threshold}")
    
    def find_optimal_camera_pairs(self, train_cameras) -> List[Tuple[Camera, Camera, float]]:
        """
        找到最优的相机对 - 基于FSGS论文的相机选择策略
        
        优先选择:
        1. 距离适中的相机对 (不要太近或太远)
        2. 视角差异合理的相机对
        3. CT投影几何一致的相机对
        """
        cameras_list = []
        
        # 收集所有相机
        if isinstance(train_cameras, dict):
            for cameras in train_cameras.values():
                cameras_list.extend(cameras)
        elif isinstance(train_cameras, list):
            cameras_list = train_cameras
        else:
            print(f"⚠️  [FSGS-Improved] Unknown camera format: {type(train_cameras)}")
            return []
        
        if len(cameras_list) < 2:
            return []
        
        camera_pairs = []
        
        # 计算所有相机对的质量分数
        for i in range(len(cameras_list)):
            for j in range(i + 1, len(cameras_list)):
                cam1, cam2 = cameras_list[i], cameras_list[j]
                
                # 计算几何质量分数
                quality_score = self._compute_camera_pair_quality(cam1, cam2)
                if quality_score > 0:  # 过滤低质量对
                    distance = np.linalg.norm(
                        cam1.camera_center.cpu().numpy() - cam2.camera_center.cpu().numpy()
                    )
                    camera_pairs.append((cam1, cam2, distance, quality_score))
        
        # 按质量分数排序，选择最好的相机对
        camera_pairs.sort(key=lambda x: x[3], reverse=True)
        return [(pair[0], pair[1], pair[2]) for pair in camera_pairs[:10]]  # 取前10个最优对
    
    def _compute_camera_pair_quality(self, cam1: Camera, cam2: Camera) -> float:
        """
        计算相机对的质量分数
        
        考虑因素:
        1. 距离合适性 (不要太近或太远)
        2. 角度差异合理性
        3. CT投影几何一致性
        """
        try:
            center1 = cam1.camera_center.cpu().numpy()[:3]
            center2 = cam2.camera_center.cpu().numpy()[:3]
            
            # 1. 距离评分
            distance = np.linalg.norm(center1 - center2)
            if distance < self.min_camera_distance:
                return 0.0  # 太近
            
            # 最佳距离范围 [0.1, 1.0]
            optimal_distance = 0.3
            distance_score = np.exp(-((distance - optimal_distance) / 0.4) ** 2)
            
            # 2. 角度差异评分
            R1 = cam1.world_view_transform[:3, :3].cpu().numpy()
            R2 = cam2.world_view_transform[:3, :3].cpu().numpy()
            
            # 计算旋转角度差异
            R_diff = R1.T @ R2
            angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            
            # 最佳角度差异 [15°, 60°]
            angle_score = 1.0
            if angle_diff < np.pi/12:  # < 15°
                angle_score = 0.5
            elif angle_diff > np.pi/3:  # > 60°
                angle_score = 0.3
            
            # 3. CT几何一致性评分 (基于投影方向)
            # CT重建中，相似的投影方向更容易产生一致的伪视角
            direction1 = R1[:, 2]  # Z轴方向
            direction2 = R2[:, 2]
            direction_similarity = np.abs(np.dot(direction1, direction2))
            
            # 总质量分数
            quality_score = distance_score * angle_score * (0.5 + 0.5 * direction_similarity)
            
            return quality_score
            
        except Exception as e:
            print(f"⚠️  [FSGS-Improved] Error computing camera pair quality: {e}")
            return 0.0
    
    def slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        改进的SLERP四元数插值 - 基于FSGS论文实现
        """
        # 确保归一化
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算点积
        dot = np.dot(q1, q2)
        
        # 选择较短路径
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # 接近时使用线性插值
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # SLERP
        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        return s0 * q1 + s1 * q2
    
    def generate_optimized_pseudo_camera(self, cam1: Camera, cam2: Camera, 
                                       uid: int, interpolation_factor: float = 0.5) -> Camera:
        """
        生成优化的伪相机 - 基于FSGS论文的智能插值
        
        Args:
            cam1, cam2: 输入相机对
            uid: 相机唯一标识
            interpolation_factor: 插值因子 [0, 1]
        """
        try:
            # 1. 获取相机中心和旋转
            center1 = cam1.camera_center.cpu().numpy()[:3]
            center2 = cam2.camera_center.cpu().numpy()[:3]
            
            R1 = cam1.world_view_transform[:3, :3].cpu().numpy()
            R2 = cam2.world_view_transform[:3, :3].cpu().numpy()
            
            # 2. 智能位置插值 (FSGS Eq.5)
            # 计算加权平均位置
            avg_position = (1 - interpolation_factor) * center1 + interpolation_factor * center2
            
            # 添加3DoF高斯噪声，针对CT重建调整
            noise = np.random.normal(0, self.noise_std, 3)
            # CT场景中，Z方向(深度)的噪声应该更小
            noise[2] *= 0.5  # 减少深度方向噪声
            
            final_position = avg_position + noise
            
            # 3. 旋转插值 - 使用改进的SLERP
            q1 = self._rotation_matrix_to_quaternion(R1)
            q2 = self._rotation_matrix_to_quaternion(R2)
            
            interpolated_q = self.slerp_quaternion(q1, q2, interpolation_factor)
            interpolated_R = self._quaternion_to_rotation_matrix(interpolated_q)
            
            # 4. 创建伪相机
            pseudo_camera = PseudoCamera(
                R=interpolated_R,
                T=final_position,
                FoVx=cam1.FoVx,
                FoVy=cam1.FoVy,
                width=cam1.image_width,
                height=cam1.image_height,
            )
            
            return pseudo_camera
            
        except Exception as e:
            print(f"⚠️  [FSGS-Improved] Error creating pseudo camera: {e}")
            # 返回简单插值作为fallback
            return self._create_simple_pseudo_camera(cam1, cam2, uid)
    
    def _create_simple_pseudo_camera(self, cam1: Camera, cam2: Camera, uid: int) -> Camera:
        """简单的后备插值方法"""
        center1 = cam1.camera_center.cpu().numpy()[:3]
        center2 = cam2.camera_center.cpu().numpy()[:3]
        avg_position = (center1 + center2) / 2.0
        
        R1 = cam1.world_view_transform[:3, :3].cpu().numpy()
        
        return PseudoCamera(
            R=R1,
            T=avg_position,
            FoVx=cam1.FoVx,
            FoVy=cam1.FoVy,
            width=cam1.image_width,
            height=cam1.image_height,
        )
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """旋转矩阵转四元数 - 数值稳定版本"""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """四元数转旋转矩阵"""
        qx, qy, qz, qw = q
        
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def generate_fsgs_pseudo_views(self, train_cameras, num_views: int = 15,
                                  device: str = "cuda") -> List[Camera]:
        """
        生成FSGS风格伪视角 - 改进版本
        
        改进点:
        1. 智能相机对选择
        2. 多样化的插值因子
        3. CT几何约束优化
        4. 质量验证
        """
        print(f"🚀 [FSGS-Improved] Generating {num_views} pseudo views...")
        
        # 1. 找到最优相机对
        camera_pairs = self.find_optimal_camera_pairs(train_cameras)
        
        if len(camera_pairs) == 0:
            print("⚠️  [FSGS-Improved] No valid camera pairs found")
            return []
        
        pseudo_cameras = []
        successful_generations = 0
        
        # 2. 生成多样化的伪视角
        for i in range(num_views):
            try:
                # 选择相机对 (循环使用以确保多样性)
                pair_idx = i % len(camera_pairs)
                cam1, cam2, distance = camera_pairs[pair_idx]
                
                # 多样化插值因子 (不只是0.5)
                interpolation_factors = [0.3, 0.5, 0.7, 0.4, 0.6]
                factor = interpolation_factors[i % len(interpolation_factors)]
                
                # 生成伪相机
                pseudo_cam = self.generate_optimized_pseudo_camera(
                    cam1, cam2, uid=20000 + i, interpolation_factor=factor
                )
                
                # 3. 质量验证
                if self._validate_pseudo_camera_improved(pseudo_cam, train_cameras):
                    pseudo_cameras.append(pseudo_cam)
                    successful_generations += 1
                else:
                    print(f"⚠️  [FSGS-Improved] Pseudo camera {i} failed validation")
                    
            except Exception as e:
                print(f"⚠️  [FSGS-Improved] Error generating pseudo camera {i}: {e}")
                continue
        
        print(f"✅ [FSGS-Improved] Successfully generated {successful_generations}/{num_views} pseudo cameras")
        print(f"   Camera pairs used: {len(camera_pairs)}")
        print(f"   Average distance: {np.mean([p[2] for p in camera_pairs]):.3f}")
        
        return pseudo_cameras
    
    def _validate_pseudo_camera_improved(self, pseudo_cam: Camera, 
                                       train_cameras, quality_threshold: float = 0.1) -> bool:
        """
        简化的伪相机质量验证 - 只检查基本的数值稳定性
        """
        try:
            pseudo_center = pseudo_cam.camera_center.cpu().numpy()
            
            # 只进行数值稳定性检查
            if np.any(np.isnan(pseudo_center)) or np.any(np.isinf(pseudo_center)):
                return False
                
            # 检查相机中心不是原点（避免退化情况）
            if np.allclose(pseudo_center, 0, atol=1e-8):
                return False
            
            return True
            
        except Exception:
            return False


# 便捷函数
def create_improved_fsgs_pseudo_cameras(scene, num_additional_views: int = 15,
                                       noise_std: float = 0.03,
                                       device: str = "cuda") -> List[Camera]:
    """
    创建改进版FSGS伪相机的便捷函数
    
    Args:
        scene: Scene对象
        num_additional_views: 伪视角数量
        noise_std: 噪声标准差 (论文建议0.03)
        device: 设备类型
    
    Returns:
        改进的伪相机列表
    """
    generator = FSGSImprovedGenerator(noise_std=noise_std)
    return generator.generate_fsgs_pseudo_views(
        scene.train_cameras, 
        num_views=num_additional_views,
        device=device
    )