"""
Advanced Pseudo-Label Generator for R²-Gaussian
高级伪标签生成器，基于Teacher-Student框架
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from r2_gaussian.dataset.cameras import Camera, PseudoCamera
from r2_gaussian.utils.pseudo_view_utils import FSGSPseudoViewGenerator


@dataclass
class PseudoLabelCandidate:
    """伪标签候选对象"""
    camera: Camera
    quality_score: float
    metrics: Dict[str, float]
    weight: float = 1.0


class QualityEvaluator:
    """伪标签质量评估器"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'render_quality': 0.4,
            'geometric_consistency': 0.3,
            'multi_view_consistency': 0.2,
            'gradient_stability': 0.1
        }
    
    def evaluate_render_quality(self, teacher_model, pseudo_camera) -> float:
        """评估Teacher模型在伪视角的渲染质量"""
        try:
            # 使用Teacher模型渲染伪视角
            with torch.no_grad():
                render_result = teacher_model.render(pseudo_camera)
                rendered_image = render_result['image']
                
                # 计算图像质量指标
                # 1. 方差(避免纯色输出)
                variance_score = torch.var(rendered_image).item()
                
                # 2. 边缘清晰度
                edge_score = self._compute_edge_sharpness(rendered_image)
                
                # 3. 颜色分布合理性
                color_score = self._compute_color_distribution_score(rendered_image)
                
                # 综合渲染质量分数
                quality = (variance_score * 0.3 + edge_score * 0.4 + color_score * 0.3)
                return min(quality, 1.0)
                
        except Exception as e:
            print(f"Render quality evaluation failed: {e}")
            return 0.0
    
    def _compute_edge_sharpness(self, image: torch.Tensor) -> float:
        """计算边缘清晰度"""
        # Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        if len(image.shape) == 3:  # CHW format
            gray = torch.mean(image, dim=0)
        else:
            gray = image
            
        # 计算梯度幅值
        grad_x = torch.nn.functional.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                                           sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = torch.nn.functional.conv2d(gray.unsqueeze(0).unsqueeze(0),
                                           sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        edge_score = torch.mean(gradient_magnitude).item()
        
        return min(edge_score, 1.0)
    
    def _compute_color_distribution_score(self, image: torch.Tensor) -> float:
        """计算颜色分布的合理性"""
        # 检查颜色范围和分布
        if len(image.shape) == 3:  # CHW
            channel_means = torch.mean(image, dim=[1, 2])
            channel_stds = torch.std(image, dim=[1, 2])
        else:
            channel_means = torch.mean(image)
            channel_stds = torch.std(image)
            
        # 评估颜色分布的合理性
        mean_score = 1.0 - torch.mean(torch.abs(channel_means - 0.5))  # 期望在0.5附近
        std_score = torch.mean(channel_stds)  # 期望有适度变化
        
        return (mean_score * 0.6 + std_score * 0.4).item()
    
    def evaluate_geometric_consistency(self, teacher_model, pseudo_camera) -> float:
        """评估几何一致性"""
        try:
            with torch.no_grad():
                # 获取深度图
                depth_map = teacher_model.get_depth_map(pseudo_camera)
                if depth_map is None:
                    return 0.5  # 中等分数
                
                # 计算深度平滑性
                smoothness = self._compute_depth_smoothness(depth_map)
                
                # 计算深度合理性(避免异常值)
                reasonableness = self._compute_depth_reasonableness(depth_map)
                
                return (smoothness * 0.6 + reasonableness * 0.4)
                
        except Exception as e:
            print(f"Geometric consistency evaluation failed: {e}")
            return 0.5
    
    def _compute_depth_smoothness(self, depth_map: torch.Tensor) -> float:
        """计算深度图平滑性"""
        # 计算深度梯度
        grad_x = torch.diff(depth_map, dim=1)
        grad_y = torch.diff(depth_map, dim=0)
        
        # 平滑性 = 1 / (1 + 平均梯度)
        avg_gradient = (torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))) / 2
        smoothness = 1.0 / (1.0 + avg_gradient.item())
        
        return smoothness
    
    def _compute_depth_reasonableness(self, depth_map: torch.Tensor) -> float:
        """计算深度值的合理性"""
        # 检查是否有NaN或Inf
        if torch.any(torch.isnan(depth_map)) or torch.any(torch.isinf(depth_map)):
            return 0.0
        
        # 检查深度范围是否合理
        depth_min, depth_max = torch.min(depth_map), torch.max(depth_map)
        if depth_max - depth_min < 1e-6:  # 深度变化太小
            return 0.3
        
        # 检查深度分布
        depth_std = torch.std(depth_map)
        depth_mean = torch.mean(depth_map)
        
        # 合理的深度应该有适度的变化
        if depth_std / depth_mean < 0.1:  # 变化太小
            return 0.5
        elif depth_std / depth_mean > 2.0:  # 变化太大
            return 0.4
        else:
            return 0.8
    
    def compute_overall_quality(self, metrics: Dict[str, float]) -> float:
        """计算综合质量分数"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                total_score += metrics[metric_name] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class AdvancedPseudoLabelGenerator:
    """高级伪标签生成器"""
    
    def __init__(self, quality_evaluator: QualityEvaluator = None):
        self.base_generator = FSGSPseudoViewGenerator()
        self.quality_evaluator = quality_evaluator or QualityEvaluator()
        
    def spherical_sampling(self, center: np.ndarray, radius: float, n_samples: int) -> List[Camera]:
        """球面均匀采样生成候选视角"""
        cameras = []
        
        # 生成球面上的均匀分布点
        for i in range(n_samples):
            # 使用Fibonacci球面采样
            k = i + 0.5
            y = 1 - 2 * k / n_samples  # y范围[-1, 1]
            radius_at_y = np.sqrt(1 - y**2)
            theta = np.pi * (1 + 5**0.5) * k  # Golden angle
            
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            
            # 相机位置
            position = center + radius * np.array([x, y, z])
            
            # 朝向中心的旋转矩阵
            forward = center - position
            forward = forward / np.linalg.norm(forward)
            
            # 构造旋转矩阵
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                up = np.array([0, 0, 1])
                right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            rotation_matrix = np.column_stack([right, up, -forward])
            
            try:
                pseudo_camera = PseudoCamera(
                    R=rotation_matrix,
                    T=position,
                    FoVx=np.pi/3,  # 60度视角
                    FoVy=np.pi/3,
                    width=512,
                    height=512
                )
                cameras.append(pseudo_camera)
            except Exception as e:
                print(f"Failed to create camera {i}: {e}")
                continue
                
        return cameras
    
    def trajectory_interpolation(self, train_cameras: List[Camera], n_samples: int) -> List[Camera]:
        """基于训练相机轨迹的插值采样"""
        cameras = []
        
        # 获取训练相机的位置和方向
        positions = []
        rotations = []
        
        for cam in train_cameras:
            if hasattr(cam, 'camera_center'):
                pos = cam.camera_center.cpu().numpy()
                rot = cam.world_view_transform[:3, :3].cpu().numpy()
                positions.append(pos)
                rotations.append(rot)
        
        if len(positions) < 2:
            return []
        
        positions = np.array(positions)
        
        # 生成轨迹插值点
        for i in range(n_samples):
            t = i / max(1, n_samples - 1)
            
            # 位置插值(使用样条插值)
            if len(positions) >= 3:
                # 使用二次插值
                idx = int(t * (len(positions) - 1))
                local_t = t * (len(positions) - 1) - idx
                
                if idx < len(positions) - 1:
                    pos = positions[idx] * (1 - local_t) + positions[idx + 1] * local_t
                else:
                    pos = positions[-1]
            else:
                # 简单线性插值
                pos = positions[0] * (1 - t) + positions[1] * t
            
            # 添加随机扰动
            pos += np.random.normal(0, 0.1, 3)
            
            # 随机旋转(基于训练相机的平均方向)
            base_rotation = rotations[0] if rotations else np.eye(3)
            
            try:
                pseudo_camera = PseudoCamera(
                    R=base_rotation,
                    T=pos,
                    FoVx=train_cameras[0].FoVx if train_cameras else np.pi/3,
                    FoVy=train_cameras[0].FoVy if train_cameras else np.pi/3,
                    width=train_cameras[0].image_width if train_cameras else 512,
                    height=train_cameras[0].image_height if train_cameras else 512
                )
                cameras.append(pseudo_camera)
            except Exception as e:
                print(f"Failed to create interpolated camera {i}: {e}")
                continue
        
        return cameras
    
    def local_perturbation_sampling(self, train_cameras: List[Camera], n_samples: int,
                                   position_std: float = 0.2, rotation_std: float = 0.1) -> List[Camera]:
        """在训练相机附近进行局部扰动采样"""
        cameras = []
        
        for i in range(n_samples):
            # 随机选择一个训练相机作为基础
            base_camera = random.choice(train_cameras)
            
            try:
                # 获取基础相机参数
                base_pos = base_camera.camera_center.cpu().numpy()
                base_rot = base_camera.world_view_transform[:3, :3].cpu().numpy()
                
                # 位置扰动
                pos_noise = np.random.normal(0, position_std, 3)
                new_pos = base_pos + pos_noise
                
                # 旋转扰动
                rotation_noise = np.random.normal(0, rotation_std, 3)
                rot_perturbation = R.from_rotvec(rotation_noise).as_matrix()
                new_rot = rot_perturbation @ base_rot
                
                pseudo_camera = PseudoCamera(
                    R=new_rot,
                    T=new_pos,
                    FoVx=base_camera.FoVx,
                    FoVy=base_camera.FoVy,
                    width=base_camera.image_width,
                    height=base_camera.image_height
                )
                cameras.append(pseudo_camera)
                
            except Exception as e:
                print(f"Failed to create perturbed camera {i}: {e}")
                continue
                
        return cameras
    
    def generate_candidates(self, train_cameras: List[Camera], n_total: int = 200) -> List[Camera]:
        """生成所有候选伪视角"""
        candidates = []
        
        # 计算场景中心和尺度
        if train_cameras:
            positions = []
            for cam in train_cameras:
                if hasattr(cam, 'camera_center'):
                    positions.append(cam.camera_center.cpu().numpy())
            
            if positions:
                positions = np.array(positions)
                scene_center = np.mean(positions, axis=0)
                scene_scale = np.max(np.linalg.norm(positions - scene_center, axis=1)) * 2
            else:
                scene_center = np.array([0, 0, 0])
                scene_scale = 5.0
        else:
            scene_center = np.array([0, 0, 0])
            scene_scale = 5.0
        
        # 1. 球面采样 (40%)
        n_sphere = int(n_total * 0.4)
        sphere_cameras = self.spherical_sampling(scene_center, scene_scale, n_sphere)
        candidates.extend(sphere_cameras)
        
        # 2. 轨迹插值 (30%)
        n_trajectory = int(n_total * 0.3)
        trajectory_cameras = self.trajectory_interpolation(train_cameras, n_trajectory)
        candidates.extend(trajectory_cameras)
        
        # 3. 局部扰动 (30%)
        n_perturbation = n_total - len(candidates)
        perturbation_cameras = self.local_perturbation_sampling(train_cameras, n_perturbation)
        candidates.extend(perturbation_cameras)
        
        print(f"Generated {len(candidates)} pseudo-label candidates")
        print(f"  - Spherical: {len(sphere_cameras)}")
        print(f"  - Trajectory: {len(trajectory_cameras)}")
        print(f"  - Perturbation: {len(perturbation_cameras)}")
        
        return candidates
    
    def evaluate_and_select(self, candidates: List[Camera], teacher_model,
                          quality_threshold: float = 0.6,
                          max_selected: int = 50) -> List[PseudoLabelCandidate]:
        """评估候选伪视角并筛选高质量样本"""
        
        evaluated_candidates = []
        
        print(f"Evaluating {len(candidates)} candidates...")
        
        for i, camera in enumerate(candidates):
            try:
                # 评估各项质量指标
                metrics = {}
                
                # 1. 渲染质量
                metrics['render_quality'] = self.quality_evaluator.evaluate_render_quality(
                    teacher_model, camera)
                
                # 2. 几何一致性
                metrics['geometric_consistency'] = self.quality_evaluator.evaluate_geometric_consistency(
                    teacher_model, camera)
                
                # 3. 多视角一致性(简化版)
                metrics['multi_view_consistency'] = 0.7  # 默认值，后续可以实现
                
                # 4. 梯度稳定性(简化版)
                metrics['gradient_stability'] = 0.8  # 默认值，后续可以实现
                
                # 计算综合质量分数
                quality_score = self.quality_evaluator.compute_overall_quality(metrics)
                
                if quality_score >= quality_threshold:
                    candidate = PseudoLabelCandidate(
                        camera=camera,
                        quality_score=quality_score,
                        metrics=metrics
                    )
                    evaluated_candidates.append(candidate)
                
                if (i + 1) % 20 == 0:
                    print(f"  Evaluated {i + 1}/{len(candidates)} candidates")
                    
            except Exception as e:
                print(f"Failed to evaluate candidate {i}: {e}")
                continue
        
        # 按质量分数排序
        evaluated_candidates.sort(key=lambda x: x.quality_score, reverse=True)
        
        # 选择top candidates
        selected = evaluated_candidates[:max_selected]
        
        print(f"Selected {len(selected)} high-quality pseudo-labels:")
        print(f"  Quality range: {selected[0].quality_score:.3f} - {selected[-1].quality_score:.3f}")
        
        return selected


def test_pseudo_label_generator():
    """测试伪标签生成器"""
    print("Testing Advanced Pseudo-Label Generator...")
    
    # 创建生成器
    generator = AdvancedPseudoLabelGenerator()
    
    # 模拟训练相机
    train_cameras = []
    for i in range(3):
        # 简单的测试相机
        pos = np.array([i*2, 0, 5])
        rot = np.eye(3)
        
        try:
            camera = PseudoCamera(
                R=rot,
                T=pos,
                FoVx=np.pi/3,
                FoVy=np.pi/3,
                width=512,
                height=512
            )
            train_cameras.append(camera)
        except:
            pass
    
    # 生成候选
    candidates = generator.generate_candidates(train_cameras, n_total=50)
    print(f"Generated {len(candidates)} candidates for testing")
    
    return candidates


if __name__ == "__main__":
    candidates = test_pseudo_label_generator()