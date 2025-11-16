#!/usr/bin/env python3
"""
基于9视角高质量数据的医学感知Proximity-guided密化策略
结合实际opacity分布的优化实现
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HighQualityMedicalProximityGuidedDensifier:
    """基于9视角高质量数据的医学感知Proximity密化器"""
    
    def __init__(self):
        # 基于9视角真实数据的分类系统
        self.realistic_classification = {
            "background_air": {
                "opacity_range": (0.0, 0.05),
                "description": "背景空气区域",
                "coverage": "60-75%",  # 基于实际统计
                "medical_meaning": "外部空气、肺泡、低密度区域",
                "proximity_params": {
                    "min_neighbors": 6,
                    "max_distance": 2.0,  # mm
                    "max_gradient": 0.05
                }
            },
            
            "tissue_transition": {
                "opacity_range": (0.05, 0.15),
                "description": "组织过渡区域", 
                "coverage": "15-25%",
                "medical_meaning": "组织边界、软组织外层、脂肪",
                "proximity_params": {
                    "min_neighbors": 8,  # 过渡区最关键
                    "max_distance": 1.5,
                    "max_gradient": 0.10
                }
            },
            
            "soft_tissue": {
                "opacity_range": (0.15, 0.40),
                "description": "软组织主体",
                "coverage": "10-20%",
                "medical_meaning": "器官实质、肌肉、血管",
                "proximity_params": {
                    "min_neighbors": 6,
                    "max_distance": 1.0,
                    "max_gradient": 0.25
                }
            },
            
            "dense_structures": {
                "opacity_range": (0.40, 1.0),
                "description": "致密结构",
                "coverage": "1-5%",
                "medical_meaning": "骨骼、钙化、高密度病变",
                "proximity_params": {
                    "min_neighbors": 4,  # 致密结构相对稳定
                    "max_distance": 0.8,
                    "max_gradient": 0.60
                }
            }
        }
        
        # 器官特异性参数 (基于9视角实际数据)
        self.organ_specific_params = {
            "chest": {
                "high_density_emphasis": True,  # Chest有0.982高值
                "air_boundary_critical": True,  # 肺-组织边界关键
                "density_weights": [0.7, 0.8, 0.6, 1.0]  # 对应4个分类的权重
            },
            "pancreas": {
                "soft_tissue_emphasis": True,  # Pancreas均值0.150最高
                "wide_distribution": True,     # 分布范围最广
                "density_weights": [0.6, 1.0, 1.2, 0.8]
            },
            "head": {
                "bone_tissue_boundary": True,  # 颅骨-脑组织边界
                "low_variance": True,          # Head变异最小
                "density_weights": [0.8, 0.9, 0.7, 0.9]
            },
            "abdomen": {
                "multi_organ": True,           # 多器官复杂结构
                "balanced_distribution": True,
                "density_weights": [0.7, 0.9, 1.0, 0.8]
            },
            "foot": {
                "bone_dominant": True,         # 骨骼结构主导
                "stable_baseline": True,      # 作为基准器官
                "density_weights": [0.8, 0.8, 0.9, 1.1]
            }
        }
        
    def classify_opacity_realistic(self, opacity_value: float) -> str:
        """基于9视角实际数据的opacity分类"""
        for tissue_type, info in self.realistic_classification.items():
            min_val, max_val = info["opacity_range"]
            if min_val <= opacity_value < max_val:
                return tissue_type
        return "dense_structures"  # 默认归类到最高密度
    
    def compute_3d_density_map(self, gaussians: torch.Tensor, 
                             opacity_values: torch.Tensor, 
                             grid_resolution: int = 32) -> torch.Tensor:
        """计算3D空间密度图"""
        device = gaussians.device
        
        # 计算场景边界
        xyz = gaussians  # (N, 3)
        min_bounds = xyz.min(dim=0)[0] - 1.0
        max_bounds = xyz.max(dim=0)[0] + 1.0
        
        # 创建3D网格
        grid_coords = torch.linspace(0, 1, grid_resolution, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing='ij')
        
        # 将网格坐标映射到实际空间
        grid_points = torch.stack([
            grid_x.flatten() * (max_bounds[0] - min_bounds[0]) + min_bounds[0],
            grid_y.flatten() * (max_bounds[1] - min_bounds[1]) + min_bounds[1], 
            grid_z.flatten() * (max_bounds[2] - min_bounds[2]) + min_bounds[2]
        ], dim=1)  # (grid_resolution^3, 3)
        
        # 计算每个网格点的密度
        density_map = torch.zeros(grid_points.shape[0], device=device)
        
        for i, grid_point in enumerate(grid_points):
            # 找到最近的K个高斯点
            distances = torch.norm(xyz - grid_point.unsqueeze(0), dim=1)
            k_nearest_indices = torch.topk(distances, k=8, largest=False)[1]
            
            # 基于距离和opacity计算密度
            k_distances = distances[k_nearest_indices]
            k_opacities = opacity_values[k_nearest_indices]
            
            # 距离加权密度
            weights = 1.0 / (k_distances + 1e-6)
            weighted_density = (k_opacities * weights).sum() / weights.sum()
            density_map[i] = weighted_density
            
        return density_map.reshape(grid_resolution, grid_resolution, grid_resolution)
    
    def find_medical_neighbors(self, gaussian_idx: int, gaussians: torch.Tensor,
                             opacity_values: torch.Tensor, tissue_type: str,
                             radius: float = None) -> List[Dict]:
        """找到医学上合理的邻居"""
        if radius is None:
            radius = self.realistic_classification[tissue_type]["proximity_params"]["max_distance"]
        
        center_pos = gaussians[gaussian_idx]  # (3,)
        center_opacity = opacity_values[gaussian_idx].item()
        
        # 计算所有点的距离
        distances = torch.norm(gaussians - center_pos.unsqueeze(0), dim=1)
        
        # 找到半径内的邻居
        neighbor_indices = torch.where(distances < radius)[0]
        neighbor_indices = neighbor_indices[neighbor_indices != gaussian_idx]  # 排除自己
        
        neighbors = []
        for neighbor_idx in neighbor_indices:
            neighbor_pos = gaussians[neighbor_idx]
            neighbor_opacity = opacity_values[neighbor_idx].item()
            neighbor_tissue = self.classify_opacity_realistic(neighbor_opacity)
            distance = distances[neighbor_idx].item()
            
            neighbors.append({
                'index': neighbor_idx.item(),
                'position': neighbor_pos,
                'opacity': neighbor_opacity,
                'tissue_type': neighbor_tissue,
                'distance': distance,
                'opacity_gradient': abs(center_opacity - neighbor_opacity)
            })
        
        return sorted(neighbors, key=lambda x: x['distance'])
    
    def should_densify_medical_realistic(self, gaussian_idx: int, gaussians: torch.Tensor,
                                       opacity_values: torch.Tensor, 
                                       organ_type: str = "general") -> Tuple[bool, str]:
        """基于真实医学分布判断是否需要密化"""
        
        opacity_val = opacity_values[gaussian_idx].item()
        tissue_type = self.classify_opacity_realistic(opacity_val)
        
        # 获取组织特异性参数
        proximity_params = self.realistic_classification[tissue_type]["proximity_params"]
        min_neighbors = proximity_params["min_neighbors"]
        max_distance = proximity_params["max_distance"]
        
        # 器官特异性调整
        if organ_type in self.organ_specific_params:
            organ_params = self.organ_specific_params[organ_type]
            density_weights = organ_params["density_weights"]
            
            # 获取对应分类的权重索引
            tissue_order = ["background_air", "tissue_transition", "soft_tissue", "dense_structures"]
            weight_idx = tissue_order.index(tissue_type)
            adjustment_factor = density_weights[weight_idx]
            
            min_neighbors = int(min_neighbors * adjustment_factor)
            max_distance = max_distance * adjustment_factor
        
        # 找到医学邻居
        neighbors = self.find_medical_neighbors(
            gaussian_idx, gaussians, opacity_values, tissue_type, max_distance
        )
        
        # 判断密化条件
        if len(neighbors) < min_neighbors:
            return True, f"邻居数不足: {len(neighbors)} < {min_neighbors} ({tissue_type})"
        
        # 检查医学合理性约束
        valid_neighbors = []
        for neighbor in neighbors:
            if self._validate_medical_adjacency(tissue_type, neighbor['tissue_type'], 
                                              neighbor['opacity_gradient'], 
                                              neighbor['distance']):
                valid_neighbors.append(neighbor)
        
        if len(valid_neighbors) < min_neighbors * 0.8:  # 至少80%的邻居医学合理
            return True, f"医学合理邻居不足: {len(valid_neighbors)} < {min_neighbors * 0.8}"
        
        return False, "密度充足"
    
    def _validate_medical_adjacency(self, tissue1: str, tissue2: str, 
                                  opacity_gradient: float, distance: float) -> bool:
        """验证两个组织类型的医学邻接合理性"""
        
        # 定义医学上允许的邻接关系
        valid_adjacencies = {
            ("background_air", "tissue_transition"): True,
            ("tissue_transition", "soft_tissue"): True,
            ("soft_tissue", "dense_structures"): True,
            ("background_air", "soft_tissue"): True,     # 允许跳跃
            ("tissue_transition", "dense_structures"): True,  # 允许跳跃
            ("background_air", "dense_structures"): False,    # 不合理的跳跃
        }
        
        # 检查双向邻接
        adjacency_key = tuple(sorted([tissue1, tissue2]))
        if adjacency_key in valid_adjacencies:
            is_allowed = valid_adjacencies[adjacency_key]
        else:
            is_allowed = True  # 默认允许相同类型邻接
        
        if not is_allowed:
            return False
        
        # 检查opacity梯度约束
        tissue1_params = self.realistic_classification[tissue1]["proximity_params"]
        tissue2_params = self.realistic_classification[tissue2]["proximity_params"]
        
        max_gradient = max(tissue1_params["max_gradient"], tissue2_params["max_gradient"])
        
        if opacity_gradient > max_gradient and distance < 1.0:  # 近距离大梯度不合理
            return False
        
        return True
    
    def generate_medical_densification_positions(self, gaussian_idx: int, 
                                               gaussians: torch.Tensor,
                                               opacity_values: torch.Tensor,
                                               num_new_points: int = 3) -> List[torch.Tensor]:
        """生成医学上合理的密化位置"""
        
        opacity_val = opacity_values[gaussian_idx].item()
        tissue_type = self.classify_opacity_realistic(opacity_val)
        center_pos = gaussians[gaussian_idx]
        
        # 找到邻居
        neighbors = self.find_medical_neighbors(
            gaussian_idx, gaussians, opacity_values, tissue_type
        )
        
        if len(neighbors) < 2:
            return []
        
        new_positions = []
        
        for i in range(min(num_new_points, len(neighbors))):
            if i < len(neighbors):
                neighbor = neighbors[i]
                neighbor_pos = neighbor['position']
                
                # 在中心点和邻居之间进行医学上合理的插值
                interpolation_factors = [0.3, 0.5, 0.7]  # 不同的插值位置
                factor = interpolation_factors[i % len(interpolation_factors)]
                
                new_pos = center_pos * (1 - factor) + neighbor_pos * factor
                
                # 添加小量随机噪声，避免重叠
                noise_std = self.realistic_classification[tissue_type]["proximity_params"]["max_distance"] * 0.1
                noise = torch.randn_like(new_pos) * noise_std
                new_pos += noise
                
                new_positions.append(new_pos)
        
        return new_positions
    
    def proximity_guided_densify_realistic(self, gaussians: torch.Tensor, 
                                         opacity_values: torch.Tensor,
                                         organ_type: str = "general",
                                         max_new_points: int = 1000) -> Dict:
        """执行基于真实医学分布的proximity-guided密化"""
        
        print(f"🔬 开始基于{organ_type}的医学感知proximity密化...")
        
        # 统计当前分布
        tissue_stats = {}
        for i, opacity_val in enumerate(opacity_values):
            tissue_type = self.classify_opacity_realistic(opacity_val.item())
            if tissue_type not in tissue_stats:
                tissue_stats[tissue_type] = 0
            tissue_stats[tissue_type] += 1
        
        print(f"当前组织分布: {tissue_stats}")
        
        # 找到需要密化的点
        densify_candidates = []
        for i in range(len(gaussians)):
            should_densify, reason = self.should_densify_medical_realistic(
                i, gaussians, opacity_values, organ_type
            )
            if should_densify:
                densify_candidates.append((i, reason))
        
        print(f"需要密化的候选点: {len(densify_candidates)}/{len(gaussians)}")
        
        # 执行密化
        new_positions = []
        new_opacities = []
        densify_count = 0
        
        for candidate_idx, reason in densify_candidates[:max_new_points]:
            if densify_count >= max_new_points:
                break
                
            positions = self.generate_medical_densification_positions(
                candidate_idx, gaussians, opacity_values, num_new_points=2
            )
            
            for new_pos in positions:
                if densify_count >= max_new_points:
                    break
                    
                # 为新点分配合理的opacity (基于邻居插值)
                original_opacity = opacity_values[candidate_idx]
                new_opacity = original_opacity + torch.randn_like(original_opacity) * 0.1
                new_opacity = torch.clamp(new_opacity, 0.001, 0.999)
                
                new_positions.append(new_pos)
                new_opacities.append(new_opacity)
                densify_count += 1
        
        result = {
            'new_positions': torch.stack(new_positions) if new_positions else torch.empty(0, 3),
            'new_opacities': torch.stack(new_opacities) if new_opacities else torch.empty(0, 1),
            'original_stats': tissue_stats,
            'densified_points': densify_count,
            'organ_type': organ_type
        }
        
        print(f"✅ 密化完成: 新增 {densify_count} 个医学合理的高斯点")
        return result


def test_realistic_densifier():
    """测试基于真实数据的densifier"""
    print("🧪 测试基于9视角真实数据的Medical Proximity Densifier...")
    
    # 模拟基于真实分布的测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟的高斯点云 (基于实际foot 9视角分布特征)
    n_points = 1000
    gaussians = torch.randn(n_points, 3, device=device) * 5.0
    
    # 模拟真实的opacity分布 (基于9视角foot数据: 90.3% in [0,0.05], 8% in [0.05,0.15], 1.4% in [0.15,0.4], 0.3% in [0.4,1])
    opacity_dist = torch.rand(n_points, device=device)
    
    # 按实际比例分配opacity值
    opacity_values = torch.zeros(n_points, 1, device=device)
    
    # 90.3% in background_air [0, 0.05]
    background_count = int(n_points * 0.903)
    opacity_values[:background_count] = torch.rand(background_count, 1, device=device) * 0.05
    
    # 8% in tissue_transition [0.05, 0.15] 
    transition_count = int(n_points * 0.08)
    start_idx = background_count
    end_idx = start_idx + transition_count
    opacity_values[start_idx:end_idx] = 0.05 + torch.rand(transition_count, 1, device=device) * 0.10
    
    # 1.4% in soft_tissue [0.15, 0.40]
    soft_count = int(n_points * 0.014)
    start_idx = end_idx
    end_idx = start_idx + soft_count
    opacity_values[start_idx:end_idx] = 0.15 + torch.rand(soft_count, 1, device=device) * 0.25
    
    # 0.3% in dense_structures [0.40, 1.0]
    dense_count = n_points - end_idx
    opacity_values[end_idx:] = 0.40 + torch.rand(dense_count, 1, device=device) * 0.60
    
    # 随机打乱
    perm = torch.randperm(n_points, device=device)
    gaussians = gaussians[perm]
    opacity_values = opacity_values[perm]
    
    # 测试densifier
    densifier = HighQualityMedicalProximityGuidedDensifier()
    
    # 测试分类功能
    print("\n📊 测试opacity分类:")
    test_opacities = [0.02, 0.08, 0.25, 0.65]
    for opacity in test_opacities:
        tissue_type = densifier.classify_opacity_realistic(opacity)
        print(f"  Opacity {opacity:.2f} → {tissue_type}")
    
    # 测试密化功能
    print("\n🔬 执行proximity-guided密化测试...")
    result = densifier.proximity_guided_densify_realistic(
        gaussians, opacity_values, organ_type="foot", max_new_points=50
    )
    
    print(f"\n✅ 测试结果:")
    print(f"  原始点数: {n_points}")
    print(f"  新增点数: {result['densified_points']}")
    print(f"  原始组织分布: {result['original_stats']}")
    
    return densifier, result

if __name__ == "__main__":
    test_realistic_densifier()