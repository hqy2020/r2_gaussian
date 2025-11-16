#!/usr/bin/env python3
"""
FSGS Proximity-guided Densification for R²-Gaussian
集成了FSGS的proximity-guided densification和医学感知策略
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from simple_knn._C import distCUDA2

warnings.filterwarnings('ignore')

class FSGSProximityDensifier:
    """
    FSGS Proximity-guided Densification for R²-Gaussian
    结合FSGS的proximity-guided思想和医学CT分级策略
    """
    
    def __init__(self, 
                 proximity_threshold: float = 10.0,
                 k_neighbors: int = 3,
                 enable_medical_constraints: bool = True,
                 organ_type: str = "general"):
        """
        初始化 FSGS proximity densifier
        
        Args:
            proximity_threshold: proximity score 阈值,超过此值则进行densification
            k_neighbors: 计算proximity时的邻居数量
            enable_medical_constraints: 是否启用医学约束
            organ_type: 器官类型,用于医学约束参数调整
        """
        self.proximity_threshold = proximity_threshold
        self.k_neighbors = k_neighbors
        self.enable_medical_constraints = enable_medical_constraints
        self.organ_type = organ_type
        
        # 医学CT分级系统(基于创新点1)
        self.medical_tissue_types = {
            "background_air": {
                "opacity_range": (0.0, 0.05),
                "proximity_params": {
                    "min_neighbors": 6,
                    "max_distance": 2.0,
                    "max_gradient": 0.05
                }
            },
            "tissue_transition": {
                "opacity_range": (0.05, 0.15),
                "proximity_params": {
                    "min_neighbors": 8,
                    "max_distance": 1.5,
                    "max_gradient": 0.10
                }
            },
            "soft_tissue": {
                "opacity_range": (0.15, 0.40),
                "proximity_params": {
                    "min_neighbors": 6,
                    "max_distance": 1.0,
                    "max_gradient": 0.25
                }
            },
            "dense_structures": {
                "opacity_range": (0.40, 1.0),
                "proximity_params": {
                    "min_neighbors": 4,
                    "max_distance": 0.8,
                    "max_gradient": 0.60
                }
            }
        }
        
    def classify_medical_tissue(self, opacity_value: float) -> str:
        """基于opacity值进行医学组织分类"""
        for tissue_type, info in self.medical_tissue_types.items():
            min_val, max_val = info["opacity_range"]
            if min_val <= opacity_value < max_val:
                return tissue_type
        return "dense_structures"
    
    def build_proximity_graph(self, gaussians: torch.Tensor) -> Dict:
        """
        构建proximity graph (基于FSGS算法)
        
        Args:
            gaussians: 高斯点位置 (N, 3)
            
        Returns:
            proximity_graph: 包含邻居关系和proximity score的字典
        """
        N = gaussians.shape[0]
        device = gaussians.device
        
        # 计算所有点之间的距离
        try:
            distances = torch.cdist(gaussians, gaussians, p=2)  # (N, N)
        except RuntimeError as e:
            # CUBLAS错误回退: 使用CPU计算或简化计算
            print(f"⚠️ CUDA distance calculation failed: {e}")
            print("   Falling back to CPU computation...")
            gaussians_cpu = gaussians.cpu()
            distances = torch.cdist(gaussians_cpu, gaussians_cpu, p=2).to(device)
        
        # 对每个点找到K个最近邻居
        proximity_graph = {}
        
        for i in range(N):
            # 找到K个最近邻居(排除自己)
            dist_row = distances[i]
            dist_row[i] = float('inf')  # 排除自己
            
            # 找到K个最近的邻居
            k_nearest_distances, k_nearest_indices = torch.topk(
                dist_row, k=min(self.k_neighbors, N-1), largest=False
            )
            
            # 计算proximity score (FSGS公式)
            proximity_score = k_nearest_distances.mean().item()
            
            proximity_graph[i] = {
                'neighbors': k_nearest_indices.tolist(),
                'distances': k_nearest_distances.tolist(),
                'proximity_score': proximity_score
            }
            
        return proximity_graph
    
    def should_densify_proximity(self, 
                               gaussian_idx: int, 
                               proximity_graph: Dict,
                               opacity_values: torch.Tensor = None) -> Tuple[bool, str]:
        """
        判断是否需要进行proximity-guided densification
        
        Args:
            gaussian_idx: 高斯点索引
            proximity_graph: proximity graph
            opacity_values: opacity值,用于医学约束(可选)
            
        Returns:
            (should_densify, reason): 是否需要密化和原因
        """
        if gaussian_idx not in proximity_graph:
            return False, "不在proximity graph中"
            
        proximity_info = proximity_graph[gaussian_idx]
        proximity_score = proximity_info['proximity_score']
        
        # FSGS基础条件: proximity score超过阈值
        if proximity_score < self.proximity_threshold:
            return False, f"proximity score过低: {proximity_score:.3f} < {self.proximity_threshold}"
        
        # 医学约束检查
        if self.enable_medical_constraints and opacity_values is not None:
            opacity_val = opacity_values[gaussian_idx].item()
            tissue_type = self.classify_medical_tissue(opacity_val)
            medical_params = self.medical_tissue_types[tissue_type]["proximity_params"]
            
            # 检查邻居数量约束
            num_neighbors = len(proximity_info['neighbors'])
            min_neighbors = medical_params["min_neighbors"]
            if num_neighbors < min_neighbors:
                return True, f"医学约束-邻居不足: {num_neighbors} < {min_neighbors} ({tissue_type})"
            
            # 检查距离约束
            avg_distance = np.mean(proximity_info['distances'])
            max_distance = medical_params["max_distance"]
            if avg_distance > max_distance:
                return True, f"医学约束-距离过大: {avg_distance:.3f} > {max_distance} ({tissue_type})"
        
        return True, f"proximity densification: score={proximity_score:.3f} > {self.proximity_threshold}"
    
    def generate_new_positions(self, 
                             gaussian_idx: int, 
                             gaussians: torch.Tensor,
                             proximity_graph: Dict,
                             opacity_values: torch.Tensor = None,
                             num_new_points: int = 2) -> List[torch.Tensor]:
        """
        基于proximity graph生成新的高斯点位置 (FSGS unpooling策略)
        
        Args:
            gaussian_idx: 源高斯点索引
            gaussians: 所有高斯点位置
            proximity_graph: proximity graph
            opacity_values: opacity值,用于医学约束
            num_new_points: 生成的新点数量
            
        Returns:
            new_positions: 新点位置列表
        """
        if gaussian_idx not in proximity_graph:
            return []
        
        proximity_info = proximity_graph[gaussian_idx]
        source_pos = gaussians[gaussian_idx]
        neighbors = proximity_info['neighbors']
        
        if len(neighbors) == 0:
            return []
        
        new_positions = []
        
        # FSGS策略: 在源点和目标点之间插入新点
        for i, neighbor_idx in enumerate(neighbors[:num_new_points]):
            neighbor_pos = gaussians[neighbor_idx]
            
            # 在中点位置插入新高斯点 (FSGS unpooling)
            new_pos = (source_pos + neighbor_pos) / 2.0
            
            # 添加小量随机噪声避免重叠
            noise_std = 0.1
            if self.enable_medical_constraints and opacity_values is not None:
                opacity_val = opacity_values[gaussian_idx].item()
                tissue_type = self.classify_medical_tissue(opacity_val)
                max_distance = self.medical_tissue_types[tissue_type]["proximity_params"]["max_distance"]
                noise_std = max_distance * 0.05  # 5%的噪声
                
            noise = torch.randn_like(new_pos) * noise_std
            new_pos += noise
            
            new_positions.append(new_pos)
        
        return new_positions
    
    def proximity_guided_densification(self, 
                                     gaussians: torch.Tensor,
                                     opacity_values: torch.Tensor = None,
                                     max_new_points: int = 1000) -> Dict:
        """
        执行FSGS proximity-guided densification
        
        Args:
            gaussians: 高斯点位置 (N, 3)
            opacity_values: opacity值 (N, 1),用于医学约束
            max_new_points: 最大新增点数
            
        Returns:
            result: 包含新增点信息的字典
        """
        # 构建proximity graph
        proximity_graph = self.build_proximity_graph(gaussians)
        
        # 找到需要densify的候选点
        densify_candidates = []
        for i in range(len(gaussians)):
            should_densify, reason = self.should_densify_proximity(
                i, proximity_graph, opacity_values
            )
            if should_densify:
                densify_candidates.append((i, reason))
        
        # 生成新点
        new_positions = []
        new_opacities = []
        densify_count = 0
        
        for candidate_idx, reason in densify_candidates:
            if densify_count >= max_new_points:
                break
                
            positions = self.generate_new_positions(
                candidate_idx, gaussians, proximity_graph, opacity_values, num_new_points=2
            )
            
            for new_pos in positions:
                if densify_count >= max_new_points:
                    break
                    
                new_positions.append(new_pos)
                
                # 为新点分配opacity (基于父点)
                if opacity_values is not None:
                    parent_opacity = opacity_values[candidate_idx]
                    # 添加小量噪声
                    new_opacity = parent_opacity + torch.randn_like(parent_opacity) * 0.05
                    new_opacity = torch.clamp(new_opacity, 0.001, 0.999)
                    new_opacities.append(new_opacity)
                
                densify_count += 1
        
        result = {
            'new_positions': torch.stack(new_positions) if new_positions else torch.empty(0, 3, device=gaussians.device),
            'new_opacities': torch.stack(new_opacities) if new_opacities else torch.empty(0, 1, device=gaussians.device),
            'densified_count': densify_count,
            'total_candidates': len(densify_candidates),
            'proximity_threshold': self.proximity_threshold,
            'medical_constraints': self.enable_medical_constraints
        }
        
        return result


def add_fsgs_proximity_to_gaussian_model(gaussian_model, 
                                        proximity_threshold: float = 10.0,
                                        enable_medical_constraints: bool = True,
                                        organ_type: str = "general"):
    """
    为GaussianModel添加FSGS proximity-guided densification功能
    
    Args:
        gaussian_model: R²-Gaussian模型实例
        proximity_threshold: proximity阈值
        enable_medical_constraints: 是否启用医学约束
        organ_type: 器官类型
    """
    
    # 添加proximity densifier作为模型属性
    gaussian_model.proximity_densifier = FSGSProximityDensifier(
        proximity_threshold=proximity_threshold,
        enable_medical_constraints=enable_medical_constraints,
        organ_type=organ_type
    )
    
    # 保存原始的densify_and_prune方法
    original_densify_and_prune = gaussian_model.densify_and_prune
    
    def enhanced_densify_and_prune(self,
                                 max_grad,
                                 min_density,
                                 max_screen_size,
                                 max_scale,
                                 max_num_gaussians,
                                 densify_scale_threshold,
                                 bbox=None,
                                 enable_proximity_densify=True):
        """
        增强版本的densify_and_prune,集成了FSGS proximity-guided densification
        """
        # 首先执行原始的gradient-based densification
        grads = original_densify_and_prune(
            max_grad, min_density, max_screen_size, max_scale, 
            max_num_gaussians, densify_scale_threshold, bbox
        )
        
        # 执行FSGS proximity-guided densification
        if enable_proximity_densify and hasattr(self, 'proximity_densifier'):
            current_points = self.get_xyz.shape[0]
            if current_points < max_num_gaussians:
                remaining_budget = max_num_gaussians - current_points
                
                # 获取opacity值用于医学约束
                opacity_values = None
                if self.proximity_densifier.enable_medical_constraints:
                    if hasattr(self, 'get_opacity'):
                        opacity_values = self.get_opacity
                    elif hasattr(self, 'get_density'):
                        # 回退到density值
                        opacity_values = self.get_density
                    else:
                        # 最后回退：使用opacity activation
                        opacity_values = self.opacity_activation(self._density)
                
                # 执行proximity-guided densification
                proximity_result = self.proximity_densifier.proximity_guided_densification(
                    self.get_xyz, opacity_values, max_new_points=min(remaining_budget, 500)
                )
                
                if proximity_result['densified_count'] > 0:
                    print(f"🌟 [FSGS-Proximity] 新增 {proximity_result['densified_count']} 个proximity-guided高斯点")
                    
                    # 添加新的高斯点
                    new_positions = proximity_result['new_positions']
                    new_opacities = proximity_result['new_opacities']
                    
                    # 为新点初始化其他参数
                    n_new = new_positions.shape[0]
                    device = new_positions.device
                    
                    # 基于最近邻初始化scaling
                    new_scaling = torch.log(torch.ones(n_new, 3, device=device) * 0.5)
                    
                    # 初始化rotation (单位四元数)
                    new_rotation = torch.zeros(n_new, 4, device=device)
                    new_rotation[:, 0] = 1.0
                    
                    # 初始化density (基于opacity)
                    if new_opacities.shape[0] > 0:
                        new_densities = self.density_inverse_activation(
                            torch.clamp(new_opacities, 0.001, 0.999)
                        )
                    else:
                        new_densities = torch.ones(n_new, 1, device=device) * 0.1
                        new_densities = self.density_inverse_activation(new_densities)
                    
                    # 初始化max_radii2D
                    new_max_radii2D = torch.zeros(n_new, device=device)
                    
                    # SSS parameters
                    new_nu = None
                    new_opacity_param = None
                    if hasattr(self, 'use_student_t') and self.use_student_t:
                        new_nu = torch.zeros(n_new, 1, device=device)
                        new_opacity_param = self.opacity_inverse_activation(new_opacities)
                    else:
                        new_opacity_param = new_densities
                    
                    # 添加到模型中
                    self.densification_postfix(
                        new_positions,
                        new_densities,
                        new_scaling,
                        new_rotation,
                        new_max_radii2D,
                        new_nu,
                        new_opacity_param
                    )
        
        return grads
    
    # 替换方法
    gaussian_model.enhanced_densify_and_prune = enhanced_densify_and_prune.__get__(gaussian_model)
    
    print(f"✅ [FSGS集成] 成功为GaussianModel添加proximity-guided densification功能")
    print(f"   - Proximity threshold: {proximity_threshold}")
    print(f"   - Medical constraints: {enable_medical_constraints}")
    print(f"   - Organ type: {organ_type}")
    
    return gaussian_model


# 测试函数
def test_fsgs_proximity():
    """测试FSGS proximity densification"""
    print("🧪 测试FSGS Proximity-guided Densification...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    n_points = 100
    gaussians = torch.randn(n_points, 3, device=device) * 2.0
    opacity_values = torch.rand(n_points, 1, device=device)
    
    # 创建densifier
    densifier = FSGSProximityDensifier(
        proximity_threshold=8.0,
        enable_medical_constraints=True,
        organ_type="foot"
    )
    
    # 测试proximity densification
    result = densifier.proximity_guided_densification(
        gaussians, opacity_values, max_new_points=50
    )
    
    print(f"✅ 测试结果:")
    print(f"   - 原始点数: {n_points}")
    print(f"   - 候选点数: {result['total_candidates']}")
    print(f"   - 新增点数: {result['densified_count']}")
    print(f"   - 医学约束: {result['medical_constraints']}")
    
    return densifier, result


if __name__ == "__main__":
    test_fsgs_proximity()