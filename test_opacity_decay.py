#!/usr/bin/env python3
"""
测试opacity decay功能是否正常工作
"""

import sys
sys.path.append("./")

from r2_gaussian.gaussian import GaussianModel
import torch
import numpy as np

def test_density_decay():
    """测试density decay功能"""
    print("🧪 测试density decay功能...")
    
    # 创建一个简单的GaussianModel实例
    gaussians = GaussianModel(0)
    
    # 创建一些测试数据
    xyz = np.random.rand(100, 3).astype(np.float32)
    density = np.random.rand(100, 1).astype(np.float32)
    
    # 初始化高斯模型
    gaussians.create_from_pcd(xyz, density, spatial_lr_scale=1.0)
    
    # 记录初始密度
    initial_density = gaussians.get_density.clone()
    print(f"✅ 初始密度范围: [{initial_density.min().item():.4f}, {initial_density.max().item():.4f}]")
    
    # 应用density decay
    decay_factor = 0.995
    gaussians.density_decay(factor=decay_factor)
    
    # 检查衰减后的密度
    decayed_density = gaussians.get_density
    print(f"✅ 衰减后密度范围: [{decayed_density.min().item():.4f}, {decayed_density.max().item():.4f}]")
    
    # 验证衰减是否正确
    expected_density = initial_density * decay_factor
    diff = torch.abs(decayed_density - expected_density).max().item()
    
    if diff < 1e-6:
        print("✅ Density decay功能测试通过！")
        return True
    else:
        print(f"❌ Density decay功能测试失败！最大差异: {diff}")
        return False

def test_opacity_decay_parameter():
    """测试opacity_decay参数是否正确添加"""
    print("\n🧪 测试opacity_decay参数...")
    
    try:
        from r2_gaussian.arguments import ModelParams
        from argparse import ArgumentParser
        
        parser = ArgumentParser()
        model_params = ModelParams(parser)
        
        # 检查opacity_decay属性是否存在
        if hasattr(model_params, 'opacity_decay'):
            print(f"✅ opacity_decay参数存在，默认值: {model_params.opacity_decay}")
            return True
        else:
            print("❌ opacity_decay参数不存在")
            return False
            
    except Exception as e:
        print(f"❌ 测试opacity_decay参数时出错: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始测试opacity decay功能...")
    
    success1 = test_density_decay()
    success2 = test_opacity_decay_parameter()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！opacity decay功能已成功实现！")
        print("\n📝 使用方法:")
        print("python train.py -s /path/to/data.pickle -m /path/to/output --opacity_decay")
    else:
        print("\n❌ 部分测试失败，请检查实现")
