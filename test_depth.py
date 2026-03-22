#!/usr/bin/env python3
"""
测试 r2-gaussian 的 depth 功能
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append("/home/qyhu/Documents/r2_ours/r2_gaussian")

from r2_gaussian.utils.depth_utils import extract_depth_from_volume_ray_casting, compute_depth_loss
from r2_gaussian.utils.loss_utils import depth_loss_fn

def test_depth_extraction():
    """测试深度提取功能"""
    print("测试深度提取功能...")
    
    # 创建一个模拟的 density volume
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, H, W = 32, 64, 64
    density_volume = torch.rand(D, H, W, device=device)
    
    # 模拟相机参数
    class MockCamera:
        def __init__(self):
            self.device = device
    
    camera_params = MockCamera()
    
    try:
        # 测试深度提取
        depth_map = extract_depth_from_volume_ray_casting(
            density_volume, 
            camera_params, 
            threshold=0.01
        )
        
        print(f"✓ 深度提取成功")
        print(f"  - 输入 volume 形状: {density_volume.shape}")
        print(f"  - 输出 depth map 形状: {depth_map.shape}")
        print(f"  - 深度值范围: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ 深度提取失败: {e}")
        return False

def test_depth_loss():
    """测试深度损失函数"""
    print("\n测试深度损失函数...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = 64, 64
    
    # 创建模拟的深度图
    rendered_depth = torch.rand(H, W, device=device)
    gt_depth = torch.rand(H, W, device=device)
    
    try:
        # 测试不同的损失类型
        for loss_type in ['l1', 'l2', 'pearson']:
            loss_val = depth_loss_fn(rendered_depth, gt_depth, loss_type=loss_type)
            print(f"✓ {loss_type} 损失计算成功: {loss_val:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 深度损失计算失败: {e}")
        return False

def test_depth_consistency():
    """测试深度一致性"""
    print("\n测试深度一致性...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = 64, 64
    
    # 创建多个视角的深度图
    depth_maps = [
        torch.rand(H, W, device=device),
        torch.rand(H, W, device=device),
        torch.rand(H, W, device=device)
    ]
    
    try:
        from r2_gaussian.utils.loss_utils import depth_consistency_loss
        consistency_loss = depth_consistency_loss(depth_maps)
        print(f"✓ 深度一致性损失计算成功: {consistency_loss:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 深度一致性计算失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("r2-gaussian Depth 功能测试")
    print("=" * 50)
    
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA 不可用，使用 CPU")
    
    # 运行测试
    tests = [
        test_depth_extraction,
        test_depth_loss,
        test_depth_consistency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！depth 功能已成功集成")
    else:
        print("✗ 部分测试失败，请检查实现")
    
    print("=" * 50)

if __name__ == "__main__":
    main()






