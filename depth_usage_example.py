#!/usr/bin/env python3
"""
r2-gaussian Depth 功能使用示例

使用方法：
1. 启用 depth 功能：
   python train.py -s /path/to/data.pickle -m /path/to/output \
     --enable_depth --depth_loss_weight 0.05 --depth_loss_type pearson

2. 禁用 depth 功能（默认）：
   python train.py -s /path/to/data.pickle -m /path/to/output

参数说明：
- --enable_depth: 启用深度功能
- --depth_loss_weight: 深度损失权重（默认 0.0）
- --depth_loss_type: 深度损失类型 ('l1', 'l2', 'pearson'，默认 'pearson')
- --depth_threshold: 深度提取阈值（默认 0.01）
"""

import sys
import os

# 添加项目路径
sys.path.append("/home/qyhu/Documents/r2_ours/r2_gaussian")

def print_usage():
    """打印使用说明"""
    print("=" * 60)
    print("r2-gaussian Depth 功能使用说明")
    print("=" * 60)
    
    print("\n1. 启用 depth 功能：")
    print("   python train.py -s /path/to/data.pickle -m /path/to/output \\")
    print("     --enable_depth --depth_loss_weight 0.05 --depth_loss_type pearson")
    
    print("\n2. 禁用 depth 功能（默认）：")
    print("   python train.py -s /path/to/data.pickle -m /path/to/output")
    
    print("\n3. 参数说明：")
    print("   --enable_depth: 启用深度功能")
    print("   --depth_loss_weight: 深度损失权重（默认 0.0）")
    print("   --depth_loss_type: 深度损失类型 ('l1', 'l2', 'pearson'，默认 'pearson')")
    print("   --depth_threshold: 深度提取阈值（默认 0.01）")
    
    print("\n4. 实现原理：")
    print("   - 使用 voxelization 从 Gaussians 提取 3D density volume")
    print("   - 通过 ray casting 从 volume 提取深度图")
    print("   - 计算深度损失并加入总损失函数")
    print("   - 支持多种损失类型：L1、L2、Pearson 相关系数")
    
    print("\n5. 预期效果：")
    print("   - 改善 3D 几何重建质量")
    print("   - 提升 PSNR 和 SSIM 指标")
    print("   - 增强多视角一致性")
    
    print("\n6. 注意事项：")
    print("   - 需要 ground truth 深度数据")
    print("   - 深度损失权重需要调参")
    print("   - 建议先用小权重测试效果")
    
    print("=" * 60)

def print_implementation_summary():
    """打印实现总结"""
    print("\n实现总结：")
    print("✓ 1. 深度提取函数：extract_depth_from_volume_ray_casting")
    print("✓ 2. 深度损失函数：depth_loss_fn")
    print("✓ 3. 训练循环集成：depth 损失计算")
    print("✓ 4. 命令行参数：enable_depth, depth_loss_weight 等")
    print("✓ 5. 测试脚本：test_depth.py")
    
    print("\n文件修改：")
    print("- 新增：r2_gaussian/utils/depth_utils.py")
    print("- 修改：r2_gaussian/utils/loss_utils.py")
    print("- 修改：r2_gaussian/arguments/__init__.py")
    print("- 修改：train.py")
    print("- 新增：test_depth.py")

if __name__ == "__main__":
    print_usage()
    print_implementation_summary()






