#!/usr/bin/env python3
"""
K-Planes 修复验证脚本

快速验证 K-Planes 特征是否正确集成到渲染流程中。
运行时间：< 30 秒

作者：Claude Code Agent
日期：2025-01-19
"""

import sys
import torch

sys.path.append("/home/qyhu/Documents/r2_ours/r2_gaussian")

from r2_gaussian.gaussian import GaussianModel
from r2_gaussian.arguments import ModelParams, OptimizationParams
import argparse
import numpy as np

def main():
    print("=" * 70)
    print("K-Planes 修复验证脚本")
    print("=" * 70)

    # 创建参数
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    args = parser.parse_args(['--model_path', 'dummy'])

    # 模拟启用 K-Planes
    args.enable_kplanes = True
    args.kplanes_resolution = 64
    args.kplanes_dim = 32

    print("\n✓ 创建 GaussianModel（启用 K-Planes）...")
    gaussians = GaussianModel(args=args)

    # 创建假数据
    print("✓ 初始化测试数据（1000 个高斯）...")
    xyz = np.random.randn(1000, 3).astype(np.float32) * 0.1
    density = np.ones((1000, 1), dtype=np.float32) * 0.5
    gaussians.create_from_pcd(xyz, density, spatial_lr_scale=1.0)
    gaussians.training_setup(opt_params)

    print("\n" + "=" * 70)
    print("检查点 1：K-Planes Encoder 已创建")
    print("=" * 70)
    print(f"✓ K-Planes encoder 存在: {gaussians.kplanes_encoder is not None}")

    kplanes_params = sum(p.numel() for p in gaussians.kplanes_encoder.parameters())
    print(f"✓ K-Planes 参数量: {kplanes_params:,}")
    print(f"  预期：{64*64*32*3:,} (分辨率^2 * 特征维度 * 3 平面)")

    print("\n" + "=" * 70)
    print("检查点 2：优化器参数组")
    print("=" * 70)
    for i, group in enumerate(gaussians.optimizer.param_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"  {i+1}. {group['name']:10s} - LR: {group['lr']:.6f} - Params: {param_count:,}")

    print("\n" + "=" * 70)
    print("检查点 3：K-Planes 特征计算")
    print("=" * 70)
    feat = gaussians.get_kplanes_features()
    print(f"✓ K-Planes 特征形状: {feat.shape}")
    print(f"✓ 特征范围: [{feat.min().item():.4f}, {feat.max().item():.4f}]")
    print(f"✓ 特征均值: {feat.mean().item():.4f}")
    print(f"✓ 特征标准差: {feat.std().item():.4f}")

    print("\n" + "=" * 70)
    print("检查点 4：🎯 关键修复 - K-Planes 是否参与渲染？")
    print("=" * 70)

    # 测试 get_density（这是关键修复）
    print("✓ 调用 get_density（应该调用 K-Planes 特征调制）...")

    # 先关闭 K-Planes，获取 baseline density
    gaussians.enable_kplanes = False
    density_baseline = gaussians.get_density

    # 再启用 K-Planes，获取调制后的 density
    gaussians.enable_kplanes = True
    density_modulated = gaussians.get_density

    print(f"✓ Baseline density 形状: {density_baseline.shape}")
    print(f"✓ Modulated density 形状: {density_modulated.shape}")
    print(f"✓ Baseline density 范围: [{density_baseline.min().item():.4f}, {density_baseline.max().item():.4f}]")
    print(f"✓ Modulated density 范围: [{density_modulated.min().item():.4f}, {density_modulated.max().item():.4f}]")

    # 检查是否有调制效果
    diff = (density_modulated - density_baseline).abs().mean().item()
    print(f"\n✓ 平均调制幅度: {diff:.6f}")

    if diff > 1e-6:
        print("✅ 成功！K-Planes 特征正在调制 density")
        print("   → 调制范围应该在 [0.8, 1.2] 之间（保守策略）")
        modulation_ratio = (density_modulated / density_baseline).mean().item()
        print(f"   → 平均调制比例: {modulation_ratio:.4f}")
    else:
        print("❌ 失败！K-Planes 特征未调制 density")
        print("   → 请检查 gaussian_model.py 的 get_density 属性")

    print("\n" + "=" * 70)
    print("检查点 5：梯度反向传播")
    print("=" * 70)

    # 测试梯度
    density_modulated.sum().backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in gaussians.kplanes_encoder.parameters())
    print(f"✓ K-Planes 参数有梯度: {has_grad}")

    if has_grad:
        print("✅ 成功！K-Planes 参数会被优化")
    else:
        print("❌ 警告！K-Planes 参数没有梯度")

    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)

    checks_passed = []
    checks_passed.append(("K-Planes encoder 已创建", gaussians.kplanes_encoder is not None))
    checks_passed.append(("K-Planes 参数已注册到优化器", any(g['name'] == 'kplanes' for g in gaussians.optimizer.param_groups)))
    checks_passed.append(("K-Planes 特征可以计算", feat is not None and feat.shape[0] == 1000))
    checks_passed.append(("K-Planes 调制 density", diff > 1e-6))
    checks_passed.append(("K-Planes 参数有梯度", has_grad))

    for check_name, check_result in checks_passed:
        status = "✅ 通过" if check_result else "❌ 失败"
        print(f"  {status}: {check_name}")

    all_passed = all(result for _, result in checks_passed)

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 所有检查通过！K-Planes 修复成功！")
        print("=" * 70)
        print("\n下一步：运行完整训练实验")
        print("  bash scripts/train_kplanes_foot3.sh")
    else:
        print("⚠️ 部分检查失败，需要进一步调试")
        print("=" * 70)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
