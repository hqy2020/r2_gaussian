#!/usr/bin/env python3
"""
测试 K-Planes + MLP Decoder 集成

验证修复后的 X²-Gaussian 实现：
1. DensityMLPDecoder 类初始化正确
2. 特征映射正常工作
3. 梯度流正确传播
"""

import sys
sys.path.insert(0, '.')

import torch
from r2_gaussian.gaussian.kplanes import KPlanesEncoder, DensityMLPDecoder


def test_decoder_initialization():
    """测试 Decoder 初始化"""
    print("=" * 70)
    print("测试 1: DensityMLPDecoder 初始化")
    print("=" * 70)

    decoder = DensityMLPDecoder(
        input_dim=96,
        hidden_dim=128,
        num_layers=3
    ).cuda()

    print(f"✓ Decoder 初始化成功")
    print(f"  - Input dim: {decoder.input_dim}")
    print(f"  - Hidden dim: {decoder.hidden_dim}")
    print(f"  - Num layers: {decoder.num_layers}")
    print(f"  - Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print()


def test_forward_pass():
    """测试前向传播"""
    print("=" * 70)
    print("测试 2: K-Planes + Decoder 前向传播")
    print("=" * 70)

    # 创建 encoder 和 decoder
    encoder = KPlanesEncoder(
        grid_resolution=64,
        feature_dim=32,
        bounds=(-1.0, 1.0)
    ).cuda()

    decoder = DensityMLPDecoder(
        input_dim=encoder.get_output_dim(),  # 96
        hidden_dim=128,
        num_layers=3
    ).cuda()

    # 随机生成高斯点
    N = 1000
    xyz = torch.randn(N, 3).cuda() * 0.5  # [-0.5, 0.5] 范围

    # 前向传播
    kplanes_feat = encoder(xyz)  # [N, 96]
    density_offset = decoder(kplanes_feat)  # [N, 1]

    print(f"✓ 前向传播成功")
    print(f"  - Input shape: {xyz.shape}")
    print(f"  - K-Planes features shape: {kplanes_feat.shape}")
    print(f"  - Density offset shape: {density_offset.shape}")
    print(f"  - Density offset range: [{density_offset.min():.4f}, {density_offset.max():.4f}]")
    print(f"  - Density offset mean: {density_offset.mean():.4f}")
    print(f"  - Density offset std: {density_offset.std():.4f}")
    print()


def test_gradient_flow():
    """测试梯度流"""
    print("=" * 70)
    print("测试 3: 梯度流验证")
    print("=" * 70)

    # 创建 encoder 和 decoder
    encoder = KPlanesEncoder(
        grid_resolution=64,
        feature_dim=32,
        bounds=(-1.0, 1.0)
    ).cuda()

    decoder = DensityMLPDecoder(
        input_dim=encoder.get_output_dim(),
        hidden_dim=128,
        num_layers=3
    ).cuda()

    # 随机生成高斯点
    N = 100
    xyz = torch.randn(N, 3, requires_grad=True).cuda() * 0.5

    # 前向传播
    kplanes_feat = encoder(xyz)
    density_offset = decoder(kplanes_feat)

    # 计算损失并反向传播
    loss = density_offset.mean()
    loss.backward()

    # 检查梯度
    encoder_grad_count = sum(
        1 for p in encoder.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    decoder_grad_count = sum(
        1 for p in decoder.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )

    total_encoder_params = len(list(encoder.parameters()))
    total_decoder_params = len(list(decoder.parameters()))

    print(f"✓ 梯度流验证成功")
    print(f"  - Encoder parameters with gradients: {encoder_grad_count}/{total_encoder_params}")
    print(f"  - Decoder parameters with gradients: {decoder_grad_count}/{total_decoder_params}")
    print(f"  - Input xyz gradient shape: {xyz.grad.shape if xyz.grad is not None else 'None'}")
    print()

    assert encoder_grad_count == total_encoder_params, "Encoder 梯度流异常"
    assert decoder_grad_count == total_decoder_params, "Decoder 梯度流异常"
    print("✓ 所有参数都有梯度！")
    print()


def test_modulation_effect():
    """测试 Decoder 对 density 的调制效果"""
    print("=" * 70)
    print("测试 4: Density 调制效果")
    print("=" * 70)

    # 创建 encoder 和 decoder
    encoder = KPlanesEncoder(
        grid_resolution=64,
        feature_dim=32,
        bounds=(-1.0, 1.0)
    ).cuda()

    decoder = DensityMLPDecoder(
        input_dim=encoder.get_output_dim(),
        hidden_dim=128,
        num_layers=3
    ).cuda()

    # 随机生成高斯点
    N = 1000
    xyz = torch.randn(N, 3).cuda() * 0.5

    # 模拟 baseline density
    base_density = torch.sigmoid(torch.randn(N, 1).cuda())  # [N, 1]

    # 计算调制后的 density
    kplanes_feat = encoder(xyz)
    density_offset = decoder(kplanes_feat)
    modulation = torch.exp(torch.clamp(density_offset, -5.0, 5.0))
    modulated_density = base_density * modulation

    print(f"✓ Density 调制测试成功")
    print(f"  - Base density range: [{base_density.min():.4f}, {base_density.max():.4f}]")
    print(f"  - Modulation range: [{modulation.min():.4f}, {modulation.max():.4f}]")
    print(f"  - Modulated density range: [{modulated_density.min():.4f}, {modulated_density.max():.4f}]")
    print(f"  - Mean modulation: {modulation.mean():.4f}")
    print()

    # 验证调制范围合理
    assert modulation.min() > 0, "Modulation 应该全部为正"
    assert modulated_density.min() >= 0, "Modulated density 应该非负"
    print("✓ 调制范围验证通过！")
    print()


if __name__ == "__main__":
    print("\n🚀 开始测试 K-Planes + MLP Decoder 集成\n")

    try:
        test_decoder_initialization()
        test_forward_pass()
        test_gradient_flow()
        test_modulation_effect()

        print("=" * 70)
        print("🎉 所有测试通过！X²-Gaussian MLP Decoder 修复成功！")
        print("=" * 70)
        print("\n下一步：运行完整训练")
        print("  bash scripts/train_foot3_x2_fixed.sh")
        print()

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
