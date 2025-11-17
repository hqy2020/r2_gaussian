#!/usr/bin/env python3
"""
测试 SSIM 类型转换修复

Bug: pseudo_view_coreg.py 中的 SSIM 返回值可能是 numpy.float64 而不是 Tensor
修复: 添加类型检查和转换
"""

import torch
from r2_gaussian.utils.pseudo_view_coreg import compute_pseudo_coreg_loss_medical


def test_ssim_type_conversion():
    """测试 SSIM 类型转换是否正确"""
    print("="*60)
    print("测试 SSIM 类型转换修复")
    print("="*60)

    # 创建测试数据（需要梯度）
    print("\n1. 创建随机测试图像...")
    render1 = torch.rand(3, 256, 256, requires_grad=True).cuda()
    render2 = torch.rand(3, 256, 256, requires_grad=True).cuda()

    render1_dict = {'render': render1}
    render2_dict = {'render': render2}

    print(f"   图像 1 形状: {render1.shape}")
    print(f"   图像 2 形状: {render2.shape}")
    print(f"   设备: {render1.device}")

    # 测试损失计算
    print("\n2. 计算 Pseudo Co-reg 损失...")
    try:
        loss_dict = compute_pseudo_coreg_loss_medical(render1_dict, render2_dict)

        print(f"   Total Loss: {loss_dict['loss'].item():.6f}")
        print(f"   L1 Loss: {loss_dict['l1'].item():.6f}")
        print(f"   D-SSIM Loss: {loss_dict['d_ssim'].item():.6f}")
        print(f"   SSIM Value: {loss_dict['ssim'].item():.6f}")

        # 类型检查
        print("\n3. 验证返回值类型...")
        for key, value in loss_dict.items():
            assert isinstance(value, torch.Tensor), f"{key} 不是 Tensor 类型: {type(value)}"
            print(f"   ✓ {key}: {type(value).__name__}")

        # 梯度检查
        print("\n4. 验证梯度计算...")
        loss = loss_dict['loss']
        assert loss.requires_grad or loss.grad_fn is not None, "损失不支持梯度计算"
        print(f"   ✓ requires_grad: {loss.requires_grad}")
        print(f"   ✓ grad_fn: {loss.grad_fn}")

        print("\n" + "="*60)
        print("✅ 所有测试通过！SSIM 类型转换修复成功。")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roi_weighted_loss():
    """测试带 ROI 权重的损失计算"""
    print("\n" + "="*60)
    print("测试 ROI 权重损失计算")
    print("="*60)

    # 创建测试数据
    print("\n1. 创建测试图像和 ROI 权重...")
    render1 = torch.rand(3, 256, 256).cuda()
    render2 = torch.rand(3, 256, 256).cuda()

    # 创建 ROI 权重图（中心区域权重为 0.3，边缘为 1.0）
    roi_weights = torch.ones(256, 256).cuda()
    roi_weights[64:192, 64:192] = 0.3  # 中心区域（模拟骨区）

    render1_dict = {'render': render1}
    render2_dict = {'render': render2}

    print(f"   ROI 权重形状: {roi_weights.shape}")
    print(f"   中心区域权重: 0.3 (骨区)")
    print(f"   边缘区域权重: 1.0 (软组织)")

    # 测试损失计算
    print("\n2. 计算带 ROI 权重的损失...")
    try:
        loss_dict = compute_pseudo_coreg_loss_medical(
            render1_dict, render2_dict, roi_weights=roi_weights
        )

        print(f"   Total Loss: {loss_dict['loss'].item():.6f}")
        print(f"   L1 Loss: {loss_dict['l1'].item():.6f}")
        print(f"   D-SSIM Loss: {loss_dict['d_ssim'].item():.6f}")
        print(f"   SSIM Value: {loss_dict['ssim'].item():.6f}")

        print("\n✅ ROI 权重损失计算成功！")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行所有测试
    test1_passed = test_ssim_type_conversion()
    test2_passed = test_roi_weighted_loss()

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"基础类型转换测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"ROI 权重损失测试: {'✅ 通过' if test2_passed else '❌ 失败'}")

    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！修复验证成功。")
        exit(0)
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
        exit(1)
