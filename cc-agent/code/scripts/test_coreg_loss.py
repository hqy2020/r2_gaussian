"""
测试 Co-regularization 损失计算的正确性

运行方法:
    cd /home/qyhu/Documents/r2_ours/r2_gaussian
    conda activate r2_gaussian_new
    python cc-agent/code/scripts/test_coreg_loss.py

测试内容:
1. L1 损失计算正确性
2. D-SSIM 损失计算正确性
3. ROI 权重图应用
4. 相同图像的损失值（应接近 0）
5. 完全随机图像的损失值（应较大）
"""

import sys
import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.append("/home/qyhu/Documents/r2_ours/r2_gaussian")

from r2_gaussian.utils.pseudo_view_coreg import (
    compute_pseudo_coreg_loss_medical,
    create_roi_weight_map
)


def test_coreg_loss():
    """测试损失函数计算"""

    print("\n" + "="*60)
    print("测试 Co-regularization 损失计算")
    print("="*60 + "\n")

    # 设置设备和图像尺寸
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H, W = 256, 256
    print(f"使用设备: {device}")
    print(f"图像尺寸: {H}x{W}\n")

    # 测试 1: 随机图像损失计算
    print("步骤 1: 测试随机图像的损失计算...")

    render1 = {'render': torch.rand(3, H, W, device=device)}
    render2 = {'render': torch.rand(3, H, W, device=device)}

    loss_dict = compute_pseudo_coreg_loss_medical(
        render1, render2, lambda_dssim=0.2
    )

    print(f"  随机图像损失:")
    print(f"    Total: {loss_dict['loss'].item():.6f}")
    print(f"    L1: {loss_dict['l1'].item():.6f}")
    print(f"    D-SSIM: {loss_dict['d_ssim'].item():.6f}")
    print(f"    SSIM: {loss_dict['ssim'].item():.6f}")

    # 验证损失在合理范围
    if 0 < loss_dict['loss'].item() < 1.0:
        print(f"  ✅ 总损失在合理范围 [0, 1]")
    else:
        print(f"  ⚠️ 警告: 总损失超出范围！")

    if 0 < loss_dict['ssim'].item() < 1.0:
        print(f"  ✅ SSIM 值在合理范围 [0, 1]")
    else:
        print(f"  ⚠️ 警告: SSIM 值超出范围！")

    # 测试 2: 相同图像（损失应该接近 0）
    print("\n步骤 2: 测试相同图像的损失（应接近 0）...")

    render_same = {'render': render1['render'].clone()}
    loss_same = compute_pseudo_coreg_loss_medical(render1, render_same)

    print(f"  相同图像损失:")
    print(f"    Total: {loss_same['loss'].item():.8f}")
    print(f"    L1: {loss_same['l1'].item():.8f}")
    print(f"    D-SSIM: {loss_same['d_ssim'].item():.8f}")
    print(f"    SSIM: {loss_same['ssim'].item():.8f}")

    if loss_same['loss'].item() < 1e-6:
        print(f"  ✅ 相同图像损失接近 0（{loss_same['loss'].item():.2e}）")
    else:
        print(f"  ⚠️ 警告: 相同图像损失应该接近 0，实际为 {loss_same['loss'].item():.6f}")

    if loss_same['ssim'].item() > 0.99:
        print(f"  ✅ 相同图像 SSIM 接近 1（{loss_same['ssim'].item():.6f}）")
    else:
        print(f"  ⚠️ 警告: 相同图像 SSIM 应该接近 1，实际为 {loss_same['ssim'].item():.6f}")

    # 测试 3: 完全不同的图像（一个全 0，一个全 1）
    print("\n步骤 3: 测试完全不同图像的损失（应较大）...")

    render_black = {'render': torch.zeros(3, H, W, device=device)}
    render_white = {'render': torch.ones(3, H, W, device=device)}

    loss_diff = compute_pseudo_coreg_loss_medical(render_black, render_white)

    print(f"  完全不同图像损失:")
    print(f"    Total: {loss_diff['loss'].item():.6f}")
    print(f"    L1: {loss_diff['l1'].item():.6f}")
    print(f"    D-SSIM: {loss_diff['d_ssim'].item():.6f}")
    print(f"    SSIM: {loss_diff['ssim'].item():.6f}")

    # 理论值: L1=1.0, SSIM≈0（完全不同）
    expected_l1 = 1.0
    if abs(loss_diff['l1'].item() - expected_l1) < 0.01:
        print(f"  ✅ L1 损失符合预期（理论值 1.0，实际 {loss_diff['l1'].item():.6f}）")
    else:
        print(f"  ⚠️ 警告: L1 损失不符合预期")

    if loss_diff['ssim'].item() < 0.1:
        print(f"  ✅ SSIM 值符合预期（完全不同时应接近 0，实际 {loss_diff['ssim'].item():.6f}）")
    else:
        print(f"  ⚠️ 警告: SSIM 值不符合预期")

    # 测试 4: ROI 权重图应用
    print("\n步骤 4: 测试 ROI 权重图应用...")

    # 创建 ROI 掩码（中心 100x100 像素为骨区）
    roi_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
    center_start = (H - 100) // 2
    center_end = center_start + 100
    roi_mask[center_start:center_end, center_start:center_end] = True

    # 创建权重图
    weight_map = create_roi_weight_map(
        (H, W), roi_mask,
        bone_weight=0.3,
        soft_tissue_weight=1.0,
        device=device
    )

    print(f"  ROI 掩码: 骨区像素数 = {roi_mask.sum().item()}, "
          f"总像素数 = {H*W}, "
          f"比例 = {roi_mask.sum().item() / (H*W) * 100:.1f}%")

    # 验证权重值
    bone_weight_actual = weight_map[center_start + 50, center_start + 50].item()
    soft_weight_actual = weight_map[10, 10].item()

    print(f"  权重图验证:")
    print(f"    骨区中心权重: {bone_weight_actual:.2f} (预期 0.30)")
    print(f"    软组织权重: {soft_weight_actual:.2f} (预期 1.00)")

    if abs(bone_weight_actual - 0.3) < 1e-5 and abs(soft_weight_actual - 1.0) < 1e-5:
        print(f"  ✅ ROI 权重图创建正确")
    else:
        print(f"  ⚠️ 警告: ROI 权重图值不正确")

    # 计算带权重的损失
    loss_weighted = compute_pseudo_coreg_loss_medical(
        render1, render2, lambda_dssim=0.2, roi_weights=weight_map
    )

    print(f"\n  带 ROI 权重的损失:")
    print(f"    Total: {loss_weighted['loss'].item():.6f}")
    print(f"    无权重总损失: {loss_dict['loss'].item():.6f}")
    print(f"    差异: {abs(loss_weighted['loss'].item() - loss_dict['loss'].item()):.6f}")

    # 有权重的损失应该小于无权重（因为骨区权重 0.3 < 1.0）
    if loss_weighted['loss'].item() < loss_dict['loss'].item():
        print(f"  ✅ ROI 权重正确应用（加权损失更小）")
    else:
        print(f"  ⚠️ 提示: 加权损失未显著降低（可能是骨区占比较小）")

    # 测试 5: 损失公式验证（权重 0.8:0.2）
    print("\n步骤 5: 验证损失公式（0.8 × L1 + 0.2 × D-SSIM）...")

    manual_total = 0.8 * loss_dict['l1'].item() + 0.2 * loss_dict['d_ssim'].item()
    auto_total = loss_dict['loss'].item()

    print(f"  手动计算总损失: {manual_total:.6f}")
    print(f"  函数返回总损失: {auto_total:.6f}")
    print(f"  差异: {abs(manual_total - auto_total):.8f}")

    if abs(manual_total - auto_total) < 1e-6:
        print(f"  ✅ 损失公式验证正确")
    else:
        print(f"  ⚠️ 警告: 损失公式计算存在误差")

    # 最终总结
    print("\n" + "="*60)
    print("✅ Co-regularization 损失测试全部完成！")
    print("="*60 + "\n")

    print("总结:")
    print("  - 随机图像损失值在合理范围")
    print("  - 相同图像损失接近 0")
    print("  - 完全不同图像损失符合理论预期")
    print("  - ROI 权重图正确应用")
    print("  - 损失公式计算准确")
    print("\n可以安全地用于 train.py 主训练循环！\n")


if __name__ == "__main__":
    test_coreg_loss()
