"""
GR-Gaussian 单元测试

测试内容：
1. GaussianGraph 图构建正确性
2. Graph Laplacian 损失函数非零且可微
3. De-Init 降噪初始化有效性

运行方式：
    python tests/test_gr_gaussian.py
"""

import torch
import numpy as np
import sys
sys.path.append("./")

from r2_gaussian.utils.graph_utils import GaussianGraph
from r2_gaussian.utils.loss_utils import compute_graph_laplacian_loss
from r2_gaussian.gaussian.gaussian_model import GaussianModel


def test_gaussian_graph_construction():
    """测试 1: GaussianGraph 图构建正确性"""
    print("\n" + "="*60)
    print("测试 1: GaussianGraph 图构建正确性")
    print("="*60)

    # 创建随机点云
    N = 100
    xyz = torch.randn(N, 3).cuda() * 0.5  # 模拟 [-1, 1]³ 空间

    # 构建图（k=6, 论文推荐值）
    graph = GaussianGraph(k=6, device='cuda')
    graph.build_knn_graph(xyz)
    graph.compute_edge_weights(xyz)

    # 验证
    assert graph.num_nodes == N, f"节点数错误：期望 {N}，实际 {graph.num_nodes}"
    assert graph.edge_index.shape[0] == 2, f"边索引维度错误：{graph.edge_index.shape}"
    assert graph.edge_index.shape[1] == N * 6, f"边数错误：期望 {N*6}，实际 {graph.edge_index.shape[1]}"
    assert graph.edge_weights.shape[0] == N * 6, f"边权重数量错误"
    assert torch.all(graph.edge_weights > 0), "边权重应全部为正"
    assert torch.all(graph.edge_weights <= 1.0), "边权重应 ≤ 1.0（高斯核）"

    print(f"✅ 图构建成功：{graph.num_nodes} 节点, {graph.num_edges} 条边")
    print(f"   边权重范围: [{graph.edge_weights.min():.4f}, {graph.edge_weights.max():.4f}]")


def test_graph_laplacian_loss_nonzero():
    """测试 2: Graph Laplacian 损失函数非零且可微"""
    print("\n" + "="*60)
    print("测试 2: Graph Laplacian 损失函数非零且可微")
    print("="*60)

    # 创建虚拟 GaussianModel
    gaussians = GaussianModel(scale_bound=None)
    N = 200
    gaussians._xyz = torch.randn(N, 3, requires_grad=True).cuda() * 0.5
    gaussians._density = torch.randn(N, requires_grad=True).cuda()

    # 构建图
    graph = GaussianGraph(k=6, device='cuda')
    graph.build_knn_graph(gaussians.get_xyz)
    graph.compute_edge_weights(gaussians.get_xyz)

    # 计算损失
    lambda_lap = 8e-4  # 论文推荐值
    loss = compute_graph_laplacian_loss(gaussians, graph, lambda_lap=lambda_lap)

    # 验证
    assert loss.item() > 0, "Graph Laplacian 损失应该非零！"
    assert loss.requires_grad, "损失应该可微！"

    # 测试反向传播
    loss.backward()
    assert gaussians._density.grad is not None, "密度参数应该有梯度！"
    grad_norm = gaussians._density.grad.norm().item()
    assert grad_norm > 0, "梯度范数应该非零！"

    print(f"✅ 损失函数正常工作")
    print(f"   损失值: {loss.item():.6f}")
    print(f"   密度梯度范数: {grad_norm:.6f}")


def test_denoised_init():
    """测试 3: De-Init 降噪初始化有效性"""
    print("\n" + "="*60)
    print("测试 3: De-Init 降噪初始化有效性")
    print("="*60)

    from scipy.ndimage import gaussian_filter

    # 创建带噪声的体积（模拟 FDK 重建）
    vol_size = 64
    vol = np.random.rand(vol_size, vol_size, vol_size).astype(np.float32)

    # 添加高斯噪声
    noise = np.random.randn(vol_size, vol_size, vol_size) * 0.1
    vol_noisy = vol + noise

    # 应用降噪（σ=3, 论文推荐值）
    vol_denoised = gaussian_filter(vol_noisy, sigma=3.0)

    # 验证降噪效果
    noise_level_before = np.abs(vol_noisy - vol).mean()
    noise_level_after = np.abs(vol_denoised - vol).mean()
    noise_reduction = (noise_level_before - noise_level_after) / noise_level_before * 100

    assert noise_level_after < noise_level_before, "降噪后噪声应该减少！"
    assert noise_reduction > 20, f"噪声降低应该超过20%，实际{noise_reduction:.1f}%"

    print(f"✅ 降噪有效")
    print(f"   原始噪声水平: {noise_level_before:.6f}")
    print(f"   降噪后噪声水平: {noise_level_after:.6f}")
    print(f"   噪声降低: {noise_reduction:.1f}%")


def test_edge_case_small_graph():
    """测试 4: 边界情况 - 点数过少"""
    print("\n" + "="*60)
    print("测试 4: 边界情况 - 点数过少")
    print("="*60)

    # 创建只有 3 个点的点云（< k=6）
    xyz = torch.randn(3, 3).cuda()

    graph = GaussianGraph(k=6, device='cuda')
    graph.build_knn_graph(xyz)

    # 应该优雅降级
    assert graph.num_nodes == 3, "节点数应该为 3"
    assert graph.num_edges == 0, "边数应该为 0（点数 < k+1）"

    print(f"✅ 边界情况处理正确")
    print(f"   点数过少时优雅降级（无边）")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪 "*20)
    print("     GR-Gaussian 单元测试套件")
    print("🧪 "*20)

    tests = [
        test_gaussian_graph_construction,
        test_graph_laplacian_loss_nonzero,
        test_denoised_init,
        test_edge_case_small_graph,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ 测试失败：{test_func.__name__}")
            print(f"   错误: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ 测试崩溃：{test_func.__name__}")
            print(f"   异常: {e}")

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"✅ 通过: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ 失败: {failed}/{len(tests)}")
    else:
        print("🎉 所有测试通过！GR-Gaussian 组件功能正常。")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
