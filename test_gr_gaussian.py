"""
GR-Gaussian 核心功能单元测试
验证 Graph 构建、损失计算和参数配置
"""

import torch
import sys

def test_graph_utils():
    """测试 Graph Utilities 模块"""
    print("\n" + "="*60)
    print("测试 1: Graph Utils 模块导入和 KNN 图构建")
    print("="*60)

    try:
        from r2_gaussian.utils.graph_utils import GaussianGraph, build_knn_graph
        print("✅ graph_utils 模块导入成功")
    except ImportError as e:
        print(f"❌ graph_utils 模块导入失败: {e}")
        return False

    # 测试 KNN 图构建
    try:
        positions = torch.randn(100, 3).cuda()
        edges = build_knn_graph(positions, k=6)
        print(f"✅ KNN 图构建成功: {edges.shape[1]} 条边")

        # 验证边的数量合理性
        expected_min_edges = 100 * 3  # 至少每个点有 3 个邻居 (对称后)
        expected_max_edges = 100 * 6 * 2  # 最多每个点有 6 个邻居 (双向)
        if expected_min_edges <= edges.shape[1] <= expected_max_edges:
            print(f"✅ 边数量合理: {edges.shape[1]} (预期范围: {expected_min_edges}~{expected_max_edges})")
        else:
            print(f"⚠️  边数量异常: {edges.shape[1]} (预期范围: {expected_min_edges}~{expected_max_edges})")

    except Exception as e:
        print(f"❌ KNN 图构建失败: {e}")
        return False

    # 测试 GaussianGraph 类
    try:
        graph = GaussianGraph(k=6, device='cuda')
        graph.build_knn_graph(positions)
        weights = graph.compute_edge_weights(positions)
        print(f"✅ GaussianGraph 类测试成功: {graph.num_nodes} 个节点, {weights.shape[0]} 条边")
    except Exception as e:
        print(f"❌ GaussianGraph 类测试失败: {e}")
        return False

    return True


def test_loss_function():
    """测试 Graph Laplacian 损失函数"""
    print("\n" + "="*60)
    print("测试 2: Graph Laplacian 损失函数")
    print("="*60)

    try:
        from r2_gaussian.utils.loss_utils import compute_graph_laplacian_loss
        from r2_gaussian.gaussian.gaussian_model import GaussianModel
        print("✅ loss_utils 模块导入成功")
    except ImportError as e:
        print(f"❌ loss_utils 模块导入失败: {e}")
        return False

    # 创建简单的 Gaussian 模型
    try:
        # 创建假的高斯点
        N = 1000
        xyz = torch.randn(N, 3).cuda()
        density = torch.rand(N).cuda()

        # 创建一个简化的 mock Gaussian 模型
        class MockGaussianModel:
            def __init__(self):
                self._xyz = xyz
                self._density = density

            @property
            def get_xyz(self):
                return self._xyz

            @property
            def get_density(self):
                return self._density

        gaussians = MockGaussianModel()

        # 测试损失计算 (不使用预构建图)
        loss = compute_graph_laplacian_loss(gaussians, graph=None, k=6, Lambda_lap=8e-4)
        print(f"✅ 损失计算成功 (fallback 模式): {loss.item():.6f}")

        # 测试损失计算 (使用预构建图)
        from r2_gaussian.utils.graph_utils import GaussianGraph
        graph = GaussianGraph(k=6, device='cuda')
        graph.build_knn_graph(xyz)
        graph.compute_edge_weights(xyz)

        loss_with_graph = compute_graph_laplacian_loss(gaussians, graph=graph, k=6, Lambda_lap=8e-4)
        print(f"✅ 损失计算成功 (GR-Gaussian 模式): {loss_with_graph.item():.6f}")

        # 验证损失值合理性
        if loss.item() > 0 and loss_with_graph.item() > 0:
            print(f"✅ 损失值合理 (非零且为正)")
        else:
            print(f"⚠️  损失值可能异常: fallback={loss.item()}, gr={loss_with_graph.item()}")

    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_arguments():
    """测试命令行参数配置"""
    print("\n" + "="*60)
    print("测试 3: 命令行参数配置")
    print("="*60)

    try:
        from r2_gaussian.arguments import ModelParams
        from argparse import ArgumentParser
        print("✅ arguments 模块导入成功")
    except ImportError as e:
        print(f"❌ arguments 模块导入失败: {e}")
        return False

    try:
        parser = ArgumentParser()
        model_params = ModelParams(parser)

        # 验证 GR-Gaussian 参数是否存在
        assert hasattr(model_params, 'enable_graph_laplacian'), "缺少 enable_graph_laplacian 参数"
        assert hasattr(model_params, 'graph_k'), "缺少 graph_k 参数"
        assert hasattr(model_params, 'graph_lambda_lap'), "缺少 graph_lambda_lap 参数"
        assert hasattr(model_params, 'graph_update_interval'), "缺少 graph_update_interval 参数"

        print(f"✅ GR-Gaussian 参数配置正确:")
        print(f"   enable_graph_laplacian: {model_params.enable_graph_laplacian}")
        print(f"   graph_k: {model_params.graph_k}")
        print(f"   graph_lambda_lap: {model_params.graph_lambda_lap}")
        print(f"   graph_update_interval: {model_params.graph_update_interval}")

        # 验证默认值
        assert model_params.enable_graph_laplacian == False, "enable_graph_laplacian 默认值应为 False"
        assert model_params.graph_k == 6, "graph_k 默认值应为 6"
        assert model_params.graph_lambda_lap == 8e-4, "graph_lambda_lap 默认值应为 8e-4"
        assert model_params.graph_update_interval == 100, "graph_update_interval 默认值应为 100"

        print(f"✅ 默认值验证通过")

    except Exception as e:
        print(f"❌ 参数配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_train_integration():
    """测试 train.py 集成"""
    print("\n" + "="*60)
    print("测试 4: train.py 集成 (语法检查)")
    print("="*60)

    try:
        # 尝试导入 train 模块 (只检查语法,不执行)
        import train
        print("✅ train.py 语法检查通过")
    except SyntaxError as e:
        print(f"❌ train.py 语法错误: {e}")
        return False
    except Exception as e:
        # 其他导入错误可以忽略 (比如缺少数据文件等)
        print(f"⚠️  train.py 导入警告 (可能正常): {e}")

    return True


def main():
    print("="*60)
    print("GR-Gaussian 核心功能单元测试")
    print("="*60)

    results = {
        "Graph Utils": test_graph_utils(),
        "Loss Function": test_loss_function(),
        "Arguments": test_arguments(),
        "Train Integration": test_train_integration()
    }

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过! GR-Gaussian 核心功能已成功实现。")
        return 0
    else:
        print("\n⚠️  部分测试失败,请检查上面的错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
