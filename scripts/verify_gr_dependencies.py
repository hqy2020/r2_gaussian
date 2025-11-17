"""
验证 GR-Gaussian 所需依赖的安装和兼容性
"""

import sys

def check_scipy():
    try:
        import scipy
        from scipy.ndimage import gaussian_filter
        print(f"✅ scipy {scipy.__version__}")
        return True
    except ImportError as e:
        print(f"❌ scipy not found: {e}")
        return False

def check_torch_geometric():
    try:
        import torch
        from torch_geometric.nn import knn_graph
        import torch_geometric

        # 测试 CUDA 兼容性
        x = torch.randn(100, 3).cuda()
        edge_index = knn_graph(x, k=6)

        print(f"✅ PyTorch Geometric {torch_geometric.__version__} (CUDA compatible)")
        print(f"   Test: 100 points → {edge_index.shape[1]} edges")
        return True
    except ImportError as e:
        print(f"❌ PyTorch Geometric not found: {e}")
        return False
    except RuntimeError as e:
        print(f"⚠️  PyG installed but CUDA test failed: {e}")
        return False

def check_yaml():
    try:
        import yaml
        print(f"✅ PyYAML")
        return True
    except ImportError:
        print(f"❌ PyYAML not found (needed for config files)")
        return False

if __name__ == "__main__":
    print("="*60)
    print("GR-Gaussian Dependency Check")
    print("="*60)

    checks = {
        "scipy": check_scipy(),
        "torch_geometric": check_torch_geometric(),
        "yaml": check_yaml()
    }

    print("\n" + "="*60)
    if all(checks.values()):
        print("🎉 All dependencies satisfied!")
        sys.exit(0)
    else:
        print("⚠️  Some dependencies missing, please install:")
        if not checks["scipy"]:
            print("   pip install scipy")
        if not checks["torch_geometric"]:
            print("   See scripts/install_torch_geometric.sh")
        if not checks["yaml"]:
            print("   pip install pyyaml")
        sys.exit(1)
