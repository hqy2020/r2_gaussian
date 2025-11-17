#!/bin/bash
# PyTorch Geometric 安装脚本
# 根据当前 PyTorch 版本自动选择兼容的 PyG 版本

echo "🔍 检测 PyTorch 环境..."

TORCH_VERSION=$(/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python -c "import torch; print(torch.version.cuda)")

echo "Detected PyTorch version: $TORCH_VERSION"
echo "Detected CUDA version: $CUDA_VERSION"

# PyTorch 1.12.1 + CUDA 11.3 对应的 PyG 版本
echo ""
echo "📦 安装 PyTorch Geometric for PyTorch $TORCH_VERSION + CUDA $CUDA_VERSION..."

# 安装 PyG (使用官方推荐的方式)
/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/pip install torch-geometric

# 安装依赖库 (对于 PyTorch 1.12.1 + CUDA 11.3)
/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

echo ""
echo "✅ 验证安装..."
/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python -c "
from torch_geometric.nn import knn_graph
import torch
import torch_geometric

print('PyTorch Geometric version:', torch_geometric.__version__)

# 测试 CUDA 兼容性
x = torch.randn(100, 3).cuda()
edge_index = knn_graph(x, k=6)
print('✅ PyTorch Geometric installed successfully')
print(f'Test: 100 points → {edge_index.shape[1]} edges (expected ~600)')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 PyTorch Geometric 安装完成!"
else
    echo ""
    echo "❌ 安装失败,请检查错误信息"
    exit 1
fi
