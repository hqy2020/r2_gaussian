#!/bin/bash
# GPU 服务器一键环境部署脚本
# 在服务器上执行: bash setup_server.sh
set -euo pipefail

echo "=== GPU Research Server Setup ==="
echo "Start: $(date)"

# === 1. 系统包 ===
echo ""
echo "[1/7] Installing system packages..."
apt update && apt install -y \
    build-essential git wget curl tmux htop unzip \
    libgl1-mesa-glx libglib2.0-0 \
    ninja-build cmake

# === 2. 检测 GPU ===
echo ""
echo "[2/7] Detecting GPU..."
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

# 根据 GPU 型号决定 CUDA 版本
CUDA_VERSION="11.6"
PYTORCH_VERSION="1.12.1"
TORCHVISION_VERSION="0.13.1"

if echo "$GPU_NAME" | grep -qiE "4090|4080|4070|H100|A100|L40|RTX 40"; then
    echo "Detected 40-series or newer GPU. Using CUDA 11.8 + PyTorch 2.0"
    CUDA_VERSION="11.8"
    PYTORCH_VERSION="2.0.0"
    TORCHVISION_VERSION="0.15.0"
fi

echo "Selected: CUDA=$CUDA_VERSION, PyTorch=$PYTORCH_VERSION"

# === 3. Miniconda ===
echo ""
echo "[3/7] Installing Miniconda..."
if ! command -v conda &> /dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    rm /tmp/miniconda.sh
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "Miniconda installed."
else
    echo "Conda already installed."
    eval "$(conda shell.bash hook)"
fi

# === 4. Conda 环境 ===
echo ""
echo "[4/7] Creating conda environment r2gs..."
if conda env list | grep -q "r2gs"; then
    echo "Environment r2gs already exists."
else
    conda create -n r2gs python=3.9 -y
fi
conda activate r2gs

# === 5. PyTorch ===
echo ""
echo "[5/7] Installing PyTorch..."
if [[ "$CUDA_VERSION" == "11.6" ]]; then
    conda install pytorch=${PYTORCH_VERSION} torchvision=${TORCHVISION_VERSION} cudatoolkit=11.6 -c pytorch -y
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    conda install pytorch=${PYTORCH_VERSION} torchvision=${TORCHVISION_VERSION} pytorch-cuda=11.8 -c pytorch -c nvidia -y
fi

# 验证 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# === 6. Pip 依赖 ===
echo ""
echo "[6/7] Installing pip dependencies..."
pip install \
    opencv-python matplotlib pydicom SimpleITK \
    open3d==0.18.0 plyfile tensorboard tensorboardX \
    pyyaml tqdm scikit-image Cython==0.29.36 \
    pyvista tifffile imagecodecs tabulate \
    numpy==1.24.1

# TIGRE
echo "Installing TIGRE..."
pip install tigre==2.3 || {
    echo "TIGRE pip install failed, trying source build..."
    cd /tmp
    git clone https://github.com/CERN/TIGRE.git
    cd TIGRE/Python
    python setup.py install
    cd /root
}

# === 7. 编译子模块 ===
echo ""
echo "[7/7] Compiling CUDA submodules..."
cd /root/r2_gaussian

# simple-knn
if [[ -d "r2_gaussian/submodules/simple-knn" ]]; then
    echo "Compiling simple-knn..."
    cd r2_gaussian/submodules/simple-knn
    pip install -e .
    cd /root/r2_gaussian
else
    echo "simple-knn not found, cloning..."
    mkdir -p r2_gaussian/submodules
    cd r2_gaussian/submodules
    git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
    cd simple-knn
    pip install -e .
    cd /root/r2_gaussian
fi

# xray-gaussian-rasterization-voxelization
if [[ -d "r2_gaussian/submodules/xray-gaussian-rasterization-voxelization" ]]; then
    echo "Compiling xray-gaussian-rasterization-voxelization..."
    cd r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
    pip install -e .
    cd /root/r2_gaussian
else
    echo "WARNING: xray-gaussian-rasterization-voxelization not found!"
fi

# === 创建目录结构 ===
echo ""
echo "Creating experiment directories..."
mkdir -p /root/experiments/results /root/experiments/logs /root/comparison_methods
mkdir -p /root/r2_gaussian/data

# === 验证 ===
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "=== Setup Complete ==="
echo "End: $(date)"
echo ""
echo "Next steps:"
echo "  1. Upload data: rsync -avz -e 'ssh -p 23' data/ root@<IP>:/root/r2_gaussian/data/"
echo "  2. Test run: python train.py -s data/foot_50_3views.pickle -m output/test --gaussiansN 1 --iterations 100"
echo "  3. Clone comparison methods: cd /root/comparison_methods && git clone ..."
