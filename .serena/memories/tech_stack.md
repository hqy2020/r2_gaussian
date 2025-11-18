# R²-Gaussian 技术栈

## 编程语言
- **Python 3.9**：主要开发语言
- **CUDA C++**：自定义 CUDA 内核和扩展

## 深度学习框架
- **PyTorch 1.12.1**：主深度学习框架
- **CUDA 11.6**：GPU 加速计算
- **TorchVision 0.13.1**：图像处理工具

## 核心依赖库

### 科学计算
- **NumPy 1.24.1**：数值计算
- **Mkl 2024.0**：数学内核库
- **SciPy**（通过 scikit-image）：科学计算

### 3D 处理
- **Open3D 0.18.0**：3D 数据处理
- **PyVista**：3D 可视化
- **PLYFile**：点云文件格式支持

### 医学影像
- **TIGRE 2.3**：CT 数据生成和 FDK 重建
- **PyDICOM**：DICOM 医学影像格式支持
- **SimpleITK**：医学图像处理

### 图像处理
- **OpenCV (opencv-python)**：图像处理
- **scikit-image**：图像算法
- **Matplotlib**：可视化
- **tifffile & imagecodecs**：图像编解码

### 工具库
- **TensorBoard & TensorBoardX**：训练可视化
- **tqdm**：进度条
- **PyYAML**：配置文件解析
- **Cython 0.29.36**：性能优化

## 自定义 CUDA 扩展

### 1. xray-gaussian-rasterization-voxelization
- **位置**：`r2_gaussian/submodules/xray-gaussian-rasterization-voxelization`
- **功能**：X 射线 Gaussian 光栅化和体素化
- **语言**：CUDA C++ + Python bindings
- **安装**：`pip install -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization`

### 2. simple-knn
- **位置**：`r2_gaussian/submodules/simple-knn`
- **功能**：快速 K 近邻搜索
- **语言**：CUDA C++ + Python bindings
- **安装**：`pip install -e r2_gaussian/submodules/simple-knn`

## 环境配置
- **Conda 环境名**：`r2_gaussian_new`
- **GPU 要求**：NVIDIA RTX 3090 或同等性能
- **操作系统**：Ubuntu 20.04 (测试环境)
- **环境配置文件**：`environment.yml`