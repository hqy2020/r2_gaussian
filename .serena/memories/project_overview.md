# R²-Gaussian 项目概述

## 项目简介
**R²-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction**

这是一个基于 3D Gaussian Splatting 的 CT 断层扫描重建项目，已被 NeurIPS 2024 接收。

## 核心目标
- **快速直接的 CT 重建**：使用 3D Gaussian Splatting 进行 CT 体积重建
- **稀疏视角重建**：支持 3/6/9 个视角的极稀疏场景重建
- **医学影像应用**：应用于医学 CT 影像重建领域

## 研究场景
- **训练数据格式**：NAF 格式（兼容 SAX-NeRF）和 NeRF 格式
- **稀疏视角配置**：3 views, 6 views, 9 views
- **目标器官**：胸部、腹部、足部等医学 CT 数据

## 关键特性
1. **深度约束 (Depth Constraint)**：使用 Pearson 相关系数优化深度一致性
2. **伪标签学习 (Pseudo Label Learning)**：从额外视角生成监督信号
3. **FDK 初始化**：基于 FDK 算法的点云初始化
4. **多高斯场景支持**：可配置单/多高斯场景

## 论文信息
- **会议**：NeurIPS 2024
- **arXiv**：https://arxiv.org/abs/2405.20693
- **项目主页**：https://ruyi-zha.github.io/r2_gaussian/r2_gaussian.html