# R²-Gaussian 项目进度记录

## 2025-11-20 实验记录

### 实验: Foot 3视角 + IPSM 增强

**实验信息:**
- **实验名称**: `2025_11_20_15_02_foot_3views_ipsm`
- **数据集**: Foot, 3 views (稀疏场景)
- **基准方法**: R²-Gaussian
- **增强技术**: IPSM (Image-guided Pseudo-view Synthesis Module)
- **训练迭代数**: 30,000
- **GPU**: NVIDIA RTX A6000 (GPU 0)
- **进程 PID**: 2254643

**IPSM 配置:**
- IPSM 激活区间: 迭代 2000-9500
- λ_IPSM (扩散损失权重): 1.0
- λ_depth (深度正则化权重): 0.5
- λ_geo (几何一致性权重): 4.0
- 深度估计器: DPT_Hybrid (MiDaS)

**技术修复记录:**
1. **问题**: X-ray Gaussian Rasterizer 不支持 `colors_precomp` 参数用于深度渲染
   - **原因**: R²-Gaussian 使用的是专门的 X-ray CT 光栅化器,只渲染投影密度,不支持颜色通道
   - **解决方案**: 重新设计深度计算逻辑
     - 通过相机变换矩阵将 3D 点变换到相机坐标系
     - 提取 Z 坐标作为深度值
     - 使用可见 Gaussians 的中值深度初始化深度图
   - **修改文件**: `r2_gaussian/gaussian/render_query.py:154-184`

**训练状态:**
- 启动时间: 2025-11-20 15:02:53
- 当前状态: ✅ 正常运行
- 训练速度: ~50-55 it/s
- Loss 趋势: 从 3.3e-01 降至 ~5.0e-03 (前1300次迭代)
- Gaussian 点数: ~5.2e+04

**预期完成时间:**
- 预计运行时长: 约 2-3 小时 (30,000次迭代)
- 预计完成时间: 2025-11-20 17:30 左右

**目标基准对比:**
- Foot 3视角 SOTA 基准: PSNR 28.4873, SSIM 0.9005
- 本次实验旨在验证 IPSM 技术能否提升稀疏视角重建质量

**监控命令:**
```bash
# 查看训练进度
tail -f output/2025_11_20_15_02_foot_3views_ipsm_train.log

# 查看进程状态
ps aux | grep 2254643

# 查看GPU使用情况
nvidia-smi
```

---

## 历史记录

_（之前的实验记录将添加在此）_
