# 知识库 (Knowledge Base)

> 本文档记录科研助手团队在项目过程中积累的所有知识、经验和教训

---

## 📚 索引

- [成功案例](#成功案例)
- [失败教训](#失败教训)
- [技术知识点](#技术知识点)
- [医学领域知识](#医学领域知识)
- [最佳实践](#最佳实践)

---

## ✅ 成功案例

### 模板
```markdown
### [日期] 案例名称
- **创新点来源：** 论文链接/名称
- **实现方法：** 简要描述
- **性能提升：** PSNR +X dB / SSIM +X
- **关键决策：** 列出 2-3 个关键决策点
- **可复用组件：** 代码路径或方法
- **参考文档：** 相关分析报告链接
```

### [2025-11-17] CoR-GS Stage 1 多视点协同正则化
- **创新点来源：** Co-Regularization Gaussian Splatting (CoR-GS) - Stage 1 Disagreement Metrics
- **实现方法：**
  - 基于PyTorch3D KNN的Geometry Disagreement计算
  - 多视点Rendering Disagreement度量
  - 集成到训练损失函数（lambda = 0.01）
- **性能提升：**
  - 3 views: -0.40 dB (需参数优化)
  - 6 views: +5.24 dB / SSIM +0.0416 (4.6%)
  - 9 views: +6.24 dB / SSIM +0.0532 (5.9%)
- **关键决策：**
  1. 采用渐进式Stage实现策略，验证一个后再进行下一个
  2. 多视点协同约束在视角充足时有显著优势
  3. 参数自适应化（lambda根据视角数调整）是改进3-views的关键
- **可复用组件：**
  - `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/disagreement_metrics.py`
  - PyTorch3D KNN加速（10-20x性能提升）
  - Disagreement Loss集成框架
- **参考文档：** `cc-agent/records/foot_369_corgs_results_2025_11_17.md`
- **后续优化建议：**
  - 调整3-views权重参数 (lambda=0.001~0.01 range)
  - 实现adaptive regularization (根据disagreement强度调整权重)
  - 在其他器官数据集验证（chest, head, abdomen）

---

## ❌ 失败教训

### 模板
```markdown
### [日期] 失败案例名称
- **尝试目标：** 想要实现什么
- **失败原因：** 根本原因分析
- **错误假设：** 事后发现的错误假设
- **性能影响：** PSNR -X dB 或其他指标下降
- **教训总结：** 下次如何避免
- **参考文档：** result_analysis.md 路径
```

### 示例记录区
*（待添加首个失败教训）*

---

## 🧠 技术知识点

### 3D Gaussian Splatting 相关
- **球谐函数 (Spherical Harmonics)：**
  - 用于表示视角相关的颜色变化
  - 阶数越高表达能力越强，但计算开销增加

- **自适应密度控制：**
  - Clone：分裂大梯度区域的 Gaussians
  - Prune：移除透明度低的 Gaussians
  - 典型阈值：opacity < 0.005

### R²-Gaussian 特有技术

- **X 射线投影 vs RGB 渲染：**
  - R²-Gaussian 使用穿透式投影（累积密度），而非标准 3DGS 的 alpha 混合
  - 自定义 CUDA 算子：`xray-gaussian-rasterization-voxelization`
  - 支持锥束和平行束几何

- **坐标归一化：**
  - 整个场景（扫描仪 + CT 体）归一化到 [-1,1]³ 空间
  - 高斯尺度参数化为体积百分比（避免绝对尺度导致的数值不稳定）
  - 关键参数：`scale_min=0.0005`, `scale_max=0.5`

- **FDK 初始化：**
  - 使用 TIGRE 工具箱的 FDK 算法从稀疏投影重建初始体数据
  - 采样高密度区域（`density_thresh=0.05`）作为初始点云
  - 密度缩放因子（`density_rescale=0.15`）补偿遮挡效应
  - **初始化质量直接决定成败** - 建议用 `--evaluate` 检查初始 PSNR

- **密化策略：**
  - 基于梯度的分裂：`densify_grad_threshold=0.00005`
  - 基于尺度的分裂：`densify_scale_threshold=0.1`（体积的 10%）
  - 剪枝阈值：`density_min_threshold=0.00001`
  - 密化周期：每 100 次迭代，从 500 迭代开始，到 15,000 迭代结束

- **损失函数组合：**
  - L1 + SSIM（`lambda_dssim=0.25`）用于 2D 投影监督
  - TV 正则化（`lambda_tv=0.05`）用于 3D 体平滑
  - 可选：深度损失、光度一致性损失（IPSM）

### PyTorch 优化技巧
- **混合精度训练：** 使用 `torch.cuda.amp` 可加速 30-50%
- **梯度累积：** 当 GPU 内存不足时的有效策略
- **学习率调度：** ExponentialLR 在 3DGS 中效果优于 CosineAnnealing
- **R²-Gaussian 学习率设置：**
  - 位置：0.0002 → 0.00002（30,000 步）
  - 密度：0.01 → 0.001
  - 尺度：0.005 → 0.0005
  - 旋转：0.001 → 0.0001

### CUDA 优化经验
- **Tile-based Rasterization：** 3DGS 的核心渲染方式
- **并行排序：** 需要高效的 radix sort 实现
- **X 射线投影加速：** R²-Gaussian 的 CUDA 内核针对穿透式累积优化

---

## 🏥 医学领域知识

### CT 成像特性
- **Hounsfield Unit (HU)：**
  - 空气: -1000 HU
  - 水: 0 HU
  - 骨骼: +400 ~ +1000 HU

- **窗宽窗位 (Window Level/Width)：**
  - 肺窗：WL=-600, WW=1500
  - 软组织窗：WL=40, WW=400
  - 骨窗：WL=400, WW=1800

### 临床评估标准
- **诊断可用性：** 优先于纯数值指标（PSNR/SSIM）
- **伪影类型：**
  - 金属伪影：高密度物体周围的条纹
  - 运动伪影：患者移动导致的模糊
  - 噪声：低剂量扫描的颗粒感

### 稀疏视角挑战
- **角度欠采样：** 少于 180° 覆盖会导致严重伪影
- **投影数不足：** 通常需要 ≥180 个投影，稀疏场景可能仅 30-60 个

---

## 🎯 最佳实践

### 实验设计
1. **消融实验原则：** 每次只改变一个变量
2. **基线对比：** 始终与原始 baseline 和 SOTA 方法对比
3. **多数据集验证：** 至少在 2-3 个数据集上验证泛化性

### 代码管理
1. **分支策略：**
   - `main`：稳定 baseline
   - `dev-feature-name`：新功能开发
   - 合并前必须通过测试

2. **提交规范：**
   ```
   [角色] 简要描述

   - 详细说明修改内容
   - 关联的 issue/实验编号
   ```

3. **配置文件版本化：** 所有超参数用 YAML/JSON 管理，避免硬编码

### 文档维护
1. **及时记录：** 完成任务后立即更新 record.md
2. **交叉引用：** 使用相对路径链接相关文档
3. **定期归档：** 每月整理 records/ 下的临时文件

---

## 🔗 快速链接

- [决策日志](./decision_log.md) - 查看历史决策
- [项目时间线](./project_timeline.md) - 查看进度
- [3DGS 专家分析报告](../3dgs_expert/analyses/)
- [实验结果汇总](../experiments/results/)

---

## 🔧 工具配置经验

### MCP 服务器安装

**必需工具：**
1. **@modelcontextprotocol/server-arxiv** - 论文搜索下载
2. **@modelcontextprotocol/server-github** - 代码调研（需 GitHub Token）
3. **@modelcontextprotocol/server-filesystem** - 文件系统访问
4. **@modelcontextprotocol/server-sqlite** - 实验数据库
5. **@modelcontextprotocol/server-brave-search** - 网络搜索（可选）

**配置位置：**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`

**API 密钥获取：**
- GitHub Token: https://github.com/settings/tokens（需 `repo` 权限）
- Brave Search: https://brave.com/search/api/（免费 2000 次/月）

**验证方法：**
```
重启 Claude Desktop 后，尝试："请使用 arXiv 工具搜索 3D Gaussian Splatting"
```

### 环境安装踩坑记录

**TIGRE 安装问题：**
- 可能需要 `--no-build-isolation` 标志
- 需要先安装 Cython：`pip install Cython==0.29.36`
- Ubuntu 需要 gcc/g++ 编译器

**CUDA 扩展编译失败：**
- 检查 CUDA 版本匹配：`nvcc --version` 应为 11.6
- 确保 PyTorch CUDA 版本一致：`torch.version.cuda`
- 子模块未初始化：`git submodule update --init --recursive`

**常见错误：**
```bash
# 错误：ModuleNotFoundError: No module named 'simple_knn._C'
# 解决：重新安装子模块
cd r2_gaussian/submodules/simple-knn
pip install -e .

# 错误：xray_gaussian_rasterization_voxelization 编译失败
# 解决：检查 CUDA_HOME 环境变量
export CUDA_HOME=/usr/local/cuda-11.6
```

---

