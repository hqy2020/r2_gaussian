# IPSM集成工作记录

> **项目**: IPSM (Inline Prior Guided Score Matching) 集成到 R²-Gaussian
> **创建时间**: 2025-11-20
> **最后更新**: 2025-11-20
> **状态**: ✅ 核心功能已完成，待实验验证

---

## 📋 已完成工作

### ✅ 1. 核心模块实现

#### 1.1 深度估计器 (`r2_gaussian/utils/depth_estimator.py`)
- **状态**: ✅ 已完成
- **功能**: DPT单目深度估计
- **特性**:
  - 支持CT灰度图像→RGB转换
  - 全局单例模式避免重复加载
  - 占位符模式（如果DPT加载失败）
- **代码行数**: ~172行

#### 1.2 扩散模型封装 (`r2_gaussian/utils/diffusion_utils.py`)
- **状态**: ✅ 已完成
- **功能**: SD Inpainting延迟加载
- **特性**:
  - 动态加载/卸载（节省显存）
  - FP16推理
  - IPSM双阶段score matching
  - CT图像转RGB功能
- **代码行数**: ~238行

#### 1.3 损失函数扩展 (`r2_gaussian/utils/loss_utils.py`)
- **状态**: ✅ 已完成
- **新增函数**:
  - `pearson_correlation_loss()`: Pearson深度正则化 (~60行)
  - `geometry_consistency_loss()`: Masked L1 loss (~21行)
  - `ipsm_depth_regularization()`: 组合seen/unseen深度loss (~35行)
- **总计**: ~116行新增代码

#### 1.4 X-ray Warping (`r2_gaussian/utils/ipsm_utils.py`)
- **状态**: ✅ 已完成
- **功能**: 体素反投影warping
- **类**: `XRayIPSMWarping`
- **核心方法**:
  - `warp_via_voxel_projection()`: 主warping函数
  - `sample_nearby_viewpoint()`: 采样伪视角
- **代码行数**: ~267行

#### 1.5 Render深度输出 (`r2_gaussian/gaussian/render_query.py`)
- **状态**: ✅ 已完成
- **修改**: render()函数新增深度输出
- **实现方式**: 使用Z坐标作为"颜色"渲染深度图
- **返回**: render_pkg字典新增`"depth"`键
- **代码行数**: ~14行新增

#### 1.6 命令行参数 (`r2_gaussian/arguments/__init__.py`)
- **状态**: ✅ 已完成
- **新增类**: `IPSMParams(ParamGroup)`
- **参数数量**: 11个可配置参数
  - `enable_ipsm`: 核心开关
  - `ipsm_start_iter`, `ipsm_end_iter`: 训练区间
  - `lambda_ipsm`, `lambda_ipsm_depth`, `lambda_ipsm_geo`: 损失权重
  - `ipsm_eta_r`, `ipsm_eta_d`: 子参数
  - `ipsm_mask_tau`, `ipsm_mask_tau_geo`: Mask阈值
  - `ipsm_cfg_scale`: CFG guidance强度
  - `ipsm_pseudo_angle_range`: 伪视角角度范围
  - `sd_model_path`: SD模型路径
- **代码行数**: ~31行

#### 1.7 训练流程集成 (`train.py`)
- **状态**: ✅ 已完成
- **集成内容**:
  - IPSM模块初始化 (iter 0)
  - 动态加载扩散模型 (iter 2K)
  - IPSM训练循环 (iter 2K-9.5K)
    - 伪视角采样
    - 渲染伪视角
    - Inverse warping
    - 深度正则化
    - 几何一致性
    - Score distillation
  - 动态卸载扩散模型 (iter 9.5K)
- **代码行数**: ~84行新增

### ✅ 2. 训练脚本

#### 2.1 快速验证脚本 (`run_ipsm_验证.sh`)
- **状态**: ✅ 已完成
- **功能**: 500迭代快速验证
- **目的**: 确认IPSM代码可运行，无crash
- **时间**: 约5-10分钟

#### 2.2 完整训练脚本 (`run_ipsm_完整训练.sh`)
- **状态**: ✅ 已完成
- **功能**: 30,000迭代完整训练
- **目的**: 与baseline对比，验证IPSM效果
- **时间**: 约1-2小时
- **数据集**: Foot-3视角

### ✅ 3. 文档

#### 3.1 实施指南 (`cc-agent/ipsm/IPSM集成实现指南.md`)
- **状态**: ✅ 已完成
- **内容**: 详细实施文档，包含代码模板和使用说明
- **行数**: ~507行

#### 3.2 完成报告 (`cc-agent/ipsm/IPSM集成完成报告.md`)
- **状态**: ✅ 已完成
- **内容**: 集成概览、快速开始、参数详解、预期结果
- **行数**: ~419行

---

## 📊 代码统计

### 新增文件 (6个)
```
r2_gaussian/utils/
├── depth_estimator.py          # 172行
├── diffusion_utils.py          # 238行
└── ipsm_utils.py               # 267行

根目录/
├── run_ipsm_验证.sh            # 39行
└── run_ipsm_完整训练.sh        # 72行
```

### 修改文件 (4个)
```
r2_gaussian/utils/loss_utils.py
  + pearson_correlation_loss()        (~60行)
  + geometry_consistency_loss()        (~21行)
  + ipsm_depth_regularization()       (~35行)

r2_gaussian/gaussian/render_query.py
  + depth渲染逻辑                     (~14行)

r2_gaussian/arguments/__init__.py
  + class IPSMParams(ParamGroup)      (~31行)

train.py
  + IPSM初始化代码                    (~22行)
  + IPSM训练循环代码                  (~59行)
  + 参数导入和传递                     (~3行)
```

**总计**:
- 新增代码: ~677行
- 修改代码: ~129行
- **总计**: ~806行

---

## ⏳ 未完成工作

### 🔄 1. 实验验证

#### 1.1 快速验证 (500迭代)
- **状态**: ⏳ 待执行
- **检查要点**:
  - [ ] 程序正常启动，无import错误
  - [ ] iter 100成功加载扩散模型
  - [ ] IPSM loss正常计算（不是NaN/Inf）
  - [ ] 可以看到loss keys: `ipsm_depth`, `ipsm_geo`, `ipsm_sd`
  - [ ] iter 400成功卸载扩散模型
  - [ ] Total loss正常下降
  - [ ] 无CUDA OOM错误

#### 1.2 完整训练 (30K迭代)
- **状态**: ⏳ 待执行
- **检查要点**:
  - [ ] 训练完成无crash
  - [ ] TensorBoard显示IPSM loss曲线
  - [ ] PSNR和SSIM指标正常记录
  - [ ] 模型checkpoint正常保存

#### 1.3 结果评估
- **状态**: ⏳ 待执行
- **检查要点**:
  - [ ] test.py成功运行
  - [ ] PSNR > 28.4873 (baseline)
  - [ ] SSIM > 0.9005 (baseline)
  - [ ] 渲染图像质量目视正常

### 🔄 2. TensorBoard监控

- **状态**: ⏳ 待验证
- **需要监控的曲线**:
  - [ ] `loss/total`: 总损失
  - [ ] `loss/render`: 渲染损失
  - [ ] `ipsm/depth_loss`: IPSM深度正则化
  - [ ] `ipsm/geo_loss`: 几何一致性损失
  - [ ] `ipsm/sd_loss`: Score distillation损失
  - [ ] `metrics/psnr_2d`: 2D投影PSNR
  - [ ] `metrics/ssim_2d`: 2D投影SSIM

### 🔄 3. 性能优化 (如果baseline超标)

#### 3.1 损失权重调整
- **状态**: ⏳ 待优化
- **选项**:
  - 提高λ_geo (4.0 → 6.0) 进一步增强几何约束
  - 降低λ_ipsm (1.0 → 0.5) 减少SD影响

#### 3.2 伪视角采样策略
- **状态**: ⏳ 待优化
- **选项**:
  - 调整angle_range (15° → 10°或20°)
  - 尝试多伪视角（同时采样2-3个）

#### 3.3 深度估计器升级
- **状态**: ⏳ 待优化
- **选项**:
  - 替换DPT为Depth Anything v2
  - 或使用医学CT专用深度模型

### 🔄 4. 消融实验 (发Paper用)

- **状态**: ⏳ 待设计
- **实验设计**:
  - [ ] Baseline vs +IPSM(no depth) vs +IPSM(no geo) vs +Full IPSM
  - [ ] 分析各组件贡献

### 🔄 5. 多数据集验证

- **状态**: ⏳ 待执行
- **数据集**:
  - [ ] Chest-3
  - [ ] Head-3
  - [ ] Abdomen-3
  - [ ] Pancreas-3
- **目的**: 证明泛化性

---

## 🎯 预期结果

### Baseline (R²-Gaussian, Foot-3)
```
PSNR: 28.4873
SSIM: 0.9005
```

### 目标 (R²-Gaussian + IPSM)
```
PSNR: > 28.5 (+0.5% 保守估计)
SSIM: > 0.901 (+0.05%)
```

**说明**:
- 由于CT与自然图像的domain gap，提升可能小于IPSM在LLFF数据集上的表现（+7.2% SSIM）
- 如果baseline超标，说明IPSM的inline prior和几何约束有效发挥作用

---

## 📝 下一步行动

### 立即执行:
1. **运行快速验证** (500迭代)
   ```bash
   ./run_ipsm_验证.sh
   ```

2. **检查验证结果**
   - 确认无crash
   - 检查loss曲线
   - 验证扩散模型加载/卸载

3. **运行完整训练** (30K迭代)
   ```bash
   ./run_ipsm_完整训练.sh
   ```

4. **评估结果**
   ```bash
   python test.py -m output/YYYY_MM_DD_HH_MM_foot_3views_ipsm
   ```

### 后续优化 (如果baseline超标):
- 调整损失权重
- 优化伪视角采样策略
- 升级深度估计器
- 设计消融实验
- 多数据集验证

---

## 📚 相关文档

- `cc-agent/ipsm/IPSM集成实现指南.md` - 详细实施文档
- `cc-agent/ipsm/IPSM集成完成报告.md` - 完成报告
- `cc-agent/ipsm/ipsm.md` - IPSM论文分析

---

## 🔍 已知问题和解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **CUDA OOM** | 扩散模型占用大量显存 | 已使用FP16推理+动态加载/卸载 |
| **DPT加载失败** | 网络问题或依赖缺失 | 会自动降级为占位符模式（返回零深度） |
| **SD加载失败** | 网络问题或HuggingFace限制 | 修改`--sd_model_path`为本地路径 |
| **Depth渲染错误** | rasterizer不支持colors_precomp | 已使用Z坐标作为颜色渲染深度 |
| **IPSM loss为NaN** | 数值不稳定 | 检查depth是否有效，降低learning rate |
| **提升不明显** | CT domain gap | 降低λ_ipsm，提高λ_geo |

---

## 📅 时间线

- **2025-11-20**: IPSM核心模块实现完成
- **2025-11-20**: 训练脚本创建完成
- **2025-11-20**: 文档编写完成
- **待定**: 快速验证执行
- **待定**: 完整训练执行
- **待定**: 结果评估

---

**文档版本**: v1.0
**最后更新**: 2025-11-20
**维护者**: 进度跟踪与协调秘书

