# GR-Gaussian 实验验证方案

**生成时间:** 2025-11-17
**版本:** v1.0
**负责专家:** 深度学习调参与分析专家

---

## 【核心结论】

GR-Gaussian 当前**仅实现了 Graph Laplacian 正则化**,缺少核心组件 De-Init 和 PGA。基于现有实现,本方案设计 **4 个消融实验** 验证 Graph Laplacian 单独效果,并为后续完整实现奠定基础。实验数据集为 foot 3 views,训练 30000 轮,目标超越 baseline (PSNR 28.55 dB)。

---

## 1. 实验背景

### 1.1 Baseline 性能

| 指标 | R²-Gaussian Baseline |
|-----|---------------------|
| PSNR | 28.547 dB |
| SSIM | 0.9008 |
| 训练轮数 | 30000 iterations |
| 数据集 | data/369 (foot, 3 views) |
| 输出目录 | output/foot_3views_r2_baseline_1113/ |

### 1.2 GR-Gaussian 技术分析

**论文完整技术栈:**
1. **De-Init (去噪点云初始化)** - ❌ **未实现**
   - 使用 Gaussian 滤波降噪 FDK volume
   - 参数: σ_d=3.0, τ=0.001

2. **PGA (Pixel-Graph-Aware Gradient)** - ❌ **未实现**
   - 基于密度差异的梯度增强
   - 公式: g_aug = g_pixel + λ_g * (Σ|ρ_i - ρ_j| / k)

3. **Graph Laplacian 正则化** - ✅ **已实现**
   - 位置: `r2_gaussian/utils/loss_utils.py::compute_graph_laplacian_loss`
   - 参数: k=6, λ_lap=8e-4
   - 状态: 已集成到 train.py,每 500 iterations 计算一次

### 1.3 实现状态评估

**当前可用功能:**
- ✅ Graph Laplacian 正则化 (KNN 图构建 + 平滑损失)
- ✅ GPU 加速 (torch.cdist + topk)
- ✅ 自动 Fallback 到 CPU (sklearn KDTree)

**缺失关键组件:**
- ❌ De-Init 降噪初始化 (预期 +0.4~0.6 dB)
- ❌ PGA 梯度增强 (预期 +0.2~0.4 dB)

**预期性能影响:**
- 仅 Graph Laplacian: **+0.1~0.3 dB** (保守估计)
- 完整 GR-Gaussian: **+0.6~0.9 dB** (论文数据)

---

## 2. 实验设计

### 2.1 核心消融实验 (4 个配置)

| 实验名称 | De-Init | PGA | Graph Laplacian | λ_lap | 预期 PSNR | 说明 |
|---------|---------|-----|-----------------|-------|-----------|------|
| **实验 1: Baseline** | ❌ | ❌ | ❌ | 0 | 28.55 dB | 重现基线 |
| **实验 2: GL-Base** | ❌ | ❌ | ✅ | 8e-4 | 28.6~28.7 dB | Graph Laplacian 单独验证 |
| **实验 3: GL-Strong** | ❌ | ❌ | ✅ | 2e-3 | 28.65~28.75 dB | 强正则化版本 |
| **实验 4: GL-Adaptive** | ❌ | ❌ | ✅ | 动态调整 | 28.7~28.8 dB | 自适应权重 |

**实验 4 说明 (GL-Adaptive):**
- 前 5000 iter: λ_lap = 2e-3 (强正则化,加速收敛)
- 5000~15000 iter: λ_lap = 8e-4 (标准,平衡重建与平滑)
- 15000~30000 iter: λ_lap = 2e-4 (弱正则化,保留细节)

### 2.2 实验参数配置

**通用训练参数 (与 baseline 保持一致):**
```bash
--iterations 30000
--test_iterations 1000 5000 10000 15000 20000 25000 30000
--save_iterations 30000
--densify_grad_threshold 0.0002
--densify_until_iter 15000
--eval
```

**Graph Laplacian 专用参数:**
```bash
# 实验 2: GL-Base
--enable_graph_laplacian
--graph_k 6
--graph_lambda_lap 8e-4

# 实验 3: GL-Strong
--enable_graph_laplacian
--graph_k 6
--graph_lambda_lap 2e-3

# 实验 4: GL-Adaptive (需代码微调,暂不实施)
--enable_graph_laplacian
--graph_k 6
--graph_lambda_lap_schedule "2e-3,8e-4,2e-4"
--graph_lambda_lap_milestones "5000,15000"
```

### 2.3 数据集配置

**数据源:**
- 训练数据: `data/369/0_foot_cone_3views.pickle`
- 初始化: `data/369/init_0_foot_cone_3views.npy`
- 视角数: 3 views (稀疏场景)

**评估指标:**
- PSNR (dB) - 主要指标
- SSIM - 结构相似性
- 训练时间 (min) - 效率指标
- Gaussian 点数量 - 复杂度指标

---

## 3. 训练脚本

### 3.1 实验 1: Baseline 重现

**目的:** 验证当前环境能否重现 baseline 性能

```bash
#!/bin/bash
# 脚本位置: scripts/run_gr_gaussian_exp1_baseline.sh

conda activate r2_gaussian_new

python train.py \
    --source_path data/369 \
    --model_path output/2025_11_17_foot_3views_baseline_rerun \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    2>&1 | tee logs/exp1_baseline_$(date +%Y%m%d_%H%M).log

echo "Baseline 实验完成,查看结果:"
echo "  输出目录: output/2025_11_17_foot_3views_baseline_rerun/"
echo "  评估文件: output/2025_11_17_foot_3views_baseline_rerun/results.json"
```

**预期结果:**
- PSNR: 28.5~28.6 dB (±0.05 dB 误差可接受)
- SSIM: 0.900~0.902
- 训练时间: 25~35 分钟 (RTX 4090)

---

### 3.2 实验 2: GL-Base (标准 Graph Laplacian)

**目的:** 验证 Graph Laplacian 单独效果

```bash
#!/bin/bash
# 脚本位置: scripts/run_gr_gaussian_exp2_gl_base.sh

conda activate r2_gaussian_new

python train.py \
    --source_path data/369 \
    --model_path output/2025_11_17_foot_3views_gl_base \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    --enable_graph_laplacian \
    --graph_k 6 \
    --graph_lambda_lap 8e-4 \
    2>&1 | tee logs/exp2_gl_base_$(date +%Y%m%d_%H%M).log

echo "GL-Base 实验完成,查看结果:"
echo "  输出目录: output/2025_11_17_foot_3views_gl_base/"
echo "  评估文件: output/2025_11_17_foot_3views_gl_base/results.json"
```

**预期结果:**
- PSNR: 28.6~28.7 dB (+0.1~0.15 dB)
- SSIM: 0.902~0.905 (+0.002~0.003)
- Graph Loss: 稳定下降,最终 <1e-5
- 训练时间: +5~10% (KNN 图构建开销)

**成功判断标准:**
- ✅ PSNR ≥ 28.6 dB
- ✅ Graph Loss 收敛 (<1e-5)
- ✅ 无训练崩溃/NaN

---

### 3.3 实验 3: GL-Strong (强正则化)

**目的:** 测试更强的平滑约束能否进一步提升

```bash
#!/bin/bash
# 脚本位置: scripts/run_gr_gaussian_exp3_gl_strong.sh

conda activate r2_gaussian_new

python train.py \
    --source_path data/369 \
    --model_path output/2025_11_17_foot_3views_gl_strong \
    --iterations 30000 \
    --test_iterations 1000 5000 10000 15000 20000 25000 30000 \
    --save_iterations 30000 \
    --eval \
    --densify_grad_threshold 0.0002 \
    --densify_until_iter 15000 \
    --enable_graph_laplacian \
    --graph_k 6 \
    --graph_lambda_lap 2e-3 \
    2>&1 | tee logs/exp3_gl_strong_$(date +%Y%m%d_%H%M).log

echo "GL-Strong 实验完成,查看结果:"
echo "  输出目录: output/2025_11_17_foot_3views_gl_strong/"
echo "  评估文件: output/2025_11_17_foot_3views_gl_strong/results.json"
```

**预期结果:**
- PSNR: 28.65~28.75 dB (可能更高)
- SSIM: 0.903~0.906
- 风险: 过度平滑导致细节丢失

**成功判断标准:**
- ✅ PSNR > 实验 2 结果
- ⚠️ 需目视检查切片,确保无过度模糊

---

### 3.4 实验 4: GL-Adaptive (暂缓)

**说明:** 需要修改代码实现动态权重调整,工作量较大 (~2 天)。建议先完成实验 1-3,根据结果决定是否实施。

**如需实施,修改点:**
- `train.py`: 添加 lambda schedule 逻辑
- `arguments/__init__.py`: 添加 schedule 参数

---

## 4. 执行计划

### 4.1 完整实验流程

**时间线 (基于 RTX 4090 单卡):**

```
Day 1 (2025-11-17):
  09:00~09:30  准备训练脚本,创建日志目录
  09:30~10:00  实验 1: Baseline 启动 (30 分钟训练)
  10:00~10:30  验证 baseline 结果,确认环境正常
  10:30~11:05  实验 2: GL-Base 启动 (35 分钟训练)
  11:05~11:30  分析 GL-Base 结果,记录指标
  11:30~12:05  实验 3: GL-Strong 启动 (35 分钟训练)

Day 1 下午:
  14:00~14:30  分析实验 3 结果
  14:30~16:00  生成实验报告,绘制对比图表
  16:00~17:00  决策: 是否实施实验 4 或完整 GR-Gaussian

总计时间: 1 天 (纯训练约 2 小时)
```

### 4.2 并行执行策略 (可选)

**如果有多张 GPU:**
```bash
# 终端 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gr_gaussian_exp1_baseline.sh &

# 终端 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 bash scripts/run_gr_gaussian_exp2_gl_base.sh &

# 终端 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 bash scripts/run_gr_gaussian_exp3_gl_strong.sh &
```

**优点:** 3 个实验同步完成,总时间 ~40 分钟
**缺点:** 需要 3 张 GPU

---

## 5. 评估指标

### 5.1 定量指标

**主要指标:**
- **PSNR (dB)** - 必须超越 baseline (28.55 dB)
- **SSIM** - 结构相似性,目标 ≥ 0.900
- **训练时间 (min)** - 监控 Graph Laplacian 开销
- **Gaussian 点数量** - 检查是否过度密化

**辅助指标:**
- Graph Laplacian Loss - 收敛趋势
- L1 Loss - 重建误差
- SSIM Loss - 结构保持
- TV Loss - 3D 平滑度

### 5.2 定性评估

**可视化检查 (从 TensorBoard):**
1. Loss 曲线 - 平滑下降无震荡
2. PSNR 趋势 - 持续上升至 plateau
3. 中间切片渲染 - 边缘清晰度,伪影抑制
4. Gaussian 密度分布 - 无异常聚集

**关键切片位置 (foot 数据集):**
- Slice 40: 踝关节区域 (高对比度)
- Slice 64: 足中部 (中等密度)
- Slice 90: 足趾区域 (细节丰富)

---

## 6. 成功判断标准

### 6.1 最低成功标准 (必须满足)

- ✅ **实验 2 (GL-Base) PSNR ≥ 28.6 dB** (+0.05 dB 超越 baseline)
- ✅ **训练稳定,无 NaN/Inf**
- ✅ **Graph Loss 收敛 (<1e-5)**
- ✅ **训练时间增加 <15%**

### 6.2 理想成功标准

- ⭐ **实验 3 (GL-Strong) PSNR ≥ 28.7 dB** (+0.15 dB)
- ⭐ **SSIM ≥ 0.905** (+0.003)
- ⭐ **目视切片质量明显改善** (边缘更清晰)

### 6.3 失败标准 (需重新设计)

- ❌ 所有实验 PSNR < 28.55 dB (未超越 baseline)
- ❌ 训练中途崩溃或 Loss 发散
- ❌ Graph Loss 不收敛 (>1e-3)
- ❌ 过度平滑导致细节丢失 (SSIM 下降)

---

## 7. 结果分析模板

**实验完成后生成报告: `cc-agent/experiments/result_analysis_gr_gaussian.md`**

### 7.1 定量对比表

| 实验配置 | PSNR (dB) | SSIM | 训练时间 (min) | Gaussian 数量 | Graph Loss (final) |
|---------|-----------|------|---------------|--------------|-------------------|
| Baseline (重现) | 28.XX | 0.90X | XX | XXk | N/A |
| GL-Base (λ=8e-4) | 28.XX | 0.90X | XX | XXk | X.XXe-X |
| GL-Strong (λ=2e-3) | 28.XX | 0.90X | XX | XXk | X.XXe-X |

### 7.2 分析要点

**收敛分析:**
- Loss 曲线对比 (3 个实验 overlay)
- PSNR 提升速度 (iter 1000/5000/10000 对比)
- Graph Loss 收敛点 (哪个 iteration 降至 1e-5)

**性能瓶颈诊断:**
- 如果 GL-Base 未超越 baseline → 检查 KNN 图构建是否正确
- 如果 GL-Strong 过度平滑 → 降低 λ_lap 或减少 k
- 如果训练时间过长 → 考虑每 1000 iter 计算 Graph Loss (而非 500)

**统计显著性:**
- 运行 3 次重复实验 (不同 random seed)
- 计算均值和标准差
- t-test 检验 PSNR 提升是否显著 (p<0.05)

---

## 8. 后续优化方向

### 8.1 短期 (基于实验结果)

**如果 GL-Base 成功 (PSNR ≥ 28.6 dB):**
1. 超参数微调: k ∈ {4,5,6,7,8}, λ_lap ∈ {4e-4, 8e-4, 1.2e-3}
2. 实施实验 4 (GL-Adaptive) 验证动态权重
3. 扩展到其他器官 (chest, head 3 views)

**如果 GL-Base 未达标 (PSNR < 28.6 dB):**
1. 检查 Graph Laplacian 实现是否正确
2. 对比 CoR-GS Disagreement Loss 是否更有效
3. 考虑实施 De-Init 降噪初始化 (预期 +0.4 dB)

### 8.2 中期 (1-2 周)

**完整 GR-Gaussian 实现:**
1. **De-Init 去噪初始化** (预期 +0.4~0.6 dB)
   - 修改 `r2_gaussian/gaussian/initialize.py`
   - 使用 `scipy.ndimage.gaussian_filter`
   - 工期: 2 天

2. **PGA 梯度增强** (预期 +0.2~0.4 dB)
   - 新建 `r2_gaussian/utils/graph_utils.py`
   - 修改 `r2_gaussian/gaussian/gaussian_model.py` 的 densification 逻辑
   - 工期: 3 天

3. **完整消融实验 (7 个配置)**
   - Baseline, De-Init, GL, PGA, De-Init+GL, De-Init+PGA, Full GR-Gaussian
   - 工期: 5 天

### 8.3 长期 (论文撰写)

**如果完整 GR-Gaussian 达标 (PSNR ≥ 29.1 dB):**
- 扩展到 X-3D 数据集全部类别
- 对比 SAX-NeRF, NAF 等 SOTA 方法
- 撰写论文章节: "GR-Gaussian for Few-Shot CT Reconstruction"

---

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|-----|------|------|---------|
| **Baseline 无法重现** | 低 | 高 | 检查数据集路径,确认初始化文件正确 |
| **Graph Laplacian 无效果** | 中 | 中 | 实施 De-Init,或尝试 CoR-GS Disagreement |
| **过度平滑** | 中 | 中 | 降低 λ_lap,或使用 adaptive schedule |
| **训练时间过长** | 低 | 低 | 减少 Graph Loss 计算频率 (1000 iter) |
| **GPU 内存不足** | 低 | 低 | 自动 Fallback 到 CPU KNN (已实现) |

---

## 10. 文件清单

### 10.1 新建文件

**训练脚本:**
- `/home/qyhu/Documents/r2_ours/r2_gaussian/scripts/run_gr_gaussian_exp1_baseline.sh`
- `/home/qyhu/Documents/r2_ours/r2_gaussian/scripts/run_gr_gaussian_exp2_gl_base.sh`
- `/home/qyhu/Documents/r2_ours/r2_gaussian/scripts/run_gr_gaussian_exp3_gl_strong.sh`

**日志目录:**
- `/home/qyhu/Documents/r2_ours/r2_gaussian/logs/` (需手动创建)

**输出目录 (训练时自动创建):**
- `output/2025_11_17_foot_3views_baseline_rerun/`
- `output/2025_11_17_foot_3views_gl_base/`
- `output/2025_11_17_foot_3views_gl_strong/`

**实验报告 (待生成):**
- `cc-agent/experiments/result_analysis_gr_gaussian.md`

### 10.2 修改文件

**需检查的现有文件:**
- `r2_gaussian/utils/loss_utils.py` - 确认 `compute_graph_laplacian_loss` 正常
- `train.py` - 确认 `--enable_graph_laplacian` 参数存在
- `r2_gaussian/arguments/__init__.py` - 确认参数定义

---

## 11. 立即行动清单

### 11.1 准备工作 (10 分钟)

```bash
# 1. 创建日志和脚本目录
mkdir -p /home/qyhu/Documents/r2_ours/r2_gaussian/logs
mkdir -p /home/qyhu/Documents/r2_ours/r2_gaussian/scripts

# 2. 确认数据集存在
ls -lh /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/0_foot_cone_3views.pickle
ls -lh /home/qyhu/Documents/r2_ours/r2_gaussian/data/369/init_0_foot_cone_3views.npy

# 3. 检查 Graph Laplacian 实现
grep -A 20 "def compute_graph_laplacian_loss" \
    /home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/loss_utils.py

# 4. 检查 train.py 参数
grep "enable_graph_laplacian\|graph_lambda_lap" \
    /home/qyhu/Documents/r2_ours/r2_gaussian/train.py
```

### 11.2 生成训练脚本 (5 分钟)

使用本文档第 3 节的脚本内容创建 3 个 `.sh` 文件,并赋予执行权限:
```bash
chmod +x scripts/run_gr_gaussian_exp*.sh
```

### 11.3 启动实验 (立即执行)

```bash
# 按顺序执行 (单卡)
bash scripts/run_gr_gaussian_exp1_baseline.sh
# 等待完成后
bash scripts/run_gr_gaussian_exp2_gl_base.sh
# 等待完成后
bash scripts/run_gr_gaussian_exp3_gl_strong.sh
```

---

## 12. 预期交付物

**实验完成后提供:**
1. **定量对比表** (3 个实验 PSNR/SSIM)
2. **Loss 曲线图** (TensorBoard 导出)
3. **切片可视化** (3 个关键位置,before/after)
4. **结果分析报告** (`result_analysis_gr_gaussian.md`)
5. **Git commit** (实验配置和结果记录)

**预计完成时间:** 2025-11-17 下午 17:00 (8 小时内)

---

## 【需要您的决策】

### 选项 A: 立即执行实验 1-3 (推荐)
- ✅ 验证 Graph Laplacian 单独效果
- ✅ 快速获得结果 (1 天内)
- ✅ 风险低,所需组件已实现

### 选项 B: 先完整实现 GR-Gaussian 再实验
- ⏰ 需要 5-7 天实现 De-Init + PGA
- ⭐ 一次性验证完整技术栈
- ⚠️ 风险较高,实现复杂度大

### 选项 C: 并行策略
- 由编程专家实现 De-Init (2 天)
- 同时由调参专家执行实验 1-3
- 完成后再执行完整实验

**建议:** 选择 **选项 A**,先用 1 天验证现有实现,根据结果决定是否继续完整实现。

---

**✋ 等待用户确认:**
1. 是否批准实验方案 (选项 A/B/C)?
2. 是否需要修改实验参数 (如 iterations, λ_lap)?
3. 是否需要并行执行 (如有多张 GPU)?
