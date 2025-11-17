# CoR-GS Stage 2 - Co-Pruning 实现方案

## 核心策略 (3-5 句总结)

采用渐进式集成策略，在现有 Stage 1 Disagreement Metrics 基础上添加 KNN 双向剪枝机制。核心修改集中在 `train.py` 训练循环和新建 `copruning.py` 工具模块，通过 `--enable_copruning` 开关实现向下兼容。实施路线分 3 天：Day 1 核心算法实现 + 单元测试，Day 2 训练集成 + 日志监控，Day 3 实验验证 + 参数调优。预期在 Foot 3-views 数据集上达到 PSNR ≥28.5 dB (目标超越 baseline 28.547 dB)。

---

## 📁 文件修改详细方案

### 修改 1: 新建核心模块 `r2_gaussian/utils/copruning.py`

**文件路径**: `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/copruning.py`

**功能**: Co-pruning 双向剪枝算法实现

**完整代码** (150 行):

```python
"""
CoR-GS Stage 2: Co-Pruning Module

Paper Reference:
- CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
- Section 4.1: Co-pruning
- Formula 1-2: KNN matching and non-matching mask computation

Author: @3dgs-research-expert
Date: 2025-11-17
Version: 1.0
"""

import torch
import torch.nn as nn
from pytorch3d.ops import knn_points


class CoPruningModule:
    """
    Co-Pruning 机制实现类

    核心算法:
    1. KNN 双向匹配: f(θ_i^1) = KNN(θ_i^1, Θ^2)
    2. 距离判断: M_i = 1 if ||θ_i^1 - f(θ_i^1)|| > τ
    3. 双向剪枝: 同时移除两侧的非匹配点
    """

    def __init__(self, tau=5.0, device='cuda'):
        """
        初始化 Co-Pruning 模块

        Args:
            tau (float): 距离阈值，默认 5.0 (针对 [-1,1]³ 归一化场景)
            device (str): 计算设备
        """
        self.tau = tau
        self.device = device

        # 统计信息
        self.stats = {
            'total_prunings': 0,
            'total_points_removed_1': 0,
            'total_points_removed_2': 0,
            'avg_removal_rate_1': 0.0,
            'avg_removal_rate_2': 0.0
        }

    def __call__(self, gaussian_model_1, gaussian_model_2):
        """
        执行 Co-Pruning 剪枝

        Args:
            gaussian_model_1: GaussianModel instance (模型 1)
            gaussian_model_2: GaussianModel instance (模型 2)

        Returns:
            tuple: (pruned_model_1, pruned_model_2, pruning_info)
        """
        return self.co_prune(gaussian_model_1, gaussian_model_2)

    def co_prune(self, gaussian_model_1, gaussian_model_2):
        """
        Co-Pruning 核心算法

        实现步骤:
        1. 提取两个模型的 3D 位置
        2. KNN 搜索找到最近邻
        3. 计算欧几里得距离
        4. 生成匹配掩码 (dist <= tau)
        5. 双向剪枝

        Returns:
            tuple: (pruned_model_1, pruned_model_2, info_dict)
        """

        # Step 1: 提取 3D 位置
        xyz_1 = gaussian_model_1.get_xyz  # [N1, 3]
        xyz_2 = gaussian_model_2.get_xyz  # [N2, 3]

        num_before_1 = xyz_1.shape[0]
        num_before_2 = xyz_2.shape[0]

        # Step 2: KNN 搜索 (使用 PyTorch3D 优化实现)
        # knn_points 返回: (dists, idx, nn)
        # dists: 最近邻距离的平方 [batch, N, K]
        knn_result_1to2 = knn_points(
            xyz_1.unsqueeze(0).to(self.device),  # [1, N1, 3]
            xyz_2.unsqueeze(0).to(self.device),  # [1, N2, 3]
            K=1,  # 只找最近邻
            return_nn=False
        )

        knn_result_2to1 = knn_points(
            xyz_2.unsqueeze(0).to(self.device),
            xyz_1.unsqueeze(0).to(self.device),
            K=1,
            return_nn=False
        )

        # Step 3: 计算欧几里得距离 (knn_points 返回平方距离)
        dist_1 = torch.sqrt(knn_result_1to2.dists.squeeze(0).squeeze(-1))  # [N1]
        dist_2 = torch.sqrt(knn_result_2to1.dists.squeeze(0).squeeze(-1))  # [N2]

        # Step 4: 生成匹配掩码 (保留 dist <= tau 的点)
        # 论文公式: M_i = 0 if dist <= tau, else 1
        # 这里反转: mask_keep = True 表示保留
        mask_keep_1 = dist_1 <= self.tau
        mask_keep_2 = dist_2 <= self.tau

        # Step 5: 剪枝操作
        gaussian_model_1.prune_points(mask_keep_1)
        gaussian_model_2.prune_points(mask_keep_2)

        num_after_1 = gaussian_model_1.get_xyz.shape[0]
        num_after_2 = gaussian_model_2.get_xyz.shape[0]

        # Step 6: 统计信息
        num_removed_1 = num_before_1 - num_after_1
        num_removed_2 = num_before_2 - num_after_2

        removal_rate_1 = num_removed_1 / num_before_1 * 100 if num_before_1 > 0 else 0
        removal_rate_2 = num_removed_2 / num_before_2 * 100 if num_before_2 > 0 else 0

        # 更新全局统计
        self.stats['total_prunings'] += 1
        self.stats['total_points_removed_1'] += num_removed_1
        self.stats['total_points_removed_2'] += num_removed_2
        self.stats['avg_removal_rate_1'] = (
            self.stats['avg_removal_rate_1'] * (self.stats['total_prunings'] - 1) +
            removal_rate_1
        ) / self.stats['total_prunings']
        self.stats['avg_removal_rate_2'] = (
            self.stats['avg_removal_rate_2'] * (self.stats['total_prunings'] - 1) +
            removal_rate_2
        ) / self.stats['total_prunings']

        # 返回详细信息
        info = {
            'num_before_1': num_before_1,
            'num_after_1': num_after_1,
            'num_removed_1': num_removed_1,
            'removal_rate_1': removal_rate_1,

            'num_before_2': num_before_2,
            'num_after_2': num_after_2,
            'num_removed_2': num_removed_2,
            'removal_rate_2': removal_rate_2,

            'mean_dist_1': dist_1.mean().item(),
            'mean_dist_2': dist_2.mean().item(),
            'max_dist_1': dist_1.max().item(),
            'max_dist_2': dist_2.max().item()
        }

        return gaussian_model_1, gaussian_model_2, info

    def get_stats(self):
        """返回累计统计信息"""
        return self.stats

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_prunings': 0,
            'total_points_removed_1': 0,
            'total_points_removed_2': 0,
            'avg_removal_rate_1': 0.0,
            'avg_removal_rate_2': 0.0
        }


# ========== 辅助函数 ==========

def compute_point_rmse(gaussian_model_1, gaussian_model_2, device='cuda'):
    """
    计算两个模型的点云 RMSE (用于 Disagreement Metrics)

    Args:
        gaussian_model_1: GaussianModel instance
        gaussian_model_2: GaussianModel instance
        device: 计算设备

    Returns:
        dict: {'rmse': float, 'fitness': float, 'mean_dist': float}
    """
    xyz_1 = gaussian_model_1.get_xyz
    xyz_2 = gaussian_model_2.get_xyz

    # KNN 搜索
    knn_result = knn_points(
        xyz_1.unsqueeze(0).to(device),
        xyz_2.unsqueeze(0).to(device),
        K=1,
        return_nn=False
    )

    dists = torch.sqrt(knn_result.dists.squeeze(0).squeeze(-1))

    # RMSE: 均方根误差
    rmse = torch.sqrt(torch.mean(dists ** 2)).item()

    # Fitness: tau=5.0 内的匹配点比例
    tau = 5.0
    fitness = (dists <= tau).float().mean().item()

    # Mean distance: 平均距离
    mean_dist = dists.mean().item()

    return {
        'rmse': rmse,
        'fitness': fitness,
        'mean_dist': mean_dist,
        'max_dist': dists.max().item(),
        'min_dist': dists.min().item()
    }
```

---

### 修改 2: 训练脚本集成 `train.py`

**文件路径**: `/home/qyhu/Documents/r2_ours/r2_gaussian/train.py`

**修改位置 1: 导入模块 (Line ~15)**

```python
# 在现有 imports 后添加
from r2_gaussian.utils.copruning import CoPruningModule
```

---

**修改位置 2: 命令行参数 (Line ~350, ArgumentParser section)**

```python
# CoR-GS Stage 2: Co-Pruning Parameters
parser.add_argument('--enable_copruning', action='store_true',
                    help='Enable CoR-GS Stage 2 Co-Pruning mechanism')
parser.add_argument('--copruning_interval', type=int, default=5,
                    help='Execute co-pruning every N densification steps (default: 5)')
parser.add_argument('--copruning_tau', type=float, default=5.0,
                    help='Distance threshold for co-pruning (default: 5.0 for normalized scenes)')
```

---

**修改位置 3: 初始化 Co-Pruning 模块 (Line ~200, training() 函数开头)**

```python
def training(dataset, opt, pipe, gaussians, scene, testing_iterations, saving_iterations):
    ...

    # ===== 新增: 初始化 Co-Pruning 模块 =====
    copruning_module = None
    densification_step_counter = 0

    if len(gaussians) == 2 and opt.enable_copruning:
        copruning_module = CoPruningModule(
            tau=opt.copruning_tau,
            device=dataset.device
        )
        print(f"[CoR-GS Stage 2] Co-Pruning enabled with tau={opt.copruning_tau}, "
              f"interval={opt.copruning_interval}")
    elif opt.enable_copruning:
        print("[Warning] Co-Pruning requires --gaussiansN 2, but got",
              len(gaussians), "models. Co-Pruning disabled.")
    # =========================================

    first_iter = 0
    ...
```

---

**修改位置 4: Densification 循环中触发 Co-Pruning (Line ~300)**

**原始代码:**
```python
# 现有 densification 逻辑 (大约在 Line 300-320)
if iteration >= opt.densify_from_iter and iteration <= opt.densify_until_iter:
    if iteration % opt.densification_interval == 0:
        for gaussian in gaussians:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussian.densify_and_prune(
                opt.densify_grad_threshold,
                0.005,
                scene.cameras_extent,
                size_threshold
            )
```

**修改后:**
```python
if iteration >= opt.densify_from_iter and iteration <= opt.densify_until_iter:
    if iteration % opt.densification_interval == 0:
        densification_step_counter += 1  # ✨ 新增计数器

        # 原始 densification 逻辑
        for gaussian in gaussians:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussian.densify_and_prune(
                opt.densify_grad_threshold,
                0.005,
                scene.cameras_extent,
                size_threshold
            )

        # ===== 新增: Co-Pruning 触发逻辑 =====
        if copruning_module is not None:
            if densification_step_counter % opt.copruning_interval == 0:
                print(f"\n[Iteration {iteration}] Executing Co-Pruning...")

                # 执行 Co-Pruning
                gaussians[0], gaussians[1], pruning_info = copruning_module(
                    gaussians[0],
                    gaussians[1]
                )

                # 详细日志输出
                print(f"  Model 1: {pruning_info['num_before_1']} → "
                      f"{pruning_info['num_after_1']} pts "
                      f"({pruning_info['num_removed_1']} removed, "
                      f"{pruning_info['removal_rate_1']:.2f}%)")
                print(f"  Model 2: {pruning_info['num_before_2']} → "
                      f"{pruning_info['num_after_2']} pts "
                      f"({pruning_info['num_removed_2']} removed, "
                      f"{pruning_info['removal_rate_2']:.2f}%)")
                print(f"  Mean Distance: Model1={pruning_info['mean_dist_1']:.6f}, "
                      f"Model2={pruning_info['mean_dist_2']:.6f}")

                # (可选) TensorBoard 记录
                if tb_writer:
                    tb_writer.add_scalar(
                        'copruning/num_points_model1',
                        pruning_info['num_after_1'],
                        iteration
                    )
                    tb_writer.add_scalar(
                        'copruning/num_points_model2',
                        pruning_info['num_after_2'],
                        iteration
                    )
                    tb_writer.add_scalar(
                        'copruning/removal_rate_model1',
                        pruning_info['removal_rate_1'],
                        iteration
                    )
        # =======================================
```

---

### 修改 3: 验证 GaussianModel 支持 (可选)

**文件**: `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/gaussian/gaussian_model.py`

**检查是否存在 `prune_points()` 方法:**

如果不存在，需要添加该方法 (参考 3DGS 官方实现):

```python
def prune_points(self, mask):
    """
    根据布尔掩码剪除点

    Args:
        mask: [N] boolean tensor
              True = 保留该点
              False = 移除该点
    """
    valid_points_mask = mask

    # 使用现有的 _prune_optimizer 方法
    optimizable_tensors = self._prune_optimizer(valid_points_mask)

    # 更新所有属性
    self._xyz = optimizable_tensors["xyz"]
    self._features_dc = optimizable_tensors["f_dc"]
    self._features_rest = optimizable_tensors["f_rest"]
    self._opacity = optimizable_tensors["opacity"]
    self._scaling = optimizable_tensors["scaling"]
    self._rotation = optimizable_tensors["rotation"]

    # 更新辅助变量
    self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
    self.denom = self.denom[valid_points_mask]
    self.max_radii2D = self.max_radii2D[valid_points_mask]
```

**验证命令:**
```bash
grep -n "def prune_points" r2_gaussian/gaussian/gaussian_model.py
```

如果输出为空，需要添加该方法。

---

## 🔧 配置参数与使用指南

### 训练命令示例

**Baseline (不启用 Co-Pruning):**
```bash
python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_17_foot_3views_baseline \
    --iterations 10000 \
    --test_iterations 1000 5000 10000 \
    --save_iterations 10000 \
    --eval \
    --gaussiansN 2
```

**启用 Co-Pruning (默认参数):**
```bash
python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_17_foot_3views_copruning \
    --iterations 10000 \
    --test_iterations 1000 5000 10000 \
    --save_iterations 10000 \
    --eval \
    --gaussiansN 2 \
    --enable_copruning \
    --copruning_interval 5 \
    --copruning_tau 5.0
```

**启用 Co-Pruning (更严格阈值):**
```bash
python train.py \
    --source_path data/369/foot_50_3views.pickle \
    --model_path output/2025_11_17_foot_3views_copruning_tau3 \
    --iterations 10000 \
    --enable_copruning \
    --copruning_tau 3.0  # 更严格的剪枝
    ...
```

---

### 参数说明与调优建议

| 参数 | 默认值 | 范围 | 说明 | 何时调整 |
|------|-------|------|------|---------|
| `--enable_copruning` | False | - | 启用 Co-Pruning | 总是需要明确指定 |
| `--copruning_interval` | 5 | 3-10 | 每 N 次 densify 执行 1 次 | 论文默认，一般不改 |
| `--copruning_tau` | 5.0 | 3.0-10.0 | KNN 距离阈值 | 根据场景尺度调整 |

**调参策略:**

1. **首次实验**: 使用论文默认值 (interval=5, tau=5.0)
2. **如果剪除过多** (>30%): 增大 tau 到 7.0 或 10.0
3. **如果剪除过少** (<5%): 减小 tau 到 3.0 或 4.0
4. **如果训练时间过长**: 增大 interval 到 7 或 10

---

## ✅ 验证检查清单

### 代码级别

- [ ] `r2_gaussian/utils/copruning.py` 文件创建完成
- [ ] `train.py` 导入 `CoPruningModule` 成功
- [ ] 命令行参数 `--enable_copruning` 添加
- [ ] Co-Pruning 逻辑正确集成到 densification 循环
- [ ] `GaussianModel.prune_points()` 方法存在或已添加

### 功能级别

- [ ] 运行 `python train.py --help` 能看到 Co-Pruning 参数
- [ ] 启动训练后日志输出 "Co-Pruning enabled" 提示
- [ ] 在 densify 时正确输出 Co-Pruning 执行信息
- [ ] 剪除比例在合理范围 (5-20%)
- [ ] 不启用 `--enable_copruning` 时 baseline 行为不变

### 性能级别

- [ ] PSNR ≥ 28.5 dB (持平 baseline 28.547 dB)
- [ ] SSIM 保持或提升 (目标 ≥ 0.90)
- [ ] 训练时间增加 <10% (Co-Pruning 开销可忽略)
- [ ] 最终 Gaussian 点数合理 (预期减少 10-20%)

---

## ⚠️ 潜在问题与调试方案

### 问题 1: 剪除比例过高 (>30%)

**症状**: 每次 Co-Pruning 移除 >30% 点

**原因**:
- tau 阈值过严格
- 场景归一化尺度问题

**调试步骤**:
1. 检查场景归一化范围 (应为 [-1,1]³)
2. 增大 tau 到 7.0 或 10.0
3. 打印 mean_dist 和 max_dist，分析距离分布

---

### 问题 2: 剪除比例过低 (<2%)

**症状**: 几乎不剪除任何点

**原因**:
- tau 阈值过宽松
- 当前双模型一致性已经很好 (Fitness=1.0)

**调试步骤**:
1. 确认当前 RMSE 值 (应为 0.011-0.012 mm)
2. 尝试减小 tau 到 3.0 或 4.0
3. 如果仍剪除很少，说明 Co-Pruning 空间确实有限

---

### 问题 3: 训练崩溃或 NaN Loss

**症状**: Co-Pruning 后出现 NaN loss

**原因**:
- 剪除过多导致点数不足
- Optimizer state 未正确更新

**调试步骤**:
1. 检查 `prune_points()` 方法是否正确更新 `xyz_gradient_accum` 等辅助变量
2. 增大 tau 减少剪除比例
3. 检查剪除后点数是否 >1000 (过少会导致训练失败)

---

### 问题 4: Co-Pruning 未执行

**症状**: 日志中看不到 "Executing Co-Pruning" 输出

**原因**:
- `--enable_copruning` 未指定
- `--gaussiansN` 不等于 2
- `densification_step_counter` 未正确递增

**调试步骤**:
1. 确认命令行参数正确
2. 在 Co-Pruning 逻辑前添加调试打印:
   ```python
   print(f"[Debug] iteration={iteration}, "
         f"densification_step_counter={densification_step_counter}, "
         f"copruning_module={copruning_module is not None}")
   ```

---

## 📊 预期结果分析框架

### 实验对比表格模板

| 配置 | PSNR (dB) | SSIM | 点数 (Model 1) | 点数 (Model 2) | 训练时间 |
|------|----------|------|---------------|---------------|---------|
| R² Baseline | 28.547 | 0.9008 | ~200k | ~200k | 15 min |
| Stage 1 Only | 28.148 | 0.8383 | ~200k | ~200k | 16 min |
| **Stage 1 + Stage 2** | **?** | **?** | **?** | **?** | **?** |

**填写指南:**
1. 训练完成后提取 `iter_010000/results.json`
2. 从日志提取最终点数 (最后一次 Co-Pruning 后)
3. 从日志提取总训练时间

---

### Co-Pruning 统计分析

**监控指标:**
```python
# 在训练结束后打印累计统计
stats = copruning_module.get_stats()
print("\n===== Co-Pruning Summary =====")
print(f"Total prunings:       {stats['total_prunings']}")
print(f"Total removed Model1: {stats['total_points_removed_1']}")
print(f"Total removed Model2: {stats['total_points_removed_2']}")
print(f"Avg removal rate M1:  {stats['avg_removal_rate_1']:.2f}%")
print(f"Avg removal rate M2:  {stats['avg_removal_rate_2']:.2f}%")
```

**预期正常范围:**
- Total prunings: 10-15 次 (基于 densify_until_iter=7000, interval=5)
- Avg removal rate: 5-20%
- 如果 <5%: Co-Pruning 空间有限
- 如果 >30%: 可能过度剪枝

---

## 🎯 成功标准与决策树

### Level 1: 代码实现成功

**标准:**
- [ ] Co-Pruning 正确执行 (日志输出正常)
- [ ] 剪除比例在合理范围 (5-30%)
- [ ] 训练完成无崩溃

**达成 → 进入 Level 2**
**未达成 → 调试代码实现**

---

### Level 2: 性能持平 Baseline

**标准:**
- [ ] PSNR ≥ 28.5 dB (vs Baseline 28.547 dB)
- [ ] SSIM ≥ 0.90

**达成 → Level 2 Success → 考虑实施 Stage 3**
**未达成 → 进入参数调优或更换策略**

---

### Level 3: 性能超越 Baseline

**标准:**
- [ ] PSNR ≥ 28.8 dB (超越 +0.25 dB)
- [ ] SSIM ≥ 0.905

**达成 → Level 3 Success → 扩展到其他数据集**
**未达成但 Level 2 达成 → 仍然成功**

---

## 📅 实施时间表

### Day 1: 核心算法实现 (2025-11-17)

**上午 (3 小时)**
- [ ] 创建 `r2_gaussian/utils/copruning.py` 文件
- [ ] 实现 `CoPruningModule` 类
- [ ] 实现辅助函数 `compute_point_rmse()`
- [ ] 代码 review + 语法检查

**下午 (2 小时)**
- [ ] 编写单元测试 (测试 KNN、剪枝逻辑)
- [ ] 验证 `GaussianModel.prune_points()` 方法
- [ ] 如不存在，添加该方法

---

### Day 2: 训练集成 (2025-11-18)

**上午 (3 小时)**
- [ ] 修改 `train.py` 添加 imports 和参数
- [ ] 集成 Co-Pruning 逻辑到 densification 循环
- [ ] 添加日志输出和 TensorBoard 记录
- [ ] 代码 review + 集成测试

**下午 (2 小时)**
- [ ] 启动首次训练 (foot 3-views, 10k iterations)
- [ ] 实时监控日志输出
- [ ] 验证 Co-Pruning 执行频率和剪除比例

---

### Day 3: 实验验证 (2025-11-19)

**上午 (2 小时)**
- [ ] 等待训练完成 (如果 Day 2 未完成)
- [ ] 提取 `iter_010000/results.json`
- [ ] 对比 baseline 和 Stage 1 结果

**下午 (3 小时)**
- [ ] 如果未达标，调整参数 (tau, interval)
- [ ] 启动第二轮实验验证
- [ ] 生成实验报告文档

---

## 🤔 您的决策点

### 决策点 1: 是否立即实施？

**如果您批准:**
- 我将立即开始 Day 1 工作 (创建 `copruning.py`)
- 预计 3 天完成全部实施和验证

**如果您暂缓:**
- 建议先完成其他方法 (如 FSGS 修复验证)
- 或等待单模型实验结果再���定

---

### 决策点 2: 参数配置选择

**选项 A: 论文默认 (推荐)**
```
--copruning_interval 5
--copruning_tau 5.0
```

**选项 B: 保守配置**
```
--copruning_interval 7
--copruning_tau 7.0
```

**选项 C: 激进配置**
```
--copruning_interval 3
--copruning_tau 3.0
```

**推荐选择 A**，再根据结果调整

---

### 决策点 3: 验证策略

**选项 A: 快速验证 (1 个实验)**
- 使用论文默认参数
- 仅在 foot 3-views 验证
- 时间: 3 天

**选项 B: 完整验证 (3 个实验)**
- 默认参数 + 两组消融 (tau=3, tau=10)
- 在 foot 3-views 验证
- 时间: 5 天

**推荐选择 A** (先快速验证是否有效)

---

## 📚 参考资料

**核心论文:**
- CoR-GS Section 4.1 (Co-pruning 算法)
- Supplementary Material Table I (超参数敏感性)
- Table 6 (消融研究)

**代码实现参考:**
- PyTorch3D knn_points API: https://pytorch3d.readthedocs.io/en/latest/modules/ops.html#pytorch3d.ops.knn_points
- 3DGS Official Implementation: https://github.com/graphdeco-inria/gaussian-splatting

**相关文档:**
- `cc-agent/3dgs_expert/innovation_analysis_corgs_stage2.md` (技术分析)
- `cc-agent/3dgs_expert/corgs_innovation_analysis.md` (Stage 1 分析)

---

**文档生成时间**: 2025-11-17 15:45
**版本**: v1.0
**字数**: 2487 字
**负责专家**: @3dgs-research-expert
**审核状态**: 待用户批准
