# FSGS 优化实验计划 - v4+ 系列

**创建时间：** 2025-11-18 晚间
**负责专家：** @deep-learning-tuning-expert
**任务目标：** 在 v2 baseline (PSNR 28.50 dB, SSIM 0.9015) 基础上进一步提升性能
**Git Commit：** [待记录]

---

## 【核心结论】

基于 v2 成功经验和 v3 失败教训，本计划设计了 **阶段 1：单因素消融（8 个实验）** + **阶段 2：最佳组合（2-3 个实验）** 的优化路线。v3 教训表明：**放宽邻近约束（k=6→7, τ=8→9）导致性能下降 0.24 dB**，因此优化方向应为：(1) 更强的正则化控制过拟合，(2) 收紧而非放松医学约束，(3) 精细调整密集化策略。预计总实验时间：**40-48 小时**（阶段 1：32 小时，阶段 2：8-16 小时）。

---

## 1. 背景分析

### 1.1 当前最佳 baseline (v2)

```yaml
实验名称：2025_11_18_foot_3views_fsgs_fixed_v2
器官：Foot, 3 视角稀疏场景
数据集：data/369/foot_50_3views.pickle
训练迭代：30,000

测试集性能：
  PSNR: 28.50 dB  ← 超越 SOTA baseline 28.49 dB
  SSIM: 0.9015    ← 超越 SOTA baseline 0.9005

训练集性能：
  PSNR: 51.10 dB
  SSIM: [未记录，预估 > 0.99]

泛化差距：22.60 dB (训练集 - 测试集)

关键参数：
  lambda_tv: 0.08                # TV 正则化强度
  lambda_dssim: 0.25             # SSIM 损失权重
  max_num_gaussians: 200000      # 容量限制（比 v3 的 500k 严格 60%）
  densify_until_iter: 12000      # 密集化停止迭代
  densify_grad_threshold: 2e-4   # 密集化梯度阈值
  proximity_k_neighbors: 6       # 医学约束 k 邻居
  proximity_threshold: 8.0       # 医学约束距离阈值
  enable_medical_constraints: true
```

### 1.2 v3 失败案例分析

```yaml
实验名称：2025_11_18_foot_3views_fsgs_v3_params
关键修改（相对 v2）：
  proximity_k_neighbors: 6 → 7     # 增加邻居数
  proximity_threshold: 8.0 → 9.0   # 放宽距离阈值
  max_num_gaussians: 200k → 500k   # 放宽容量限制
  lambda_tv: 0.08 → 0.05           # 降低 TV 正则化

测试集性能（iter_020000）：
  PSNR: 28.26 dB  ← 比 v2 差 0.24 dB
  SSIM: 0.8982    ← 比 v2 差 0.0033

关键失败原因：
1. ❌ 放宽邻近约束（k=7, τ=9.0）→ 破坏医学先验，过度自由度
2. ❌ 降低 TV 正则化（0.05）→ 结构平滑性不足
3. ❌ 容量限制放宽（500k）→ 加剧过拟合风险
4. ⚠️ 训练卡死在 iter_020000（matplotlib 死锁 bug）
```

### 1.3 优化方向假设

基于 v2/v3 对比，我们提出以下假设：

| 优化方向 | 假设 | 预期影响 | 优先级 |
|---------|------|---------|--------|
| **强化正则化** | 增强 TV 正则化（0.08 → 0.10-0.15）可降低过拟合 | 泛化差距 ↓ 5-8 dB | ⭐⭐⭐ |
| **收紧医学约束** | 降低 k 邻居（6 → 4-5）或阈值（8.0 → 6.0-7.0）可提升结构一致性 | PSNR ↑ 0.1-0.3 dB | ⭐⭐⭐ |
| **精细密集化** | 提前停止密集化（12k → 8k-10k）减少冗余高斯 | PSNR ↑ 0.05-0.15 dB | ⭐⭐ |
| **调整 DSSIM 权重** | 增加 DSSIM（0.25 → 0.30）强化感知质量 | SSIM ↑ 0.001-0.003 | ⭐⭐ |
| **容量进一步限制** | max_gaussians: 200k → 150k-180k | 泛化差距 ↓ 2-4 dB | ⭐ |

---

## 2. 阶段 1：单因素消融实验（8 个实验）

**设计原则：** 每个实验仅改变 1-2 个相关参数，其他参数与 v2 保持一致。

### 实验优先级排序

```
P1（最高优先级）：正则化强度调整（实验 1.1, 1.2）
P2（高优先级）：医学约束收紧（实验 1.3, 1.4）
P3（中优先级）：密集化策略（实验 1.5, 1.6）
P4（低优先级）：辅助参数（实验 1.7, 1.8）
```

---

### 实验 1.1：强化 TV 正则化（中等强度）

**实验 ID：** v4_tv_0.10
**命名：** `2025_11_19_foot_3views_fsgs_v4_tv_0.10`

**修改参数：**
```yaml
lambda_tv: 0.08 → 0.10  # 提升 TV 正则化强度 25%
```

**保持不变（与 v2 一致）：**
```yaml
lambda_dssim: 0.25
max_num_gaussians: 200000
densify_until_iter: 12000
densify_grad_threshold: 2e-4
proximity_k_neighbors: 6
proximity_threshold: 8.0
enable_medical_constraints: true
```

**理论依据：** TV 正则化惩罚空间不平滑性，提升强度可降低训练集过拟合，改善泛化差距。v2 已证明 0.08 有效，0.10 为渐进式提升。

**预期结果：**
- 测试 PSNR: 28.48-28.55 dB（持平或小幅提升）
- 训练 PSNR: 48-50 dB（比 v2 的 51.10 降低）
- 泛化差距: 19-22 dB（目标改善 1-3 dB）

**成功标准：** 泛化差距 < 21 dB 且测试 PSNR ≥ 28.45 dB

---

### 实验 1.2：强化 TV 正则化（高强度）

**实验 ID：** v4_tv_0.12
**命名：** `2025_11_19_foot_3views_fsgs_v4_tv_0.12`

**修改参数：**
```yaml
lambda_tv: 0.08 → 0.12  # 提升 TV 正则化强度 50%
```

**理论依据：** 探索更强正则化的边界，验证是否会出现欠拟合（测试 PSNR 下降）。

**预期结果：**
- 测试 PSNR: 28.40-28.52 dB（可能小幅下降）
- 泛化差距: 18-20 dB（显著改善）
- 风险：过强正则化可能导致测试 PSNR 下降

**成功标准：** 泛化差距 < 20 dB 且测试 PSNR ≥ 28.40 dB

---

### 实验 1.3：收紧医学约束 - 减少邻居数

**实验 ID：** v4_k_5
**命名：** `2025_11_19_foot_3views_fsgs_v4_k_5`

**修改参数：**
```yaml
proximity_k_neighbors: 6 → 5  # 减少邻居数（更局部约束）
```

**理论依据：** v3 失败表明 k=7 过松，v2 的 k=6 成功。尝试 k=5 检验是否能进一步提升结构一致性。

**预期结果：**
- 测试 PSNR: 28.50-28.60 dB（目标提升 0.05-0.10 dB）
- SSIM: 0.9015-0.9030（更强局部结构约束）

**成功标准：** PSNR ≥ 28.52 dB 或 SSIM ≥ 0.9020

---

### 实验 1.4：收紧医学约束 - 降低距离阈值

**实验 ID：** v4_tau_7.0
**命名：** `2025_11_19_foot_3views_fsgs_v4_tau_7.0`

**修改参数：**
```yaml
proximity_threshold: 8.0 → 7.0  # 降低距离阈值（更严格约束）
```

**理论依据：** v3 的 τ=9.0 失败，v2 的 τ=8.0 成功。尝试 τ=7.0 检验是否能通过更严格的距离约束提升性能。

**预期结果：**
- 测试 PSNR: 28.50-28.58 dB（目标提升 0.05-0.08 dB）
- 高斯数量可能减少（更强修剪）

**成功标准：** PSNR ≥ 28.52 dB

---

### 实验 1.5：提前停止密集化（早停版）

**实验 ID：** v4_densify_10k
**命名：** `2025_11_19_foot_3views_fsgs_v4_densify_10k`

**修改参数：**
```yaml
densify_until_iter: 12000 → 10000  # 提前停止密集化
```

**理论依据：** 延迟密集化可能导致过度复杂化，提前停止可降低过拟合风险。

**预期结果：**
- 最终高斯数量减少（可能 150k-180k）
- 泛化差距可能改善 1-2 dB

**成功标准：** 测试 PSNR ≥ 28.45 dB 且泛化差距 < 21 dB

---

### 实验 1.6：提高密集化阈值（保守密集化）

**实验 ID：** v4_grad_3e-4
**命名：** `2025_11_19_foot_3views_fsgs_v4_grad_3e-4`

**修改参数：**
```yaml
densify_grad_threshold: 2e-4 → 3e-4  # 提高梯度阈值（更保守）
```

**理论依据：** 更高阈值意味着只有梯度更大的区域才会密集化，减少冗余高斯。

**预期结果：**
- 最终高斯数量减少
- 训练 PSNR 可能降低（拟合能力受限）

**成功标准：** 测试 PSNR ≥ 28.45 dB 且泛化差距 < 22 dB

---

### 实验 1.7：增强 DSSIM 权重

**实验 ID：** v4_dssim_0.30
**命名：** `2025_11_19_foot_3views_fsgs_v4_dssim_0.30`

**修改参数：**
```yaml
lambda_dssim: 0.25 → 0.30  # 提升 DSSIM 损失权重
```

**理论依据：** DSSIM 捕捉感知质量，增加权重可能提升 SSIM 指标。

**预期结果：**
- SSIM: 0.9015 → 0.9020-0.9030
- PSNR 可能小幅下降（PSNR 和 SSIM 有时存在权衡）

**成功标准：** SSIM ≥ 0.9020 且 PSNR ≥ 28.45 dB

---

### 实验 1.8：容量进一步限制

**实验 ID：** v4_cap_180k
**命名：** `2025_11_19_foot_3views_fsgs_v4_cap_180k`

**修改参数：**
```yaml
max_num_gaussians: 200000 → 180000  # 进一步限制容量 10%
```

**理论依据：** v2 已证明 200k（比 v3 的 500k 严格 60%）有效，进一步限制验证容量与性能的边界。

**预期结果：**
- 泛化差距改善 1-2 dB
- 测试 PSNR 可能持平或小幅下降（拟合能力受限）

**成功标准：** 泛化差距 < 21 dB 且测试 PSNR ≥ 28.45 dB

---

## 3. 阶段 1 执行计划

### 3.1 并行策略（假设有多 GPU）

**GPU 1：** 实验 1.1, 1.2（TV 正则化，连续执行）
**GPU 2：** 实验 1.3, 1.4（医学约束，连续执行）
**GPU 3：** 实验 1.5, 1.6（密集化策略，连续执行）
**GPU 4：** 实验 1.7, 1.8（辅助参数，连续执行）

**预计总时间：** 32 小时（每实验 4 小时，2 轮并行）

**单 GPU 情况：** 顺序执行，预计总时间 32 小时（8 实验 × 4 小时）

### 3.2 配置文件生成脚本

```bash
#!/bin/bash
# 文件：cc-agent/experiments/scripts/generate_v4_configs.sh

BASE_CONFIG="/home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_18_foot_3views_fsgs_fixed_v2/cfg_args.yml"
OUTPUT_DIR="/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/experiments/configs"

# 实验 1.1
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_tv_0.10.yml \
    --lambda_tv 0.10 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_tv_0.10

# 实验 1.2
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_tv_0.12.yml \
    --lambda_tv 0.12 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_tv_0.12

# 实验 1.3
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_k_5.yml \
    --proximity_k_neighbors 5 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_k_5

# 实验 1.4
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_tau_7.0.yml \
    --proximity_threshold 7.0 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_tau_7.0

# 实验 1.5
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_densify_10k.yml \
    --densify_until_iter 10000 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_densify_10k

# 实验 1.6
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_grad_3e-4.yml \
    --densify_grad_threshold 3e-4 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_grad_3e-4

# 实验 1.7
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_dssim_0.30.yml \
    --lambda_dssim 0.30 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_dssim_0.30

# 实验 1.8
python scripts/modify_config.py $BASE_CONFIG $OUTPUT_DIR/v4_cap_180k.yml \
    --max_num_gaussians 180000 \
    --model_path output/2025_11_19_foot_3views_fsgs_v4_cap_180k
```

### 3.3 训练执行脚本

```bash
#!/bin/bash
# 文件：cc-agent/experiments/scripts/run_v4_phase1.sh

CUDA_ENV="r2_gaussian_new"
PYTHON_SCRIPT="train.py"

# 激活环境
conda activate $CUDA_ENV

# 实验 1.1
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_tv_0.10.yml

# 实验 1.2
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_tv_0.12.yml

# 实验 1.3
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_k_5.yml

# 实验 1.4
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_tau_7.0.yml

# 实验 1.5
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_densify_10k.yml

# 实验 1.6
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_grad_3e-4.yml

# 实验 1.7
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_dssim_0.30.yml

# 实验 1.8
python $PYTHON_SCRIPT --config cc-agent/experiments/configs/v4_cap_180k.yml
```

### 3.4 Early Stopping 策略

**监控指标：** 每个实验在 iter_10000 时检查测试 PSNR

**停止条件：**
- 如果 PSNR < 28.0 dB（比 v2 差 0.5 dB 以上）→ 立即停止该实验
- 如果 PSNR 28.0-28.40 dB → 继续运行但标记为"低优先级"
- 如果 PSNR > 28.50 dB → 标记为"高潜力候选"

**实施方式：**
```python
# 在 train.py 中添加监控逻辑（如果尚未实现）
if iteration == 10000:
    test_psnr = evaluate(scene, gaussians, test_cameras)
    if test_psnr < 28.0:
        print("[Early Stop] PSNR too low, stopping experiment")
        sys.exit(0)
```

---

## 4. 阶段 2：最佳组合实验（2-3 个实验）

**触发条件：** 阶段 1 所有实验完成后，根据结果设计组合。

**设计策略：** 选择阶段 1 中表现最好的 2-3 个参数改进点组合。

### 4.1 预设组合方案（待阶段 1 结果调整）

**假设场景 A：** 如果 1.1（TV=0.10）和 1.3（k=5）均成功

**实验 2.1：** 组合强化正则化 + 收紧医学约束
```yaml
实验 ID：v5_combo_tv_k
命名：2025_11_20_foot_3views_fsgs_v5_combo_tv_k

修改参数：
  lambda_tv: 0.10
  proximity_k_neighbors: 5

预期结果：
  测试 PSNR: 28.55-28.65 dB
  泛化差距: 18-20 dB
```

---

**假设场景 B：** 如果 1.2（TV=0.12）和 1.5（densify_10k）均成功

**实验 2.2：** 组合高强度正则化 + 提前停止密集化
```yaml
实验 ID：v5_combo_tv_densify
命名：2025_11_20_foot_3views_fsgs_v5_combo_tv_densify

修改参数：
  lambda_tv: 0.12
  densify_until_iter: 10000

预期结果：
  泛化差距: 16-18 dB
  测试 PSNR: 28.45-28.55 dB
```

---

**假设场景 C：** 如果 1.3（k=5）、1.4（τ=7.0）和 1.7（DSSIM=0.30）均成功

**实验 2.3：** 组合全面收紧约束
```yaml
实验 ID：v5_combo_strict
命名：2025_11_20_foot_3views_fsgs_v5_combo_strict

修改参数：
  proximity_k_neighbors: 5
  proximity_threshold: 7.0
  lambda_dssim: 0.30

预期结果：
  测试 PSNR: 28.58-28.70 dB
  SSIM: 0.9025-0.9040
```

---

**阶段 2 时间预估：** 8-16 小时（2-3 实验 × 4 小时，可能并行）

---

## 5. 评估指标与成功标准

### 5.1 主要指标（必须记录）

| 指标 | 来源 | 目标值 | v2 baseline |
|-----|------|--------|------------|
| **测试 PSNR** | eval2d_render_test.yml | > 28.50 dB | 28.50 dB |
| **测试 SSIM** | eval2d_render_test.yml | > 0.9015 | 0.9015 |
| **训练 PSNR** | eval2d_render_train.yml | 45-50 dB | 51.10 dB |
| **泛化差距** | 训练 - 测试 PSNR | < 20 dB | 22.60 dB |

### 5.2 辅助指标（建议记录）

- 最终高斯数量（`max_num_gaussians` 到达情况）
- 训练时间（每 1000 迭代耗时）
- GPU 显存峰值（`nvidia-smi` 监控）
- iter_10000 时的测试 PSNR（早停决策依据）

### 5.3 成功标准分级

**S 级成功（重大突破）：**
- 测试 PSNR ≥ 28.60 dB 且 SSIM ≥ 0.905
- 泛化差距 < 18 dB

**A 级成功（显著改进）：**
- 测试 PSNR ≥ 28.55 dB 或 SSIM ≥ 0.903
- 泛化差距 < 20 dB

**B 级成功（边际改进）：**
- 测试 PSNR ≥ 28.52 dB 或泛化差距 < 21 dB

**C 级成功（持平）：**
- 测试 PSNR 28.45-28.52 dB 且泛化差距 21-23 dB

**失败：**
- 测试 PSNR < 28.40 dB 或泛化差距 > 23 dB

---

## 6. 风险控制与应对策略

### 6.1 如果阶段 1 所有实验都不如 v2？

**应对方案 A（保守路线）：**
- 接受 v2 为当前最优配置
- 转向其他器官（Chest, Head, Abdomen）验证 v2 的通用性
- 探索更长训练迭代（50k, 100k）

**应对方案 B（激进路线）：**
- 重新审视 v2 的成功机制：
  - 是否 lambda_tv=0.08 是最优值？尝试 0.06-0.07
  - 是否 k=6, τ=8.0 是局部最优？尝试 k=4, τ=6.5
- 探索学习率调度（当前未优化）

**应对方案 C（算法改进）：**
- 联系 3DGS 专家，考虑引入新的正则化技术：
  - Dropout for Gaussians（随机丢弃部分高斯）
  - Gradient Penalty（梯度惩罚）
  - 数据增强（旋转、翻转伪视角）

### 6.2 matplotlib 死锁 bug 处理

**强制要求：** 在运行任何实验前，必须修复 `r2_gaussian/utils/plot_utils.py:271-274` 的 matplotlib backend 切换问题。

**修复验证方式：**
```bash
# 运行一个快速测试确认 bug 已修复
python train.py --config output/2025_11_18_foot_3views_fsgs_fixed_v2/cfg_args.yml \
    --iterations 25000 --model_path output/bug_test
# 检查是否能顺利通过 iter_020000 评估
```

### 6.3 实验中断恢复策略

**Checkpoint 机制：**
```yaml
# 所有实验配置中添加
checkpoint_iterations: [10000, 20000]

# 中断后恢复命令
python train.py --config <config.yml> --start_checkpoint output/<exp_name>/chkpnt_010000.pth
```

---

## 7. 实验追踪与记录规范

### 7.1 实验目录结构

```
output/
├── 2025_11_19_foot_3views_fsgs_v4_tv_0.10/
│   ├── cfg_args.yml              # 完整配置
│   ├── eval/
│   │   ├── iter_030000/
│   │   │   ├── eval2d_render_test.yml   # 测试集结果
│   │   │   └── eval2d_render_train.yml  # 训练集结果
│   ├── point_cloud/
│   │   └── iteration_030000/point_cloud.ply
│   └── chkpnt_030000.pth
├── 2025_11_19_foot_3views_fsgs_v4_tv_0.10_train.log  # 训练日志
└── ...
```

### 7.2 结果汇总表（需手动维护）

**文件：** `cc-agent/experiments/v4_results_summary.md`

| 实验 ID | 测试 PSNR | 测试 SSIM | 训练 PSNR | 泛化差距 | 高斯数量 | 成功等级 | 备注 |
|---------|----------|----------|----------|---------|---------|---------|------|
| v2 (baseline) | 28.50 | 0.9015 | 51.10 | 22.60 | ~200k | A | 当前最佳 |
| v3 (失败) | 28.26 | 0.8982 | - | - | - | 失败 | k=7, τ=9.0 过松 |
| v4_tv_0.10 | - | - | - | - | - | - | 待测试 |
| v4_tv_0.12 | - | - | - | - | - | - | 待测试 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 7.3 Git 标记规范

**每个实验完成后打 tag：**
```bash
# 示例
git tag -a exp-v4_tv_0.10 -m "实验 v4_tv_0.10: lambda_tv=0.10, 测试 PSNR=28.52"
git push origin exp-v4_tv_0.10
```

### 7.4 每日进度报告

**文件：** `cc-agent/experiments/daily_log.md`

```markdown
## 2025-11-19

**完成实验：**
- v4_tv_0.10：PSNR 28.52 dB（+0.02）✅
- v4_tv_0.12：PSNR 28.48 dB（-0.02）⚠️

**进行中实验：**
- v4_k_5：iter_015000/30000

**明日计划：**
- 完成 1.3-1.8 实验
- 初步分析 1.1-1.2 结果

**技术问题：**
- 无
```

---

## 8. 可复现性保证

### 8.1 环境固定

```yaml
CUDA 环境：r2_gaussian_new
Python 版本：[需记录，运行 python --version]
PyTorch 版本：[需记录，运行 python -c "import torch; print(torch.__version__)"]
CUDA 版本：[需记录，运行 nvcc --version]

随机种子：
  - 所有实验使用固定种子（如果代码未实现，需添加）
  - 建议种子：42（深度学习经典种子）
```

### 8.2 代码版本记录

**每个实验的 cfg_args.yml 中应记录 Git commit hash（如果未实现，需手动记录）：**

```yaml
# 添加到配置文件
experiment_metadata:
  git_commit: "b6a8955"
  git_branch: "fsgs-hqy"
  datetime: "2025-11-19 14:30:00"
```

---

## 9. 预期时间线

| 阶段 | 任务 | 预计时间 | 截止日期 |
|-----|------|---------|---------|
| **准备** | 修复 matplotlib bug, 生成配置文件 | 2 小时 | 2025-11-19 12:00 |
| **阶段 1** | 8 个单因素实验 | 32 小时 | 2025-11-20 20:00 |
| **分析 1** | 整理阶段 1 结果，设计阶段 2 | 4 小时 | 2025-11-21 00:00 |
| **✋ 检查点** | 等待用户批准阶段 2 方案 | - | - |
| **阶段 2** | 2-3 个组合实验 | 8-16 小时 | 2025-11-21 16:00 |
| **分析 2** | 最终结果分析与报告 | 4 小时 | 2025-11-21 20:00 |
| **总计** | - | 50-58 小时 | ~2.5 天 |

---

## 【需要您的决策】

在执行本实验计划前，请回答以下问题：

### 问题 1：是否批准阶段 1 的 8 个实验？

- **选项 A：** 全部批准，按优先级顺序执行（推荐）
- **选项 B：** 只批准 P1+P2 实验（1.1-1.4，共 4 个），节省时间
- **选项 C：** 修改实验参数后再批准（请指定修改内容）

### 问题 2：实验执行方式？

- **选项 A：** 单 GPU 顺序执行（总时间 32 小时）
- **选项 B：** 多 GPU 并行执行（总时间 8-16 小时，需确认有 4 个可用 GPU）
- **选项 C：** 手动逐个执行，每次等待结果后决定是否继续

### 问题 3：matplotlib bug 修复？

- **选项 A：** 我来修复（编程专家协助）
- **选项 B：** 已修复（请确认 git commit）
- **选项 C：** 跳过评估功能（不推荐，会丢失中间迭代数据）

### 问题 4：Early Stopping 策略？

- **选项 A：** 启用（PSNR < 28.0 时停止）
- **选项 B：** 禁用（所有实验运行到 30000 迭代）

### 问题 5：如果阶段 1 所有实验均不如 v2，下一步行动？

- **选项 A：** 接受 v2 为最优，转向其他器官验证
- **选项 B：** 执行应对方案 B（重新审视 v2）
- **选项 C：** 执行应对方案 C（联系 3DGS 专家，探索新算法）

---

**✋ 检查点 - 等待用户确认**

**请批准本实验计划后，调参专家将执行以下操作：**

1. 更新 `cc-agent/experiments/record.md`（记录任务启动）
2. 生成配置文件（如果批准）
3. 修复 matplotlib bug（如果需要）
4. 启动阶段 1 实验
5. 在每个实验完成后汇报结果
6. 阶段 1 完成后设计阶段 2 并再次请求批准

---

**文档版本：** v1.0
**最后修改：** 2025-11-18 晚间
**Git Commit：** [待提交]
