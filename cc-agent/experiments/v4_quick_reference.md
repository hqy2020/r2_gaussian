# FSGS v4+ 实验快速参考

**创建时间：** 2025-11-18
**状态：** 等待用户批准 ✋

---

## 核心目标

**提升指标：**
- 测试 PSNR：28.50 → 28.60+ dB
- 测试 SSIM：0.9015 → 0.905+
- 泛化差距：22.60 → <20 dB

---

## 阶段 1：8 个单因素实验（32 小时）

| 实验 ID | 参数修改 | 优先级 | 预期影响 |
|---------|---------|--------|---------|
| v4_tv_0.10 | lambda_tv: 0.08→0.10 | ⭐⭐⭐ | 泛化差距 ↓ 1-3 dB |
| v4_tv_0.12 | lambda_tv: 0.08→0.12 | ⭐⭐⭐ | 泛化差距 ↓ 3-5 dB, PSNR 可能小幅降 |
| v4_k_5 | k_neighbors: 6→5 | ⭐⭐⭐ | PSNR ↑ 0.05-0.10 dB |
| v4_tau_7.0 | threshold: 8.0→7.0 | ⭐⭐⭐ | PSNR ↑ 0.05-0.08 dB |
| v4_densify_10k | densify_until: 12k→10k | ⭐⭐ | 泛化差距 ↓ 1-2 dB |
| v4_grad_3e-4 | grad_threshold: 2e-4→3e-4 | ⭐⭐ | 减少高斯数量 |
| v4_dssim_0.30 | lambda_dssim: 0.25→0.30 | ⭐⭐ | SSIM ↑ 0.001-0.003 |
| v4_cap_180k | max_gaussians: 200k→180k | ⭐ | 泛化差距 ↓ 1-2 dB |

---

## 关键决策点（需用户回答）

### Q1：批准阶段 1 实验？
- **A：** 全部批准（8 个实验，32 小时）← 推荐
- **B：** 只批准 P1+P2（4 个实验，16 小时）
- **C：** 修改后批准（请说明）

### Q2：执行方式？
- **A：** 单 GPU 顺序（32 小时）
- **B：** 多 GPU 并行（8-16 小时）← 如有 4 GPU 推荐
- **C：** 手动逐个（每次等结果）

### Q3：matplotlib bug？
- **A：** 我来修复（需编程专家）
- **B：** 已修复（请确认 commit）
- **C：** 跳过评估（不推荐）

### Q4：Early Stopping？
- **A：** 启用（PSNR < 28.0 时停止）← 推荐
- **B：** 禁用（全部跑完 30k）

### Q5：若阶段 1 全失败？
- **A：** 接受 v2 为最优，测试其他器官
- **B：** 重新审视 v2（尝试 0.06-0.07 TV 等）
- **C：** 联系 3DGS 专家，探索新算法

---

## 快速执行命令（批准后）

```bash
# 1. 修复 bug（如需要）
# 见 plot_utils.py:271-274，将 matplotlib.use("Agg") 移到文件顶部

# 2. 生成配置（假设工具脚本已准备）
bash cc-agent/experiments/scripts/generate_v4_configs.sh

# 3. 执行实验（单 GPU 示例）
conda activate r2_gaussian_new
python train.py --config cc-agent/experiments/configs/v4_tv_0.10.yml
python train.py --config cc-agent/experiments/configs/v4_tv_0.12.yml
# ... 依次执行其他 6 个

# 4. 汇总结果
python cc-agent/experiments/scripts/summarize_results.py
```

---

## 成功标准

| 等级 | 测试 PSNR | SSIM | 泛化差距 |
|-----|----------|------|---------|
| **S** | ≥28.60 | ≥0.905 | <18 dB |
| **A** | ≥28.55 | ≥0.903 | <20 dB |
| **B** | ≥28.52 | - | <21 dB |
| **C** | 28.45-28.52 | - | 21-23 dB |
| 失败 | <28.40 | - | >23 dB |

---

## v2 vs v3 对比（教训）

| 参数 | v2 (成功) | v3 (失败) | 结论 |
|-----|----------|----------|------|
| lambda_tv | 0.08 | 0.05 | ↓正则化 → 性能下降 |
| k_neighbors | 6 | 7 | 增加邻居 → 性能下降 |
| threshold | 8.0 | 9.0 | 放宽阈值 → 性能下降 |
| max_gaussians | 200k | 500k | 放宽容量 → 过拟合风险 |
| **测试 PSNR** | **28.50** | **28.26 (-0.24)** | v2 明显更优 |
| **测试 SSIM** | **0.9015** | **0.8982 (-0.0033)** | v2 明显更优 |

**关键教训：** 收紧约束（更强正则化 + 更严格医学约束）是正确方向，放宽约束导致失败。

---

## 详细文档
- 完整计划：`cc-agent/experiments/fsgs_optimization_plan_v4_plus.md`
- 任务记录：`cc-agent/experiments/record.md`
- 进度追踪：`cc-agent/records/progress.md`

---

**✋ 等待用户确认后开始执行**
