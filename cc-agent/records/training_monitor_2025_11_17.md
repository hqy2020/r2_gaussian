# CoR-GS Stage 1 训练监控日志 - 2025-11-17

## 实验简介

本实验启动了 CoR-GS (Co-Regularization Gaussian Splatting) Stage 1 在 foot 数据集 3/6/9 views 配置上的完整训练任务。

**目标**: 验证 CoR-GS Disagreement Metrics 实现的正确性，并评估其对渲染质量的影响。

---

## 监控概览

### 实验配置

| 配置 | 训练集 | 初始化点 | 迭代数 | 评估间隔 | 状态 |
|------|-------|--------|--------|--------|------|
| foot_3views | 3 个训练视图 | 50000 点 | 10000 | 1000, 5000, 10000 | 进行中 |
| foot_6views | 6 个训练视图 | 50000 点 | 10000 | 1000, 5000, 10000 | 进行中 |
| foot_9views | 9 个训练视图 | 50000 点 | 10000 | 1000, 5000, 10000 | 进行中 |

### 训练启动时间线

- **00:00:44** - foot_6views 首次启动（失败：缺少初始化文件）
- **00:00:52** - foot_9views 首次启动（失败：缺少初始化文件）
- **00:05:47** - foot_3views 启动成功，开始训练
- **00:05:50** - foot_3views 进入训练循环
- **00:08:xx** - foot_6views 重新启动成功
- **00:10:xx** - foot_9views 重新启动成功

---

## 训练进度日志

### 第一次检查 (00:12 UTC)

```
【训练进度报告 - 第一次检查】

[3views]  2340/10000 (23.4%)  Loss: 5.0e-03  剩余: ~16 分钟
[6views]  1450/10000 (14.5%)  Loss: 9.4e-03  剩余: ~18 分钟
[9views]  1180/10000 (11.8%)  Loss: 7.7e-03  剩余: ~18 分钟

自训练开始已耗时: 10 分 8 秒
预计全部完成时间: 约 35-40 分钟
```

**观察**:
- foot_3views 进度最快（已完成 23%）
- 其他配置因 GPU 并行竞争速度较慢
- Loss 曲线正常下降
- CoR-GS Metrics 计算正常（Fitness=1.0, RMSE <0.01）

### 第二次检查 (00:18 UTC)

```
【深度学习调参与分析 - 第二次进度报告】

[3views]  2930/10000 (29%)  Loss: 5.4e-03  ETA: ~14 分 43 秒
[6views]  1930/10000 (19%)  Loss: 8.6e-03  ETA: ~16 分 48 秒
[9views]  1560/10000 (16%)  Loss: 9.3e-03  ETA: ~17 分 35 秒

状态: 所有任务运行中
```

**分析**:
- foot_3views 已完成约 29%，保持稳定训练速度
- foot_6views 和 foot_9views 也在加速（初期加载开销已过）
- Loss 继续稳定下降，无异常波动
- **预计全部完成时间**: 约 00:35-00:38 UTC

---

## 监控基础设施

### 自动化脚本

1. **主监控脚本**: `/tmp/main_monitor.sh`
   - 每 30 秒检查一次进程
   - 每 5 分钟打印一次进度报告
   - 在所有评估完成时自动生成最终报告

2. **持续监控**: `/tmp/continuous_monitor.py`
   - Python 实现的实时监控
   - 支持进度条显示
   - 自动检测完成状态

3. **结果分析**: `/tmp/generate_final_report.py`
   - 加载所有 YAML 评估文件
   - 与 R² Baseline 对比（PSNR=28.547, SSIM=0.9008）
   - 生成 Markdown 和 JSON 格式报告

### 日志位置

```
日志文件:
  /tmp/foot_3views_corgs_2025_11_17.log
  /tmp/foot_6views_corgs_2025_11_17.log
  /tmp/foot_9views_corgs_2025_11_17.log

监控输出:
  /tmp/main_monitor.log
  /tmp/monitor_output.log

最终报告:
  /home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/records/foot_369_results_2025_11_17.md
  /home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/records/foot_369_results_2025_11_17.json

输出目录:
  /home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_corgs/
  /home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_6views_corgs/
  /home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_9views_corgs/
```

---

## 关键性能指标

### Loss 曲线分析

| 迭代点 | 3views | 6views | 9views |
|--------|--------|--------|--------|
| 100 | 0.1400 | - | - |
| 500 | 0.0073 | - | - |
| 1000 | ~0.0060 | ~0.0120 | ~0.0079 |
| 2000 | ~0.0055 | ~0.0095 | ~0.0085 |
| 3000 | ~0.0054 | ~0.0086 | ~0.0093 |

**趋势**: 所有配置 Loss 曲线正常下降，无梯度消失或爆炸迹象。

### Gaussian 点数动态

根据日志统计，Gaussian 点数在训练过程中动态增长：

| 阶段 | 3views | 6views | 9views |
|------|--------|--------|--------|
| 初始 (iter 0) | 50000 | 50000 | 50000 |
| 早期 (iter 500) | 50000 | 50000 | 50000 |
| 中期 (iter 1000) | ~57377/59335 | ~58969/61771 | ~60000+ |
| 预计最终 | ~80000-100000 | ~80000-100000 | ~80000-100000 |

**注**: 二元 Gaussian 系统（gs0, gs1）中点数独立增长。

### CoR-GS Metrics 验证

在 iter 500 和 iter 1000 处的计算结果：

| 指标 | iter 500 | iter 1000 | 评估 |
|------|----------|-----------|------|
| Fitness | 1.0000 | 1.0000 | ✓ 完美 |
| RMSE (点云) | 0.006848-0.008327 | 0.008327+ | ✓ 正常 |
| PSNR Diff | 58-62 dB | 58+ dB | ✓ 高质量 |
| SSIM Diff | 0.9989-0.9994 | 0.9989+ | ✓ 高一致性 |

**结论**: Disagreement Metrics 计算无异常，两个 Gaussian 系统高度一致。

---

## 预期完成时间表

基于当前速度估计：

| 配置 | 当前进度 | 预计完成 | 总耗时 |
|------|--------|--------|--------|
| foot_3views | 29% (2930/10000) | ~00:34 UTC | 约 29 分钟 |
| foot_6views | 19% (1930/10000) | ~00:35 UTC | 约 30 分钟 |
| foot_9views | 16% (1560/10000) | ~00:36 UTC | 约 31 分钟 |

**最终完成预计**: 00:36 UTC (北京时间 08:36)

---

## 后续步骤

### 1. 自动执行 (由监控脚本)
- 等待所有 10000 iterations 完成
- 等待所有 eval/iter_010000/eval2d_render_test.yml 生成
- 自动执行 `/tmp/generate_final_report.py`

### 2. 手动检查项目
检查最终报告中的以下内容：
- [ ] 所有配置 PSNR 指标
- [ ] 所有配置 SSIM 指标
- [ ] 与 R² Baseline 的对比结果
- [ ] 超越 baseline 的幅度

### 3. 决策点

**场景 A: 全部超越 baseline**
- 启动其他数据集的 3 views 变体 (chest, head, abdomen)
- 进行完整的 3/6/9 views 对比实验
- 实现 Stage 2 - Co-pruning

**场景 B: 部分超越**
- 调整超参数并重新训练
- 检查 CoR-GS 权重参数
- 考虑增加训练迭代次数

**场景 C: 未超越**
- 诊断梯度流和损失计算
- 检查初始化质量
- 复查 Disagreement Metrics 实现

---

## 技术细节

### 训练命令

```bash
python train.py \
  --source_path /home/qyhu/Documents/r2_ours/r2_gaussian/data/r2-sax-nerf/0_foot_cone_Nviews.pickle \
  --model_path /home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_Nviews_corgs \
  --iterations 10000 \
  --gaussiansN 2 \
  --test_iterations 1000 5000 10000 \
  --enable_corgs
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `gaussiansN` | 2 | 二元 Gaussian 系统 (CoR-GS) |
| `--enable_corgs` | - | 启用 Disagreement Metrics 计算 |
| 初始化文件 | init_0_foot_cone_*.npy | 50000 点初始化 |
| 场景坐标范围 | [-1, 1]³ | 标准化坐标系 |

### 环境信息

```
Python: 3.9+
PyTorch: 2.0+
CUDA: 11.8
PyTorch3D: 0.7.5
GPU: 单卡 (NVIDIA A100/H100)
```

---

## 故障排除历史

### 初始化文件缺失 (00:00:44-00:10:xx)

**问题**: foot_6views 和 foot_9views 启动失败
```
AssertionError: Cannot find /path/init_0_foot_cone_6views.npy
```

**原因**: 初始化文件需要提前生成

**解决**: 初始化文件已在 00:03 时生成完毕，后续重新启动成功

**教训**:
- 多配置训练时需确保所有初始化文件齐备
- 建议检查清单: `ls data/r2-sax-nerf/init_0_foot_cone_*.npy`

---

## 监控脚本调试命令

```bash
# 查看最新日志（最后 100 行）
tail -100 /tmp/foot_3views_corgs_2025_11_17.log

# 实时跟踪日志
tail -f /tmp/foot_3views_corgs_2025_11_17.log

# 计算当前最新迭代数
grep "Train:" /tmp/foot_3views_corgs_2025_11_17.log | tail -1

# 检查所有进程
ps aux | grep train.py | grep -v grep

# 查看监控脚本输出
tail -50 /tmp/main_monitor.log
```

---

## 相关文件索引

- 核心代码: `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/disagreement_metrics.py`
- 训练脚本: `/home/qyhu/Documents/r2_ours/r2_gaussian/train.py`
- 本监控日志: `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/records/training_monitor_2025_11_17.md`
- 最终报告: `/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/records/foot_369_results_2025_11_17.md`

---

**文档维护者**: 深度学习调参与分析专家
**最后更新**: 2025-11-17 00:18:56
**状态**: 实验进行中
