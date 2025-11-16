# CoR-GS 训练监控系统 - 完整指南

## 系统概览

本文档描述了为 CoR-GS Stage 1 foot 3/6/9 views 实验设计的**全自动监控和分析系统**。

```
训练进程 (3个)
    ↓
  日志输出
    ↓
监控脚本 (/tmp/main_monitor.sh)
    ├─ 每30秒检查完成状态
    ├─ 每5分钟打印进度报告
    └─ 当全部完成时自动触发
        ↓
    分析脚本 (/tmp/generate_final_report.py)
        ├─ 加载YAML评估文件
        ├─ 与Baseline对比
        └─ 生成报告
            ↓
        最终报告 (.md + .json)
```

---

## 文件清单

### 训练脚本 (原始)

| 位置 | 用途 | 状态 |
|------|------|------|
| `train.py` | 主训练脚本 | ✓ 运行中 (3个进程) |
| `r2_gaussian/disagreement_metrics.py` | CoR-GS Metrics | ✓ 正常计算 |
| `data/r2-sax-nerf/*.pickle` | 训练数据 | ✓ 已加载 |
| `data/r2-sax-nerf/init_*.npy` | 初始化点云 | ✓ 已生成 |

### 监控脚本 (新增)

| 位置 | 用途 | 状态 | 输出 |
|------|------|------|------|
| `/tmp/main_monitor.sh` | 主监控脚本 | ✓ 运行 (PID 3710299) | `/tmp/main_monitor.log` |
| `/tmp/continuous_monitor.py` | 持续进度监控 | ✓ 运行 (PID 3709824) | 控制台 |
| `/tmp/generate_final_report.py` | 结果分析和报告 | ⏳ 自动触发 | `.md` + `.json` |
| `/tmp/monitor_training.py` | 备用监控脚本 | 可选 | - |

### 输出文档 (自动生成)

| 位置 | 文件名 | 类型 | 何时生成 |
|------|--------|------|---------|
| `cc-agent/records/` | `foot_369_results_2025_11_17.md` | Markdown | 训练完成后 |
| `cc-agent/records/` | `foot_369_results_2025_11_17.json` | JSON | 训练完成后 |
| `cc-agent/records/` | `training_monitor_2025_11_17.md` | Markdown | 手动记录 |
| `cc-agent/records/` | `EXPERIMENT_STATUS_2025_11_17.md` | Markdown | 手动记录 |
| `cc-agent/records/` | `QUICK_REFERENCE_2025_11_17.md` | Markdown | 手动记录 |

### 训练日志 (临时)

| 位置 | 内容 | 大小 |
|------|------|------|
| `/tmp/foot_3views_corgs_2025_11_17.log` | foot_3views 训练日志 | ~50+ MB |
| `/tmp/foot_6views_corgs_2025_11_17.log` | foot_6views 训练日志 | ~30+ MB |
| `/tmp/foot_9views_corgs_2025_11_17.log` | foot_9views 训练日志 | ~20+ MB |
| `/tmp/main_monitor.log` | 监控脚本日志 | ~1 MB |

---

## 工作流程详解

### 第 1 阶段: 训练启动 (00:05-00:10)

```
用户命令
  ↓
启动 3 个训练进程
  ├─ foot_3views: /tmp/foot_3views_corgs_2025_11_17.log
  ├─ foot_6views: /tmp/foot_6views_corgs_2025_11_17.log
  └─ foot_9views: /tmp/foot_9views_corgs_2025_11_17.log
  ↓
启动监控脚本
  ├─ /tmp/main_monitor.sh (后台)
  └─ /tmp/continuous_monitor.py (后台)
```

### 第 2 阶段: 训练进行中 (00:10-00:35)

```
每30秒
  ↓
main_monitor.sh 检查完成条件:
  ├─ eval/iter_010000/eval2d_render_test.yml 是否存在?
  ├─ 所有3个都存在? → 完成!
  └─ 否 → 继续监听
  ↓
每5分钟 (监控脚本自动)
  ↓
打印进度报告:
  ├─ 当前迭代数
  ├─ 当前Loss
  ├─ 预计剩余时间
  └─ 进度条
```

### 第 3 阶段: 完成检测 (00:35-00:40)

```
当条件满足:
  ✓ foot_3views: 10000 iterations + eval 完成
  ✓ foot_6views: 10000 iterations + eval 完成  
  ✓ foot_9views: 10000 iterations + eval 完成
  ↓
main_monitor.sh 自动执行:
  ↓
generate_final_report.py
  ├─ 加载 3 个 YAML 文件
  ├─ 提取 PSNR, SSIM 指标
  ├─ 与 R² Baseline 对比
  │   └─ PSNR: 28.547
  │   └─ SSIM: 0.9008
  ├─ 生成分析结论
  └─ 保存报告
      ├─ foot_369_results_2025_11_17.md
      └─ foot_369_results_2025_11_17.json
  ↓
自动显示报告内容
```

---

## 监控指标说明

### Loss 曲线

```
含义: 训练目标函数值
目标: 单调下降，收敛到稳定值
异常: 震荡、上升、无反应
```

| Config | 初始 | iter 500 | iter 1000 | 趋势 |
|--------|------|----------|-----------|------|
| 3views | 0.14 | 0.0073 | ~0.006 | ✓ 快速下降 |
| 6views | - | ~0.009 | ~0.012 | ✓ 正常 |
| 9views | - | ~0.008 | ~0.008 | ✓ 稳定 |

### Gaussian 点数

```
含义: 动态密化的高斯点数
目标: 从 50000 增长到 80000-100000
过程: 基于梯度信号的渐进式增长
```

### CoR-GS Metrics

```
Fitness = 1.0: 两个系统完全匹配
RMSE < 0.01: 几何差异极小
PSNR Diff > 50 dB: 渲染视觉一致性极高
SSIM Diff > 0.99: 结构相似性完美
```

---

## 预期的最终报告内容

### Markdown 格式 (.md)

```markdown
# CoR-GS Stage 1 - Foot 3/6/9 Views 实验结果报告

## 【核心结论】
- 所有/部分配置是否超越 baseline
- 改进幅度
- 后续建议

## 【详细分析】
### 1. 定量结果对比
| 配置 | PSNR (2D) | SSIM (2D) | vs Baseline |
...

### 2. Baseline 对比
R² Gaussian:
  PSNR: 28.547
  SSIM: 0.9008

### 3. CoR-GS Stage 1 验证
- Geometry Disagreement 指标
- Rendering Disagreement 指标
- 系统稳定性验证

## 【需要您的决策】
选项 A: 继续...
选项 B: 调优...
选项 C: 诊断...
```

### JSON 格式 (.json)

```json
{
  "3views": {
    "psnr_2d": 28.8,
    "ssim_2d": 0.905,
    "psnr_3d": 22.5,
    "ssim_3d": 0.65,
    "improvement": "+0.25 dB"
  },
  "6views": {...},
  "9views": {...},
  "baseline": {
    "psnr_2d": 28.547,
    "ssim_2d": 0.9008
  }
}
```

---

## 常用操作

### 查看进度

```bash
# 快速检查所有三个的最新进度
for f in /tmp/foot_*views_corgs_2025_11_17.log; do
  name=$(basename $f | sed 's/.log//')
  iter=$(grep -o '[0-9]\+/10000' $f | tail -1)
  echo "[$name] $iter"
done
```

### 实时监控

```bash
# 只看 loss 变化
tail -f /tmp/foot_3views_corgs_2025_11_17.log | grep "loss="

# 看完整的进度信息
tail -f /tmp/foot_3views_corgs_2025_11_17.log | grep Train
```

### 检查评估文件

```bash
# 查看是否生成了评估文件
ls -la output/2025_11_17_foot_*/eval/iter_010000/eval2d_render_test.yml

# 查看 YAML 内容
cat output/2025_11_17_foot_3views_corgs/eval/iter_010000/eval2d_render_test.yml
```

### 手动触发报告

```bash
# 如果自动报告失败，可以手动执行
python /tmp/generate_final_report.py
```

---

## 故障排除

### 监控脚本没有输出

```bash
# 检查是否还在运行
ps aux | grep main_monitor

# 查看是否有错误
tail -50 /tmp/main_monitor.log | grep -i error

# 重新启动
/tmp/main_monitor.sh > /tmp/main_monitor_new.log 2>&1 &
```

### 训练卡住

```bash
# 检查进程 CPU 使用率
ps aux | grep train.py | grep -v grep

# 如果 CPU% 为 0，说明进程可能卡死
# 可以查看最后的日志
tail -100 /tmp/foot_3views_corgs_2025_11_17.log | tail -30
```

### 评估文件未生成

```bash
# 检查输出目录
ls -la output/2025_11_17_foot_3views_corgs/eval/

# 检查权限
stat output/2025_11_17_foot_3views_corgs/

# 检查磁盘空间
du -sh output/2025_11_17_foot_*/
df -h .
```

---

## 后续工作

### 短期 (实验完成后)

1. **查看最终报告**
   ```
   cat cc-agent/records/foot_369_results_2025_11_17.md
   ```

2. **根据结果决策**
   - 所有超越 → 启动其他数据集
   - 部分超越 → 调参重试
   - 未超越 → 诊断实现

3. **更新 progress.md**
   ```bash
   # 记录最终结果
   ```

### 中期 (下一阶段)

- 实现 Stage 2 - Co-pruning
- 测试其他数据集 (chest, head, abdomen)
- 完整的 3/6/9 views 对比

### 长期 (完整项目)

- Stage 3 - Pseudo-view generation
- Stage 4 - Full system integration
- 论文撰写和发表

---

## 常见问题解答

**Q: 需要手动干预吗?**
A: 不需要。所有监控、检测和报告生成都是全自动的。

**Q: 可以在训练过程中关闭终端吗?**
A: 可以。所有脚本都在后台运行，与终端无关。

**Q: 完成时间会不会延迟?**
A: 如果 GPU 负载很高，可能会晚几分钟，但不会影响结果。

**Q: 可以中途暂停训练吗?**
A: 可以，但建议让其完成。如果需要中断，使用 `kill -9 <PID>`。

**Q: 报告生成失败了怎么办?**
A: 可以手动执行 `python /tmp/generate_final_report.py`。

---

## 技术支持

| 问题 | 解决方案 |
|------|--------|
| 进度不前进 | 检查日志最后 30 行 |
| 评估未生成 | 检查输出目录权限和磁盘空间 |
| 报告格式错误 | 查看 YAML 文件内容是否完整 |
| 监控卡住 | 重启监控脚本 |

---

**版本**: 1.0
**创建时间**: 2025-11-17 00:22 UTC
**维护者**: 深度学习调参与分析专家

