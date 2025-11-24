# GR-Gaussian 快速修复指南

## 📋 诊断总结

**问题**: GR-Gaussian 性能低于 baseline 0.96 dB

**根因**:
1. **代码问题 (30%)**:
   - λ_lap=0.0008 过小 → Graph Laplacian loss 被压制
   - De-Init 未启用 → 缺失 50% 核心功能

2. **算法问题 (70%)**:
   - graph_update_interval 从 500 改到 1000 → 性能下降 0.78 dB
   - 双模型混淆因素 → 对比不公平

---

## 🚀 快速修复步骤

### 步骤 1: 快速验证 (2-3小时)

```bash
# 运行 10k 快速测试
bash scripts/run_gr_foot3_10k_quick_test.sh

# 查看结果
OUTPUT_DIR=$(ls -td output/*gr_foot3_10k_quick_test* | head -1)
cat $OUTPUT_DIR/eval/iter_010000/eval2d_render_test.yml | grep -E "psnr_2d|ssim_2d" | head -2
```

**判断标准**:
- ✅ PSNR ≥ 28.5 dB → 进入步骤 2
- ❌ PSNR < 28.5 dB → 需要调整超参数

---

### 步骤 2: 完整训练 (6-8小时)

```bash
# 运行 30k 完整训练
bash scripts/run_gr_foot3_30k_FIXED.sh
```

**预期性能**:
- PSNR > 28.8 dB (超越 baseline 28.547 dB)
- SSIM > 0.910

---

## 📊 监控命令

### 实时监控

```bash
# 查找最新的 GR 训练目录
OUTPUT_DIR=$(ls -td output/*gr_foot3* | head -1)

# 使用监控脚本（每10秒刷新）
watch -n 10 "bash scripts/monitor_gr_training.sh $OUTPUT_DIR"
```

### 查看 Graph Loss

```bash
# 查看 graph_loss 变化趋势
grep "graph_loss" $OUTPUT_DIR/train.log | tail -20
```

**期望值**:
- 修复前: graph_loss ≈ 0.000000~0.000001 ❌
- 修复后: graph_loss ≈ 0.0001~0.001 ✅ (提升 100-1000倍)

### 验证 De-Init

```bash
# 检查 De-Init 是否生效
grep "De-Init" $OUTPUT_DIR/train.log
```

**期望输出**:
```
[De-Init] Applying Gaussian filter with sigma=3.0
[De-Init] Noise reduced: 0.XXXXXX (mean absolute change)
```

---

## 🎯 性能目标

| 指标 | Baseline | 目标 | 提升 |
|------|----------|------|------|
| PSNR 2D | 28.547 dB | > 28.8 dB | +0.25 dB |
| SSIM 2D | 0.9008 | > 0.910 | +0.01 |

---

## 🔧 核心修复参数

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `gaussiansN` | 2 | **1** | 单模型公平对比 |
| `graph_update_interval` | 1000 | **500** | 恢复 good case 配置 (+0.78 dB) |
| `graph_lambda_lap` | 0.0008 | **0.008** | 增强 Graph Laplacian (10x) |
| `enable_denoise` | - | **true** | 启用 De-Init (+0.2-0.5 dB) |
| `denoise_sigma` | - | **3.0** | 论文推荐值 |

---

## 📈 预期提升分解

1. **graph_update_interval=500**: +0.78 dB
2. **De-Init 降噪初始化**: +0.2~0.5 dB
3. **λ_lap 增大 10倍**: +0.1~0.3 dB

**总计预期**: +1.0~1.5 dB

---

## ⚠️ 故障排查

### 问题 1: graph_loss 仍然很小 (< 1e-5)

**原因**: λ_lap 可能仍然不够大

**解决**:
```bash
# 尝试更大的 lambda
python train.py ... --graph_lambda_lap 0.08  # 提升到 100x
```

### 问题 2: De-Init 无日志输出

**原因**: 参数传递可能有问题

**解决**:
```bash
# 检查参数是否正确传递
grep "enable_denoise" $OUTPUT_DIR/cfg_args.yml
```

### 问题 3: 性能仍低于 baseline

**下一步**:
1. 检查是否真的是单模型 (gaussiansN=1)
2. 对比 baseline 的配置文件
3. 考虑进行超参数网格搜索

---

## 📝 实验记录

### 实验配置对比

```bash
# Baseline (单模型)
output/foot_3views_r2_baseline_1113/
  PSNR: 28.547 dB
  SSIM: 0.9008
  gaussiansN: 1
  enable_graph_laplacian: false

# Good Case (双模型+GR, interval=500)
output/2025_11_18_gr_SUCCESS/
  PSNR: 28.366 dB (-0.18 dB)
  SSIM: 0.9006
  gaussiansN: 2
  graph_update_interval: 500

# Bad Case (双模型+GR, interval=1000)
output/2025_11_23_20_25_gr_foot3_30k_final/
  PSNR: 27.586 dB (-0.96 dB)
  SSIM: 0.9009
  gaussiansN: 2
  graph_update_interval: 1000

# Fixed (单模型+完整GR)
output/[待运行]/
  目标: PSNR > 28.8 dB
  gaussiansN: 1
  graph_update_interval: 500
  graph_lambda_lap: 0.008
  enable_denoise: true
```

---

## 📚 后续工作

### 短期 (1-2天)

- [ ] 验证修复方案有效性
- [ ] 超参数网格搜索优化
- [ ] 实现 PGA (Pixel-Graph-Aware Gradient)

### 中期 (1周)

- [ ] 消融实验: 分离各组件贡献
- [ ] 对比 CoR-GS + GR-Gaussian 组合
- [ ] 在其他器官数据集验证

---

**生成时间**: 2025-11-24
**诊断报告**: cc-agent/experiments/gr_diagnosis_report.md
