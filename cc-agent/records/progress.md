# R²-Gaussian 项目进度记录

> 最后更新：2025-11-23 21:13

---

## 📋 已完成

### SSS (Student Splatting and Scooping) 修复与验证（2025-11-23）

- **诊断 SSS v2 失败根因**：发现 `r2_gaussian/gaussian/render_query.py:151` 使用 `density` 而非 `opacity`，导致 Signed Opacity 参数完全未学习，30k 迭代性能从 28.5dB 持续下降至 27.9dB
- **修复核心 bug**：在 `render_query.py:147-156` 添加条件判断 `opacity = pc.get_opacity if pc.use_student_t else None`，仅 3 行代码，完全向下兼容 baseline
- **2k 快速验证成功**：Opacity 参数正常学习（range=[-0.171, 1.515]，std=0.163，49968 个不同值），Nu 参数 range=[7.61, 10.61]，证明 Student's t 渲染正常工作
- **启动 30k 完整训练**：PID 150707，输出目录 `output/2025_11_23_21_12_foot_3views_sss_v3_full/`，目标 PSNR>28.8dB（超越 baseline 28.487dB）

---

## 🔄 待完成

### SSS v3 后续工作

- **等待 30k 训练完成**：预计 2025-11-24 凌晨 3-5 点完成（6-8 小时），需验证最终 PSNR 和 SSIM 是否超越 baseline
- **分析训练结果**：检查 Opacity 学习曲线、Nu 参数分布、组件回收统计，确认 Student's t 长尾分布优势是否体现
- **Balance Loss 测试**：如基础训练稳定，测试启用 L1 正则化（`opacity_reg_weight=0.001`）对 sparsity 的影响
- **性能对比实验**：完成 SSS v3 vs baseline vs SSS v1（高斯）的全面对比，验证各组件贡献度

### 其他待办

- **GR-Gaussian 实验验证**：代码已实现（Graph Laplacian + De-Init），等待 SSS 完成后启动实验
- **CoR-GS 结果分析**：检查之前启动的 CoR-GS Stage 3 训练结果（如已完成）
- **X²-Gaussian P0 对齐验证**：K-Planes 初始化和 TV 损失已修改，需实验验证效果

---

## 📊 关键指标记录

| 实验版本 | PSNR 2D (dB) | SSIM 2D | 状态 | 说明 |
|---------|-------------|---------|------|------|
| **Baseline** | 28.487 | 0.900 | ✅ | 高斯 + density |
| **SSS v1** | 28.524 | 0.897 | ✅ | 高斯 + opacity（30k） |
| **SSS v2** | 27.869 ❌ | 0.885 | ❌ | Student's t + 错误 density（bug） |
| **SSS v3 (2k)** | - | - | 🔄 | Student's t + 正确 opacity（验证通过） |
| **SSS v3 (30k)** | 目标 >28.8 | 目标 >0.905 | 🚀 | 训练中，预计明早完成 |

---

## 🐛 已知技术债

- **Balance Loss 暂时禁用**：`opacity_reg_weight=0.0`，需在 Student's t 渲染稳定后重新测试最优权重
- **组件回收阈值未调优**：当前使用默认 `opacity_threshold=0.005, max_recycle_ratio=0.05`，可能需要针对 CT 任务调整
- **学习率调度未验证**：opacity 和 nu 的学习率衰减策略（init→final）需实验验证是否最优

---

## 💡 重要发现

### SSS 修复关键洞察（2025-11-23）

1. **问题本质**：参数存在但未进入计算图 → 梯度为 0 → 参数冻结
2. **诊断方法**：检查训练日志中参数变化 + 检查点文件分析 → 快速定位
3. **修复策略**：条件化参数选择 + 向下兼容设计 → 3 行代码解决
4. **验证标准**：参数标准差 > 0.01 + 包含负值（Signed Opacity）+ 每个点独立优化

### 教训

- **CUDA 实现正确 ≠ 系统工作**：Student's t 渲染和梯度传递都正确，但 Python 层参数传递错误导致整个系统失效
- **快速迭代验证重要性**：2k 迭代（~2 分钟）即可验证参数学习，避免浪费 30k 训练时间
- **参数可视化是关键**：直接检查参数分布（range, std, 唯一值数量）比只看 loss 曲线更能发现问题
