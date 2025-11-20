# R²-Gaussian 项目进度记录

**最后更新**: 2025-11-20 16:18

---

## 已完成

### 2025-11-20: Bino 训练脚本修复与批量启动
- 诊断并修复 Bino 训练在 5000 步停止的问题（test_iterations 配置末尾有错误的 `1`）
- 创建单任务训练脚本 `scripts/train_bino_foot3.sh`，支持自定义器官和视角数
- 创建批量训练脚本 `scripts/train_bino_batch.sh`，支持并行训练多器官
- 创建后台批量启动脚本 `scripts/train_bino_all_background.sh`，实现无人值守训练
- 创建训练监控脚本 `scripts/monitor_bino_training.sh`，实时监控所有训练任务
- 成功启动所有 5 个器官（foot、chest、head、abdomen、pancreas）的 3 views Bino 训练任务
- 编写完整的使用文档 `scripts/BINO_TRAINING_GUIDE.md` 和状态报告 `BINO_TRAINING_STATUS.md`
- 验证训练进程正常运行，loss 正常下降，GPU 使用率 100%

---

## 待完成

### Bino 实验相关
- 等待训练完成（预计 50-60 分钟），收集所有器官的最终评估结果（30000 步）
- 对比 Bino 与 R² Baseline 的性能差异，分析 Opacity Decay 策略的有效性
- 如果 Bino 效果不佳，考虑调整 `opacity_decay_factor` 参数或密化策略
- 考虑将训练任务分配到 GPU 1，充分利用双卡资源加速训练

### 后续实验计划
- 测试 Bino 在 6 views 和 9 views 数据集上的表现
- 将 Bino 的 Opacity Decay 策略与其他技术（FSGS、CoR-GS、Graph Laplacian）组合测试
- 分析不同器官对 Bino 方法的敏感度差异

### 技术债务
- 之前失败的训练目录（`output/2025_11_20_15_46_foot_3views_bino`）需要清理或归档
- 考虑优化训练脚本，支持自动 GPU 负载均衡（当前所有任务都在 GPU 0）
- 添加训练完成后的自动邮件通知功能
