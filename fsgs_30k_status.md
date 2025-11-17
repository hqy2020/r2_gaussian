## 📊 FSGS 30k 训练进度报告

**训练状态**: ✅ 运行中 (已完成 890/30000 iterations, 约 3%)

**当前指标**:
- Loss: 1.3e-02
- 高斯点数: 130,000 (已增长 30%)
- 训练速度: ~18 it/s

**预计完成时间**: 约 7-8 小时后 (凌晨 2-3 点)

**监控方式**:
1. 每 30 分钟手动检查:
   ```bash
   bash /home/qyhu/Documents/r2_ours/r2_gaussian/monitor_fsgs_30k.sh
   tail -30 /home/qyhu/Documents/r2_ours/r2_gaussian/fsgs_30k_monitor.log
   ```

2. 查看训练日志:
   ```bash
   tail -50 /home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_fsgs_30k/training.log
   ```

3. 检查评估结果目录:
   ```bash
   ls -lh /home/qyhu/Documents/r2_ours/r2_gaussian/output/2025_11_17_foot_3views_fsgs_30k/eval/
   ```

**已设置的评估点**: 5000, 10000, 15000, 20000, 25000, 30000

**下一步**:
- ⏰ 每 30 分钟检查一次进度
- 🎯 等待 iter 5000 的首次评估结果
- 📊 完成后分析完整的 6 个评估点结果

