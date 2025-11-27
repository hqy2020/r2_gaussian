# R²-Gaussian 组合实验待办清单

**最后更新**: 2025-11-27
**当前分支**: integration（已整合 4 种技术）

---

## 当前状态

### 已完成 ✅

- [x] **分支整合**
  - [x] 合并 init-pcd 分支（密度加权采样）
  - [x] 合并 x2-hqy 分支（K-Planes + TV 正则化）
  - [x] 合并 fsgs-hqy 分支（伪视角深度监督）
  - [x] Bino 已在 integration 分支（双目一致性损失）

- [x] **验证整合**
  - [x] 核心模块可导入
  - [x] 关键参数全部注册
  - [x] 创建组合实验脚本

---

## Phase 1: 快速验证（10k iterations）

**目标**: 筛选有效组合，预计 1-2 天

### GPU 0 任务
- [ ] 组合 A (Init-PCD + X²) - Foot 3views
- [ ] 组合 A (Init-PCD + X²) - Chest 3views
- [ ] 组合 A (Init-PCD + X²) - Pancreas 3views

### GPU 1 任务
- [ ] 组合 B (Init-PCD + X² + FSGS) - Foot 3views
- [ ] 组合 B (Init-PCD + X² + FSGS) - Chest 3views
- [ ] 组合 B (Init-PCD + X² + FSGS) - Pancreas 3views

### 筛选标准
- 10k PSNR < 28.0 → 放弃
- 10k PSNR 28.0-28.2 → 调参重试
- 10k PSNR >= 28.2 → 继续 30k

---

## Phase 2: 完整训练（30k iterations）

**目标**: 选定组合全器官验证，预计 3-6 天

### 器官优先级
1. **Foot** (P0) - 所有技术已验证，baseline 28.487
2. **Chest** (P1) - baseline 最低 26.506，提升空间大
3. **Pancreas** (P2) - X² 验证 +0.15 dB，baseline 28.767
4. **Head** (P3) - baseline 26.692
5. **Abdomen** (P4) - baseline 最高 29.290

### 视角优先级
3 views → 6 views → 9 views

### 组合 A 完整验证
- [ ] Foot: 3v / 6v / 9v
- [ ] Chest: 3v / 6v / 9v
- [ ] Pancreas: 3v / 6v / 9v
- [ ] Head: 3v / 6v / 9v
- [ ] Abdomen: 3v / 6v / 9v

### 备选组合（根据 Phase 1 结果）
- [ ] 组合 C (Init + X² + Bino)
- [ ] 组合 D (Init + FSGS + Bino)

---

## Phase 3: 最优组合全面验证

**目标**: 15 组实验全部超越 SOTA

### 性能目标

| 器官 | Baseline | 目标 (3v) | 目标 (6v) | 目标 (9v) |
|------|----------|-----------|-----------|-----------|
| Foot | 28.487 | 29.0+ | 30.0+ | 31.0+ |
| Chest | 26.506 | 27.1+ | 28.0+ | 29.0+ |
| Head | 26.692 | 27.2+ | 28.0+ | 29.0+ |
| Abdomen | 29.290 | 29.6+ | 30.5+ | 31.5+ |
| Pancreas | 28.767 | 29.2+ | 30.0+ | 31.0+ |

### 最终验证
- [ ] 所有 15 组实验完成
- [ ] 结果整理到表格
- [ ] 更新记忆库

---

## 快速命令参考

```bash
# 单个实验
./scripts/run_combo_experiments.sh A foot 3 0

# Phase 1 批量启动
./scripts/run_combo_batch.sh phase1

# 查看结果
./scripts/analyze_combo_results.sh

# 监控训练
tail -f logs/phase1_gpu0.log
```

---

## 4 种组合配置速查

| 组合 | 技术 | 关键参数 |
|------|------|----------|
| **A** | Init-PCD + X² | `--sampling_strategy density_weighted --enable_kplanes --lambda_plane_tv 0.002` |
| **B** | A + FSGS | A + `--enable_fsgs_depth --depth_pseudo_weight 0.03` |
| **C** | A + Bino | A + `--enable_binocular_consistency --binocular_start_iter 10000` |
| **D** | Init + FSGS + Bino | 无 K-Planes，节省显存 |

---

## 问题跟踪

### 待解决
- [ ] （暂无）

### 已解决
- [x] train.py 合并冲突 - 保留所有技术代码
- [x] initialize_pcd.py 重复降噪代码 - 删除重复部分
- [x] arguments/__init__.py 参数冲突 - 合并所有参数

---

## 相关文档

- 实验计划: `~/.claude/plans/buzzing-plotting-tulip.md`
- 记忆模板: `cc-agent/记忆模板.md`
- MCP 工具指南: `cc-agent/MCP工具使用指南.md`
- 工作流指南: `cc-agent/工作流详细指南.md`

---

**下一步**: 启动 Phase 1 快速验证 `./scripts/run_combo_batch.sh phase1`
