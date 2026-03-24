# SPAGS Audit Report

- Generated at: `2026-03-24T00:32:26Z`
- Repo root: `/Users/openingcloud/IdeaProjects/PG/r2_gaussian`
- Audit signature: `cb2b9a6f9707`

## 模块映射

### SPS

- Intent: 伪视角生成、伪标签监督、稀疏视角先验合成链路
- Participates in training: `yes`
- Controlled by flags: `yes`
- Status: `wired`
- Code paths:
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/train.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/pseudo_view_utils.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/advanced_pseudo_label.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/warp_utils.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/depth_estimator.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/fsgs_complete.py`

### ADM

- Intent: densification / prune / proximity-guided densification
- Participates in training: `yes`
- Controlled by flags: `yes`
- Status: `wired`
- Code paths:
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/train.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/fsgs_proximity_optimized.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/fsgs_proximity.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/realistic_proximity_guided.py`

### GAR

- Intent: depth / graph Laplacian / smoothness / geometry-aware regularization
- Participates in training: `yes`
- Controlled by flags: `yes`
- Status: `wired`
- Code paths:
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/train.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/loss_utils.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/depth_estimator.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/fsgs_depth_renderer.py`
  - `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/utils/fsgs_complete.py`

## 高风险发现

### multi_gaussian 伪视角 loss 退化为自比较

- Severity: `high`
- File: `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/train.py`
- Lines: `[370]`
- Summary: 当前 pseudo-view loss 把 pseudo_image 与自身 detach 后做 L1，几乎不提供有效监督信号，属于优先修复项。

### arguments 默认值重复定义且后值覆盖前值

- Severity: `high`
- File: `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/arguments/__init__.py`
- Lines: `[48, 60, 65, 67, 68, 69, 80, 81, 84, 85]`
- Summary: 多个 FSGS/depth 参数在 ModelParams 中被重复赋值，导致配置意图和实际默认行为可能不一致。

### SSS hybrid optimizer 已创建但训练更新仍走原 optimizer

- Severity: `high`
- File: `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/train.py`
- Lines: `[37, 144, 914]`
- Summary: 训练循环里没有看到 sss_optimizer.step()，这意味着 SSS 相关优化器可能根本没有参与参数更新。

### 多个研究开关默认启用，容易发生隐式串扰

- Severity: `medium`
- File: `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/r2_gaussian/arguments/__init__.py`
- Lines: `[53, 58, 54, 79, 82]`
- Summary: 默认开启会让 SPAGS 候选配置难以做到显式 on/off，对快筛和矩阵基线都不友好。

### FSGS complete / pseudo / depth 多层启用逻辑存在重叠风险

- Severity: `medium`
- File: `/Users/openingcloud/IdeaProjects/PG/r2_gaussian/train.py`
- Lines: `[194, 200, 223, 231, 564, 231]`
- Summary: 完整系统、旧版 pseudo 路径和 depth 监督路径并行存在，默认值稍有变化就可能造成隐式串扰或行为回退。

## 最小可复现验证

- Files present: `True`
- Duplicate defaults detected: `5`
- Findings detected: `5`
