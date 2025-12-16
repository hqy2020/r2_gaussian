原因分析（结合你现有 15 场景历史结果）

  - 从 output/*/eval/iter_030000/eval2d_render_test.yml 汇总看，你的 spags_3k 相比 baseline 在 15 场景平均 psnr2d 反而约 -0.18dB，
    ssim2d 仅 +0.0007；退化主要集中在 chest(3/6)、abdomen(6/9)、pancreas(3/6/9)。
  - 关键混淆项：你历史 spags_3k 的 ply_path 指向 data/369-sps-3k/*（见例如 output/2025_12_05_13_58_chest_3views_spags_3k/
    cfg_args.yml:4），而当前仓库标准 SPS init 是 data/369-sps/*；若点数/分布不一致，就不是“纯 SPS/纯 SPAGS”对比，容易出现不稳定
    结论。
  - SPS 的本质副作用：密度加权采样会让初始化点更聚集且整体密度均值更高（你现有 data/369-sps 相比 data/369 多数场景 density 均值显著
    上升），训练早期更容易出现“过密/过暗/过拟合”，对软组织场景（abdomen/pancreas）尤其敏感，从而拉低新视角 psnr2d/ssim2d。
  - 若生成 SPS init 没在 r2_gaussian_new 环境中跑，scipy 缺失会导致去噪被跳过（FDK 伪影更强），SPS 更容易被伪影引导
    （initialize_pcd.py 里会直接跳过 gaussian denoise）。

  修复方法总结（已落地到代码）

  - 解耦“采样位置”和“初始化密度”：新增 --sps_density_init_mode raw|match_valid_mean|match_valid_median，可把 SPS 采样点的
    密度缩放到与“有效体素(>density_thresh)”均值/中位数一致，并对缩放系数做裁剪，降低 SPS 导致的整体初始密度偏高与跨器官不稳
    （initialize_pcd.py:88、initialize_pcd.py:389）。
  - 更新 15 场景 SPS init 生成脚本默认超参：更保守的采样强度（uniform_ratio=0.3、gamma=0.8）+ 启用 match_valid_mean（cc-agent/
    scripts/generate_sps_init_369.sh:90）。
  - 同步文档：README.md:96、cc-agent/docs/SPS_空间先验播种.md:191 已更新到正确路径/正确命令。

  之后要做的事（按优先级）

  - 先消除混淆：确保 baseline 和 spags 用同一套 n_points（建议 50k）与同一版本代码；不要用旧的 data/369-sps-3k 和新的 data/369 混
    着比。
  - 重新生成 SPS init（15 场景）：bash ./cc-agent/scripts/generate_sps_init_369.sh <gpu_id> 50000 1（输出到 data/369-sps）。
  - 先 smoke 再 full：可临时 export SPAGS_ITERS=3000 跑完 15 场景看趋势，再回到 30000。
  - 训练与评估：用 ./cc-agent/scripts/run_spags_ablation.sh baseline <organ> <3|6|9> <gpu> 和 ./cc-agent/scripts/
    run_spags_ablation.sh spags <organ> <3|6|9> <gpu>；评估看 output/<run>/eval/iter_030000/eval2d_render_test.yml 或跑 python
    test.py -m output/<run>。
  - 若仍有个别器官退化：优先尝试把 --sps_density_init_mode 换成 match_valid_median，或进一步增大 sps_uniform_ratio / 降低
    sps_density_gamma（通常对 pancreas/abdomen 更稳）。

  如何“正确运行/使用” cc-agent 下面的文档

  - 文档就是 Markdown：在 cc-agent/docs/（例如 cc-agent/docs/SPS_空间先验播种.md:186、cc-agent/docs/GAR_几何感知细化.md、cc-agent/
    docs/SPAGS_技术架构.md）；直接用编辑器打开，或终端 less cc-agent/docs/SPS_空间先验播种.md。
  - 文档里的命令建议优先用脚本跑（它会激活 r2_gaussian_new、做 CUDA 检查并 cd 到项目根）：cc-agent/scripts/
    generate_sps_init_369.sh、cc-agent/scripts/run_spags_ablation.sh（示例见 cc-agent/docs/SPS_空间先验播种.md:191）。

---

## Smoke Test 实验结果（2025-12-15）

### 实验配置
- **迭代次数**: 3000（smoke test）
- **SPS 初始化**: data/369-sps/（2025-12-13 生成，使用 match_valid_mean）
- **Baseline 初始化**: data/369/init_*.npy（50K 点）
- **GPU**: 双卡并行（GPU 0: baseline, GPU 1: spags）

### 结果对比

| 场景 | baseline PSNR | spags PSNR | Δ PSNR | baseline SSIM | spags SSIM | Δ SSIM |
|------|--------------|------------|--------|--------------|------------|--------|
| chest_3views | 26.41 | 25.46 | **-0.95** | 0.839 | 0.826 | **-0.013** |
| chest_6views | 32.79 | 32.31 | **-0.49** | 0.926 | 0.923 | -0.003 |
| chest_9views | 36.04 | 35.79 | -0.25 | 0.951 | 0.950 | -0.001 |
| foot_3views | 28.26 | 28.37 | +0.11 | 0.893 | 0.892 | -0.001 |
| foot_6views | 32.01 | 32.28 | **+0.27** | 0.935 | 0.938 | +0.003 |
| foot_9views | 34.57 | 34.72 | +0.15 | 0.955 | 0.957 | +0.002 |
| head_3views | 26.68 | 26.98 | **+0.30** | 0.923 | 0.920 | -0.002 |
| head_6views | 32.37 | N/A | - | 0.970 | N/A | - |
| head_9views | 34.84 | N/A | - | 0.978 | N/A | - |
| **平均 (n=7)** | **30.97** | **30.85** | **-0.12** | **0.917** | **0.915** | **-0.002** |

### 关键发现

1. **chest 场景退化明显**（与历史诊断一致）：
   - chest_3views: PSNR -0.95 dB（最严重）
   - chest_6views: PSNR -0.49 dB
   - 原因：SPS 密度加权采样导致初始化密度偏高，chest 软组织场景更敏感

2. **foot/head 场景有提升**：
   - foot_6views: PSNR +0.27 dB
   - head_3views: PSNR +0.30 dB
   - 这些场景骨骼结构清晰，SPS 能更好地定位高密度区域

3. **实验未完成**：
   - abdomen 和 pancreas 训练中断（GPU 资源竞争）
   - head_6views/9views 的 spags 结果缺失

### 结论与后续建议

- **当前 SPS 配置（match_valid_mean）对 chest 场景仍有退化**
- 建议尝试以下调整：
  1. `--sps_density_init_mode match_valid_median`（对异常值更鲁棒）
  2. 增大 `sps_uniform_ratio` 0.3 → 0.4（减少密度加权采样占比）
  3. 降低 `sps_density_gamma` 0.8 → 0.6（弱化密度权重）
- 需要补跑 abdomen 和 pancreas 场景以获得完整对比

---
## codex 建议
• SPS 做完但“不好”是正常现象：它更像初始化加速/稳定手段，不一定单独把 15 场景都抬上去；真正要“稳定超过 baseline”，下一步应把重心转
  到 ADM（优先）→ GAR（其次）→ 组合（你在 cc-agent/codex/sps.md:1 的 smoke 结果也已经指向 chest 是主要拖累项）。

  建议下一步按这个顺序推进

  - 先把 SPS 止损：只盯 chest_3views 做 3 个小网格（match_valid_median / uniform_ratio↑ / gamma↓），跑 SPAGS_ITERS=3000~5000 快速筛
    掉“明显掉分”的配置；同时用 cc-agent/docs/DIAGNOSIS_TOOLS.md:86 的 diagnose_sps.py 对比 data/369 vs data/369-sps，确认是不是“过
    聚集/密度偏高”导致的退化。
  - 转向 ADM（最可能稳定赢 baseline）：先跑 15 场景 smoke，再跑 30k 全量；直接用脚本：cc-agent/scripts/run_baseline_369_15.sh:1
    对齐 baseline，cc-agent/scripts/run_adm_369_15.sh:1 跑 ADM；如果 ADM 个别场景回退，再按 cc-agent/codex/adm.md:1 的建议优先调
    --adm_max_range/--adm_warmup_iters/--lambda_plane_tv。
  - 再做 GAR（用诊断选阈值）：跑 cc-agent/scripts/run_gar_369_15.sh:1，重点看训练日志里的 [GAR 诊断] 和点数是否增长；再用 cc-agent/
    + uniform_ratio↑ + gamma↓），不要硬追“所有器官都靠 SPS 提升”。

  你现在就可以跑的最小闭环（建议先 smoke）

  - bash cc-agent/scripts/generate_sps_init_369.sh <gpu> 50000 1
  - SPAGS_ITERS=5000 SPAGS_TEST_ITERS="1 5000" bash cc-agent/scripts/run_baseline_369_15.sh <gpu> exp_smoke
  - SPAGS_ITERS=5000 SPAGS_TEST_ITERS="1 5000" bash cc-agent/scripts/run_adm_369_15.sh <gpu> exp_smoke

  如果你把这三组跑完后的 15 场景 eval2d_render_test.yml（baseline/adm/gar 或 baseline/spags）前缀发我（或告诉我 output 目录名），我
  可以基于真实"完整 15 场景"结果给出下一轮最小调参集合（优先把 chest 拉回不掉分）。

---

## SPS 止损调参实验结果（2025-12-15）

### 实验配置
- **目标场景**: chest_3views（之前退化最严重，-0.95 dB）
- **迭代次数**: 5000（smoke test）
- **对照组**: baseline（data/369/init_chest_50_3views.npy）

### 测试配置

| 配置 | init_mode | uniform_ratio | gamma | 输出目录 |
|------|-----------|---------------|-------|----------|
| A | match_valid_median | 0.3 | 0.8 | data/369-sps-A/ |
| B | match_valid_mean | 0.4 | 0.8 | data/369-sps-B/ |
| C | match_valid_mean | 0.3 | 0.6 | data/369-sps-C/ |

### 结果对比

| 配置 | PSNR (dB) | SSIM | Δ PSNR vs Baseline |
|------|-----------|------|---------------------|
| **baseline** | 26.46 | 0.839 | - |
| **sps-A** (median) | **27.50** | **0.850** | **+1.04** |
| **sps-B** (uniform=0.4) | 27.46 | 0.851 | +1.00 |
| **sps-C** (gamma=0.6) | 27.15 | 0.851 | +0.69 |

### 关键发现

1. **配置 A（`match_valid_median`）表现最好**：
   - PSNR 提升 **1.04 dB**（26.46 → 27.50）
   - SSIM 提升 **0.011**（0.839 → 0.850）
   - 与之前使用 `match_valid_mean` 退化 0.95 dB 相比，**反转为显著提升**

2. **原因分析**：
   - SPS 诊断显示：使用 `match_valid_mean` 时，SPS 密度是 baseline 的 **2.08 倍**
   - `match_valid_median` 对 FDK 伪影高密度区域更鲁棒（scale=0.792）
   - 中位数比均值更能抵抗异常值影响

3. **配置 B/C 也有效但略逊**：
   - 增大 uniform_ratio (+1.00 dB) 和降低 gamma (+0.69 dB) 都能改善
   - 但 `match_valid_median` 效果最好

### 结论

**`--sps_density_init_mode match_valid_median` 是正确的修复方向**

建议更新 `generate_sps_init_369.sh` 脚本，将默认配置改为：
```bash
--sps_density_init_mode match_valid_median  # 原为 match_valid_mean
```

### 下一步

1. 使用 `match_valid_median` 重新生成 15 场景 SPS 初始化
2. 跑 15 场景全量训练验证 SPS 效果