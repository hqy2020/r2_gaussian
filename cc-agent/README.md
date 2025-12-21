# CC-Agent 目录索引

AI 科研助手系统的核心工作目录，包含实验数据、脚本和生成的图片。

---

## 支持的 3DGS 方法 (6 种)

| 方法 | 全称 | 说明 |
|------|------|------|
| **spags** | SPAGS | 我们的方法 (SPS+GAR+ADM) |
| **baseline** | R²-Gaussian | 基准方法 |
| **xgaussian** | X-Gaussian | 球谐特征 + opacity |
| **fsgs** | FSGS | 邻近高斯密集化 |
| **dngaussian** | DN-Gaussian | 深度正则化 |
| **corgs** | CoR-GS | 协同正则化 |

---

## 实验数据 (`experiment/`)

### 主要实验结果

| 文件 | 说明 | 实验规模 |
|------|------|----------|
| `results_6methods_90experiments.md` | **6 方法 × 5 器官 × 3 视角对比** | 90 实验 |
| `ablation_results.md` | **SPAGS 消融实验** (7 配置正交组合) | 105 实验 |
| `ablation_results.json` | 消融实验结果 (JSON) | - |
| `all_90_experiments.json` | 90 个实验的完整配置 | - |
| `optimized_spags_selection.json` | 优化后的 SPAGS 选择 | - |

### 实验配置

- **方法对比**: 6 种 3DGS 方法 (SPAGS, baseline, xgaussian, fsgs, dngaussian, corgs)
- **器官**: chest, foot, head, abdomen, pancreas (5 种)
- **视角**: 3, 6, 9 views
- **消融配置**: sps, adm, gar, sps_adm, sps_gar, gar_adm, spags (7 种)

---

## 脚本 (`scripts/`)

### 训练脚本
| 文件 | 说明 |
|------|------|
| `run_spags_ablation.sh` | **消融实验主脚本** - 支持所有 6 种方法和消融配置 |
| `run_batch_training.sh` | 批量训练脚本 |
| `generate_sps_init_369.sh` | SPS 点云初始化生成脚本 (3/6/9 视角) |

### 绘图脚本
| 文件 | 说明 |
|------|------|
| `plot_7methods_3views.py` | 7 方法 3 视角对比图 |
| `plot_7methods_9views.py` | 7 方法 9 视角对比图 |
| `plot_6methods_qualitative.py` | 6 方法定性对比图 |
| `plot_fig4_5_qualitative.py` | 论文图 4.5 - 定性对比 |
| `plot_fig4_6_full_ablation.py` | 论文图 4.6 - 完整消融实验 |

### 诊断工具 (`scripts/diagnosis/`)
| 文件 | 说明 |
|------|------|
| `diagnose_sps.py` | SPS (采样策略) 诊断 |
| `diagnose_gar.py` | GAR (邻近密化) 诊断 |
| `diagnose_adm.py` | ADM (密度调制) 诊断 |
| `analyze_training.py` | 训练过程分析 |
| `generate_diagnosis_report.py` | 生成诊断报告 |

---

## 图片 (`figures/`)

### 论文图片
| 文件 | 说明 |
|------|------|
| `fig4_5_qualitative_3views.*` | 图 4.5 - 3 视角定性对比 (png/pdf/yml) |
| `fig4_6_ablation_full_3views.*` | 图 4.6 - 3 视角完整消融 (png/pdf/svg) |

### 方法对比图
| 文件 | 说明 |
|------|------|
| `fig_7methods_3views_cn.*` | 7 方法 3 视角对比 (中文, png/pdf) |
| `fig_7methods_9views_cn.*` | 7 方法 9 视角对比 (中文, png/pdf) |

---

## 论文资料 (`论文/`)

| 文件 | 说明 |
|------|------|
| `chapter4_tables.tex` | 第四章 LaTeX 表格 |

---

## 其他目录

| 目录 | 说明 |
|------|------|
| `docs/` | 文档资料 |
| `codex/` | Codex 相关 |
| `r2gs/` | R²-Gaussian 相关资料 |
| `xgs/` | X-Gaussian 相关资料 |
| `3dgs/` | 3DGS 通用资料 |
| `DNGaussian-main/` | DN-Gaussian 参考代码 |
| `FSGS-main/` | FSGS 参考代码 |
| `CoR-GS-main/` | CoR-GS 参考代码 |

---

## 快速使用

### 运行方法对比实验
```bash
# SPAGS (我们的方法)
./scripts/run_spags_ablation.sh spags foot 3 0

# 其他 3DGS 方法
./scripts/run_spags_ablation.sh xgaussian foot 3 0
./scripts/run_spags_ablation.sh fsgs foot 3 0
./scripts/run_spags_ablation.sh dngaussian foot 3 0
./scripts/run_spags_ablation.sh corgs foot 3 0
```

### 运行消融实验
```bash
./scripts/run_spags_ablation.sh sps foot 3 0
./scripts/run_spags_ablation.sh sps_gar foot 3 0
./scripts/run_spags_ablation.sh sps_adm foot 3 0
./scripts/run_spags_ablation.sh gar_adm foot 3 0
```
