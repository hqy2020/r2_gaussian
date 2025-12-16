# SPAGS 诊断工具套件

本文档介绍 SPAGS 项目的诊断工具，用于分析模型训练过程中的问题。

## 工具概览

| 工具 | 文件 | 功能 |
|------|------|------|
| GAR诊断 | `diagnose_gar.py` | 分析邻近分数分布和密化行为 |
| ADM诊断 | `diagnose_adm.py` | 分析K-Planes特征和调制效果 |
| SPS诊断 | `diagnose_sps.py` | 分析初始化点云的空间分布 |
| 训练分析 | `analyze_training.py` | 对比不同实验的训练曲线 |
| 综合报告 | `generate_diagnosis_report.py` | 整合所有诊断结果 |

所有工具位于 `cc-agent/scripts/diagnosis/` 目录。

---

## 1. GAR诊断工具 (diagnose_gar.py)

### 功能
- 从checkpoint加载高斯点云
- 计算每个高斯的邻近分数
- 分析不同阈值下的密化候选点数
- 生成3D可视化和诊断报告

### 使用方法

```bash
# 基础使用
python cc-agent/scripts/diagnosis/diagnose_gar.py \
    --checkpoint output/xxx/point_cloud/iteration_30000/point_cloud.pickle \
    --output_dir diagnosis/gar/

# 指定参数
python cc-agent/scripts/diagnosis/diagnose_gar.py \
    --checkpoint output/xxx/point_cloud/iteration_30000/point_cloud.pickle \
    --k 5 \
    --threshold 0.05 \
    --output_dir diagnosis/gar/
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | 点云checkpoint路径 (.pickle) |
| `--k` | 5 | K近邻数量 |
| `--threshold` | 0.05 | 邻近分数阈值 |
| `--output_dir` | `diagnosis/gar` | 输出目录 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `gar_proximity_histogram.png` | 邻近分数分布直方图 |
| `gar_3d_visualization.html` | 交互式3D可视化（用颜色标注邻近分数） |
| `gar_diagnosis_report.json` | 诊断统计数据 |

### 诊断报告字段

```json
{
  "summary": {
    "total_gaussians": 4818,
    "proximity_score_mean": 0.076,
    "proximity_score_std": 0.035,
    "proximity_score_range": [0.011, 0.379]
  },
  "statistics": {
    "count": 4818,
    "mean": 0.076,
    "std": 0.035,
    "median": 0.069,
    "percentile_25": 0.052,
    "percentile_75": 0.092,
    "percentile_90": 0.122,
    "percentile_95": 0.143,
    "percentile_99": 0.191
  },
  "threshold_analysis": {
    "threshold_0.05": {"count": 3740, "ratio": 0.776},
    "threshold_0.07": {"count": 2378, "ratio": 0.493},
    "threshold_0.10": {"count": 950, "ratio": 0.197}
  },
  "diagnosis": [
    {
      "level": "WARNING",
      "issue": "阈值设置过低",
      "detail": "阈值 0.05 下有 77.63% 的点会被密化，可能过度密化",
      "suggestion": "提高阈值或使用自适应阈值"
    }
  ]
}
```

### 解读指南

**健康指标**：
- 阈值筛选比例应在 5-15% 范围内
- 邻近分数分布应有明显区分度（std > 0.02）

**常见问题**：
| 问题 | 表现 | 建议 |
|------|------|------|
| 过度密化 | >30%的点被筛选 | 提高阈值 |
| 密化不足 | <3%的点被筛选 | 降低阈值 |
| 无区分度 | std < 0.01 | 检查K值或坐标归一化 |

---

## 2. ADM诊断工具 (diagnose_adm.py)

### 功能
- 从checkpoint加载K-Planes参数
- 可视化三个正交平面的特征图
- 分析offset/confidence分布
- 评估密度调制效果

### 使用方法

```bash
python cc-agent/scripts/diagnosis/diagnose_adm.py \
    --checkpoint output/xxx/chkpnt30000.pth \
    --output_dir diagnosis/adm/
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必填 | 模型checkpoint路径 (.pth) |
| `--output_dir` | `diagnosis/adm` | 输出目录 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `kplanes_xy.png` | XY平面特征热力图 |
| `kplanes_xz.png` | XZ平面特征热力图 |
| `kplanes_yz.png` | YZ平面特征热力图 |
| `adm_diagnosis_report.json` | 诊断统计数据 |

### 解读指南

**健康指标**：
- K-Planes应显示有意义的空间模式，不应是全0或纯噪声
- offset分布应在 [-0.5, 0.5] 范围内有变化
- confidence不应全为0或1

---

## 3. SPS诊断工具 (diagnose_sps.py)

### 功能
- 分析初始化点云的空间分布
- 计算密度分布统计
- 评估分布均匀性
- 支持 Baseline vs SPS 对比

### 使用方法

```bash
# 单文件分析
python cc-agent/scripts/diagnosis/diagnose_sps.py \
    --init_file data/369/init_foot_50_3views.npy \
    --output_dir diagnosis/sps/

# 对比分析
python cc-agent/scripts/diagnosis/diagnose_sps.py \
    --baseline_init data/369/init_foot_50_3views.npy \
    --sps_init data/369-sps/init_foot_50_3views.npy \
    --output_dir diagnosis/sps/
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--init_file` | 无 | 单个初始化文件路径 (.npy) |
| `--baseline_init` | 无 | Baseline初始化文件（用于对比） |
| `--sps_init` | 无 | SPS初始化文件（用于对比） |
| `--output_dir` | `diagnosis/sps` | 输出目录 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `sps_spatial_distribution.png` | 空间分布可视化（三视图投影+直方图） |
| `sps_comparison.png` | Baseline vs SPS 对比图（对比模式） |
| `sps_diagnosis_report.json` | 诊断统计数据 |

### 诊断报告字段

```json
{
  "name": "SPS",
  "spatial_statistics": {
    "count": 10000,
    "bbox_min": [-0.5, -0.5, -0.5],
    "bbox_max": [0.5, 0.5, 0.5],
    "bbox_size": [1.0, 1.0, 1.0],
    "center": [0.0, 0.0, 0.0],
    "std_xyz": [0.25, 0.25, 0.25],
    "nn_distance_mean": 0.02,
    "nn_distance_std": 0.01
  },
  "density_statistics": {
    "mean": 0.5,
    "std": 0.2,
    "min": 0.0,
    "max": 1.0,
    "median": 0.45,
    "percentile_90": 0.8,
    "zero_ratio": 0.05
  },
  "uniformity": {
    "n_bins": 10,
    "total_voxels": 1000,
    "non_empty_voxels": 450,
    "occupancy_ratio": 0.45,
    "cv": 1.2
  },
  "diagnosis": [...]
}
```

### 解读指南

**关键指标**：
| 指标 | 健康范围 | 说明 |
|------|----------|------|
| 占用率 | 30-60% | 太低=点太少，太高=过度采样 |
| 变异系数CV | < 1.5 | 太高=分布不均匀 |
| 零密度比例 | < 10% | 太高=采样质量差 |
| 密度标准差 | > 0.01 | 太低=缺乏区分度 |

---

## 4. 训练曲线分析 (analyze_training.py)

### 功能
- 解析多个实验的评估结果
- 生成PSNR/SSIM对比曲线
- 输出对比表格
- 诊断训练问题

### 使用方法

```bash
python cc-agent/scripts/diagnosis/analyze_training.py \
    --baseline_dir output/2025_12_06_15_52_foot_3views_baseline/ \
    --spags_dir output/2025_12_05_13_58_foot_3views_spags_3k/ \
    --output_dir diagnosis/foot_3views/
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--baseline_dir` | 必填 | Baseline实验目录 |
| `--spags_dir` | 必填 | SPAGS实验目录 |
| `--output_dir` | `diagnosis/comparison` | 输出目录 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `training_curves.png` | PSNR/SSIM训练曲线对比 |
| `training_analysis_report.json` | 详细对比数据 |

---

## 5. 综合诊断报告 (generate_diagnosis_report.py)

### 功能
- 整合GAR、ADM、训练分析的诊断结果
- 生成人类可读的Markdown报告
- 自动汇总问题和建议

### 使用方法

```bash
python cc-agent/scripts/diagnosis/generate_diagnosis_report.py \
    --gar_report diagnosis/gar/gar_diagnosis_report.json \
    --adm_report diagnosis/adm/adm_diagnosis_report.json \
    --training_report diagnosis/comparison/training_analysis_report.json \
    --output diagnosis/full_report.md
```

### 输出示例

```markdown
# SPAGS 综合诊断报告

生成时间: 2025-12-12 16:53:23

## 执行摘要

**状态**: ⚠️ 发现 1 个警告

## GAR 诊断结果
- ⚠️ **阈值设置过低**
  - 阈值 0.05 下有 77.63% 的点会被密化
  - 建议: 提高阈值或使用自适应阈值

## 综合建议
1. 根据诊断结果，调整GAR阈值
2. 运行控制实验验证
```

---

## 典型诊断流程

### 完整诊断流程

```bash
# 1. 运行所有诊断工具
BASELINE_DIR="output/2025_12_06_15_52_foot_3views_baseline"
SPAGS_DIR="output/2025_12_05_13_58_foot_3views_spags_3k"
OUTPUT_DIR="diagnosis/foot_3views"

mkdir -p $OUTPUT_DIR

# GAR诊断
python cc-agent/scripts/diagnosis/diagnose_gar.py \
    --checkpoint $SPAGS_DIR/point_cloud/iteration_30000/point_cloud.pickle \
    --output_dir $OUTPUT_DIR/gar/

# SPS诊断
python cc-agent/scripts/diagnosis/diagnose_sps.py \
    --init_file data/369/init_foot_50_3views.npy \
    --output_dir $OUTPUT_DIR/sps/

# 训练分析
python cc-agent/scripts/diagnosis/analyze_training.py \
    --baseline_dir $BASELINE_DIR \
    --spags_dir $SPAGS_DIR \
    --output_dir $OUTPUT_DIR/

# 生成综合报告
python cc-agent/scripts/diagnosis/generate_diagnosis_report.py \
    --gar_report $OUTPUT_DIR/gar/gar_diagnosis_report.json \
    --training_report $OUTPUT_DIR/training_analysis_report.json \
    --output $OUTPUT_DIR/full_report.md

# 2. 查看报告
cat $OUTPUT_DIR/full_report.md
```

### 快速诊断（仅GAR）

```bash
# 当你只想快速检查GAR是否正常工作时
python cc-agent/scripts/diagnosis/diagnose_gar.py \
    --checkpoint output/xxx/point_cloud/iteration_30000/point_cloud.pickle \
    --output_dir /tmp/gar_check/

# 查看关键指标
cat /tmp/gar_check/gar_diagnosis_report.json | python -m json.tool | grep -A5 "threshold_0.05"
```

---

## 常见问题诊断

### 问题1：PSNR没有提升

**诊断步骤**：
1. 运行 `analyze_training.py` 对比训练曲线
2. 检查是否在相同迭代数下对比
3. 检查各模块是否真的在工作（GAR有新增高斯吗？ADM有调制吗？）

### 问题2：GAR过度密化

**诊断步骤**：
1. 运行 `diagnose_gar.py`
2. 查看 `threshold_analysis` 字段
3. 如果 >30% 的点被筛选，需要提高阈值

### 问题3：训练不稳定

**诊断步骤**：
1. 查看训练曲线是否有剧烈波动
2. 检查密化时机是否合适
3. 检查学习率设置

---

## 相关文档

- [AI辅助科研方法论](./AI_ASSISTED_RESEARCH_METHODOLOGY.md) - 如何使用这些工具进行科研优化
- [GAR几何感知细化](./GAR_几何感知细化.md) - GAR模块技术文档
- [ADM自适应密度调制](./ADM_自适应密度调制.md) - ADM模块技术文档
- [SPS空间先验播种](./SPS_空间先验播种.md) - SPS模块技术文档
