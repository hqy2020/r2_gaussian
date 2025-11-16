# 数据集错配问题记录

**发现时间**: 2025-11-16 22:50
**状态**: ⚠️ 待处理
**影响**: 当前 3 views 验证实验无法与 R² baseline 对比

---

## 问题描述

在执行 CoR-GS Stage 1 验证时,发现使用了错误的数据集,导致测试集大小不匹配,无法与 R² baseline 进行公平对比。

---

## 数据集对比

### ❌ 当前使用的数据集 (错误)

**路径**: `/home/qyhu/Documents/r2_ours/r2_gaussian/data/foot_3views/`

**格式**: NPY 格式
- `proj_train/` - 训练投影
- `proj_test/` - 测试投影
- `init_0_foot_cone.npy` - 初始化点云
- `vol_gt.npy` - Ground truth 体数据
- `meta_data.json` - 元数据

**视角配置**:
- 训练视角: **3 个** ✅ (正确)
- 测试视角: **100 个** ❌ (错误 - 与 baseline 不匹配)

**训练命令**:
```bash
python train.py \
    --source_path /home/qyhu/Documents/r2_ours/r2_gaussian/data/foot_3views \
    --model_path /home/qyhu/Documents/r2_ours/r2_gaussian/output/foot_3views_corgs_stage1 \
    --iterations 10000 \
    --gaussiansN 2 \
    --test_iterations 1000 5000 10000 \
    --enable_corgs
```

**当前训练状态**:
- 进度: ~2000/10000 iterations (20%)
- 日志: `/tmp/foot_3views_corgs.log`
- 输出: `output/foot_3views_corgs_stage1/`
- 后台进程 ID: 330915

---

### ✅ 正确的 Baseline 数据集

**路径**: `/home/qyhu/Documents/r2_ours/r2_gaussian/data/369/foot_50_3views.pickle`

**格式**: Pickle 格式 (单文件)
- 大小: 118 MB
- 创建时间: 2024-10-12

**视角配置**:
- 训练视角: **3 个** ✅
- 测试视角: **50 个** ✅ (与 R² baseline 一致)

**R² Baseline 性能** (在此数据集上):
- PSNR: **28.547**
- SSIM: **0.9008**

---

## 影响分析

### 1. 测试集大小不同

| 项目 | 当前错误数据 | 正确 Baseline |
|------|-------------|--------------|
| 训练视角 | 3 | 3 |
| 测试视角 | **100** | **50** |
| 格式 | NPY | Pickle |

**问题**:
- 测试视角数量不同 (100 vs 50) 会导致评估指标无法直接对比
- 更多的测试视角可能导致 PSNR/SSIM 略有不同
- 无法验证是否真正超越 R² baseline

### 2. 数据格式不同

- **NPY 格式**: 多文件结构,需要读取多个 `.npy` 文件
- **Pickle 格式**: 单文件,可能包含预处理后的数据

**可能的差异**:
- 数据归一化方式
- 相机参数格式
- 点云初始化方法

---

## 当前训练状态

**Iteration 1000 结果**:
- PSNR 2D: 26.83
- SSIM 2D: 0.8098
- 训练速度: ~6.5 it/s

**CoR-GS Disagreement 指标**:

| Iteration | Point Fitness | Point RMSE | PSNR_diff | SSIM_diff |
|-----------|---------------|------------|-----------|-----------|
| 500 | 1.0000 | 0.007787 | 60.59 dB | 0.9986 |
| 1000 | 1.0000 | 0.007926 | 59.29 dB | 0.9994 |
| 1500 | 1.0000 | 0.008489 | 59.01 dB | 0.9992 |
| 2000 | (计算中) | (计算中) | 58.68 dB | 0.9992 |

---

## 解决方案选项

### 选项 A: 停止当前训练,立即重新开始

**优点**:
- 节省计算资源
- 尽快获得可对比的结果

**缺点**:
- 损失已完成的 20% 训练进度
- 当前 Disagreement 指标数据作废

**执行步骤**:
1. Kill 进程 330915
2. 使用 `data/369/foot_50_3views.pickle` 重新训练
3. 预计总时间: 仍需 ~20 分钟

---

### 选项 B: 让当前训练继续,作为参考实验

**优点**:
- 保留完整的 Disagreement 指标时间线
- 可以作为不同测试集规模的对比
- 验证 CoR-GS 实现的正确性

**缺点**:
- 无法与 baseline 直接对比
- 需要额外时间重新训练正确数据集

**执行步骤**:
1. 让进程 330915 继续运行
2. 训练完成后分析 Disagreement 指标
3. 另起新训练使用正确数据集

---

### 选项 C: 并行训练 (推荐)

**优点**:
- 同时获得两组数据
- 可以对比不同测试集规模的影响
- 充分利用 GPU 资源(如果有多卡)

**缺点**:
- 需要双倍 GPU 显存
- 训练速度可能下降

**执行步骤**:
1. 保持进程 330915 继续运行
2. 如果 GPU 显存充足,立即启动正确数据集训练
3. 如果显存不足,等待当前训练完成后再开始

---

## 后续行动计划

### 立即行动 (等待用户决策)

**待确认**:
1. 选择哪个解决方案 (A/B/C)?
2. 是否需要先检查 `foot_50_3views.pickle` 的数据格式?
3. 是否需要修改数据加载代码以支持 Pickle 格式?

### 中期行动 (确定方案后)

**如果选择 A 或 C**:
1. 检查 `data/369/foot_50_3views.pickle` 数据结构
2. 验证数据加载器是否兼容 Pickle 格式
3. 如需修改,更新 `scene/__init__.py` 或相关加载代码
4. 启动正确数据集训练

**如果选择 B**:
1. 等待当前训练完成 (~15 分钟)
2. 分析 Disagreement 指标和结果
3. 生成参考报告
4. 再启动正确数据集训练

### 长期行动

1. **统一数据集格式**: 建议将所有数据集转换为统一格式
2. **文档记录**: 在 `cc-agent/records/datasets.md` 中记录所有可用数据集
3. **自动验证**: 添加训练前的数据集检查脚本

---

## 技术细节

### 当前数据加载流程

**文件**: `scene/__init__.py`

**加载逻辑**:
```python
# 假设从 source_path 读取
train_cameras = readCamerasFromTransforms(source_path, "proj_train")
test_cameras = readCamerasFromTransforms(source_path, "proj_test")
```

### Pickle 格式数据加载

**需要验证**:
1. Pickle 文件的数据结构
2. 是否需要特殊的解析逻辑
3. 相机参数格式是否一致

**可能需要的修改**:
```python
import pickle

if source_path.endswith('.pickle'):
    with open(source_path, 'rb') as f:
        data = pickle.load(f)
    # 解析 data 结构
    train_cameras = parse_cameras(data['train'])
    test_cameras = parse_cameras(data['test'])
```

---

## 相关文件

### 日志和输出
- 训练日志: `/tmp/foot_3views_corgs.log`
- 输出目录: `output/foot_3views_corgs_stage1/`
- TensorBoard: `tensorboard --logdir=output/foot_3views_corgs_stage1`

### 代码位置
- 数据加载: `scene/__init__.py`
- 训练主程序: `train.py`
- 参数配置: `r2_gaussian/arguments/__init__.py`

### 相关文档
- 进度报告: `cc-agent/records/progress_stage1_completion.md`
- 阶段1实现: `cc-agent/code/stage1_implementation_log.md`

---

## 经验教训

### 问题根因

1. **缺少数据集文档**: 没有清晰记录各数据集的用途和规格
2. **缺少预检查**: 训练前未验证数据集是否与 baseline 一致
3. **命名不清晰**: `foot_3views` 和 `foot_50_3views` 容易混淆

### 改进措施

1. **创建数据集清单**:
   ```markdown
   # cc-agent/records/datasets.md

   | 数据集 | 训练视角 | 测试视角 | 格式 | 用途 |
   |--------|---------|---------|------|------|
   | foot_3views | 3 | 100 | NPY | ❌ 错误 |
   | 369/foot_50_3views.pickle | 3 | 50 | Pickle | ✅ Baseline |
   ```

2. **添加训练前检查**:
   ```python
   # 在 train.py 开头添加
   def validate_dataset(source_path, expected_config):
       # 验证数据集配置
       pass
   ```

3. **标准化命名规则**:
   - 格式: `{organ}_{train_views}train_{test_views}test.{format}`
   - 示例: `foot_3train_50test.pickle`

---

**记录时间**: 2025-11-16 22:50
**记录人**: Claude Code
**下次更新**: 用户决策后
