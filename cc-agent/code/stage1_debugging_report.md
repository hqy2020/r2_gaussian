# CoR-GS 阶段 1 调试报告

**日期**: 2025-11-16
**状态**: 🔧 调试中 - 已定位问题，待最终解决
**实验者**: Claude Code

---

## 📋 执行摘要

CoR-GS 阶段 1（Disagreement Metrics 日志记录）代码已完成实现（380 行），但在运行时遇到**指标未记录**问题。经过深入调试，已定位到具体卡点：`log_corgs_metrics()` 函数在渲染或 KNN 计算时挂起或耗时过长。

**关键发现**：
- ✅ 代码无语法错误，可正常编译
- ✅ TensorBoard 正常工作（有训练指标）
- ✅ 双模型成功初始化（GsDict 包含 gs0 和 gs1）
- ✅ 所有参数正确传递到 `training_report()`
- ❌ `log_corgs_metrics()` 调用后无返回，疑似计算过慢或卡死

---

## 🔍 问题定位过程

### 阶段 1: 环境问题排查
**问题**: 训练运行但无 CoR-GS 指标输出
**尝试**:
1. 检查 Python 环境 → 确认使用 `r2_gaussian_new`
2. 清理 Python 缓存 → 发现缓存导致旧代码执行
3. 验证 TensorBoard → 确认 tb_writer 存在且正常工作

**结论**: 环境配置正确，问题在代码逻辑

---

### 阶段 2: 代码执行路径追踪

添加了 15 个 DEBUG 检查点，追踪代码执行路径：

```python
# train.py:1002
[DEBUG-REPORT] Iter 500: gaussiansN=2, GsDict=True, tb_writer=True ✅

# train.py:1019
[DEBUG-CORGS-1] Iter 500: enable_corgs_logging=True ✅

# train.py:1024
[DEBUG-CORGS-2] Iter 500: Entering CoR-GS logging block ✅

# train.py:1027
[DEBUG-CORGS-3] Import successful ✅

# train.py:1032
[DEBUG-CORGS-4] gs2=True, pipe=True ✅

# train.py:1038
[DEBUG-CORGS-5] test_cameras length=100 ✅

# train.py:1043 - 调用 log_corgs_metrics()
(之后无任何输出) ❌
```

**日志文件**: `/tmp/corgs_final.log`（最新）

---

### 阶段 3: 根本原因定位

**卡点**: `log_corgs_metrics()` 函数内部
**位置**: `r2_gaussian/utils/corgs_metrics.py:218`

**可能的慢速/卡死点**:
1. **KNN 计算** (line 253): `compute_point_disagreement(xyz_1, xyz_2, threshold)`
   - 100k × 100k 点的距离矩阵计算
   - 即使批处理，仍需 10 次 10k×100k 的 `torch.cdist()` 调用

2. **渲染计算** (line 259-260): `render(test_camera, gaussians_1, pipe, background)`
   - 两次完整的 Gaussian Splatting 渲染
   - 每次渲染 100k 个 Gaussians

3. **PSNR 计算** (line 265): 理论上很快，不太可能是瓶颈

**证据**:
- 训练速度从 13.74 it/s 突降至 2.86 it/s（iteration 500 时）
- 与测试评估的时间模式一致（渲染 100 个测试相机）
- 无错误信息，说明不是崩溃，而是正常计算但太慢

---

## 📁 已修改文件清单

### 1. `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/arguments/__init__.py`
**修改**: Line ~285, 添加 5 个 CoR-GS 参数
```python
self.enable_corgs = False
self.corgs_tau = 0.3
self.corgs_coprune_freq = 500
self.corgs_pseudo_weight = 1.0
self.corgs_log_freq = 500
```

---

### 2. `/home/qyhu/Documents/r2_ours/r2_gaussian/r2_gaussian/utils/corgs_metrics.py` (新文件)
**大小**: 276 行
**功能**: 实现 Point Disagreement 和 Rendering Disagreement 计算

**核心函数**:
```python
def compute_point_disagreement(
    gaussians_1_xyz: torch.Tensor,  # [N1, 3]
    gaussians_2_xyz: torch.Tensor,  # [N2, 3]
    threshold: float = 0.3,
    max_points: int = 100000
) -> Tuple[float, float]:
    """
    使用 PyTorch KNN (torch.cdist) 计算 Fitness 和 RMSE
    批处理: 10k × 100k 避免显存爆炸
    """
    # ... 实现见文件

def log_corgs_metrics(...) -> dict:
    """
    主入口函数，计算所有 CoR-GS 指标
    """
    # 1. Point Disagreement (KNN)
    # 2. Rendering Disagreement (渲染 + PSNR)
    # 返回: {'point_fitness', 'point_rmse', 'render_psnr_diff'}
```

---

### 3. `/home/qyhu/Documents/r2_ours/r2_gaussian/train.py`
**修改位置**:

#### (1) Line 291-292: 定义 background 变量
```python
background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
```

#### (2) Line 961-981: 传递 CoR-GS 参数
```python
training_report(
    ...,
    GsDict=GsDict,      # 新增
    pipe=pipe,          # 新增
    background=background,  # 新增
)
```

#### (3) Line 984-996: 修改函数签名
```python
def training_report(
    ...,
    GsDict=None,      # 新增
    pipe=None,        # 新增
    background=None,  # 新增
):
```

#### (4) Line 1002-1003: 添加入口 DEBUG
```python
if iteration % 500 == 0:
    print(f"[DEBUG-REPORT] Iter {iteration}: gaussiansN={gaussiansN}, GsDict={GsDict is not None}, tb_writer={tb_writer is not None}", flush=True)
```

#### (5) Line 1017-1062: CoR-GS 日志记录逻辑（带 DEBUG）
```python
enable_corgs_logging = gaussiansN >= 2 and GsDict is not None
if iteration % 500 == 0:
    print(f"[DEBUG-CORGS-1] Iter {iteration}: enable_corgs_logging={enable_corgs_logging}", flush=True)

if enable_corgs_logging:
    log_freq = 500
    if iteration % log_freq == 0:
        print(f"[DEBUG-CORGS-2] Iter {iteration}: Entering CoR-GS logging block", flush=True)
        try:
            from r2_gaussian.utils.corgs_metrics import log_corgs_metrics
            print(f"[DEBUG-CORGS-3] Import successful", flush=True)

            gaussians_1 = GsDict.get("gs0", scene.gaussians)
            gaussians_2 = GsDict.get("gs1", None)
            print(f"[DEBUG-CORGS-4] gs2={gaussians_2 is not None}, pipe={pipe is not None}", flush=True)

            if gaussians_2 is not None and pipe is not None:
                threshold = 0.3
                test_cameras = scene.getTestCameras()
                print(f"[DEBUG-CORGS-5] test_cameras length={len(test_cameras)}", flush=True)

                if len(test_cameras) > 0:
                    test_camera = test_cameras[0]
                    bg_color = background if background is not None else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

                    # ⚠️ 这里调用后卡住
                    corgs_metrics = log_corgs_metrics(
                        gaussians_1, gaussians_2,
                        test_camera, pipe, bg_color,
                        threshold=threshold
                    )

                    for metric_name, metric_value in corgs_metrics.items():
                        tb_writer.add_scalar(f"corgs/{metric_name}", metric_value, iteration)

                    print(f"[CoR-GS Metrics @ Iter {iteration}] "
                          f"Fitness={corgs_metrics['point_fitness']:.4f}, "
                          f"RMSE={corgs_metrics['point_rmse']:.6f}, "
                          f"PSNR_diff={corgs_metrics['render_psnr_diff']:.2f} dB")

        except ImportError as e:
            print(f"⚠️ CoR-GS metrics module not available: {e}")
        except Exception as e:
            print(f"⚠️ Error computing CoR-GS metrics: {e}")
```

---

## 🧪 测试命令

### 当前使用的测试命令
```bash
# 环境: r2_gaussian_new
# 数据: foot cone 50 views

cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 清理缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 快速调试运行（600 iterations）
/home/qyhu/anaconda3/envs/r2_gaussian_new/bin/python train.py \
    --source_path data/cone_ntrain_50_angle_360/0_foot_cone \
    --model_path output/foot_corgs_final \
    --iterations 600 \
    --gaussiansN 2 \
    --test_iterations 500 \
    2>&1 | tee /tmp/corgs_final.log
```

### 日志检查命令
```bash
# 查看 DEBUG 输出
grep "DEBUG-CORGS" /tmp/corgs_final.log

# 查看完整执行流程
grep -E "(DEBUG-REPORT|DEBUG-CORGS|CoR-GS Metrics)" /tmp/corgs_final.log

# 检查 TensorBoard 指标
python -c "
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('output/foot_corgs_final')
ea.Reload()
print('All tags:', ea.Tags()['scalars'])
print('CoR-GS tags:', [t for t in ea.Tags()['scalars'] if 'corgs' in t.lower()])
"
```

---

## 🔧 待解决问题

### 问题: `log_corgs_metrics()` 执行过慢或卡死

**下一步调试方案**:

1. **在 `corgs_metrics.py` 内部添加 DEBUG 输出**
   ```python
   # 在 log_corgs_metrics() 函数中添加
   def log_corgs_metrics(...):
       print("[DEBUG-CORGS-6] Starting log_corgs_metrics", flush=True)

       print("[DEBUG-CORGS-7] Getting xyz", flush=True)
       xyz_1 = gaussians_1.get_xyz.detach()
       xyz_2 = gaussians_2.get_xyz.detach()

       print(f"[DEBUG-CORGS-8] Shapes: {xyz_1.shape}, {xyz_2.shape}", flush=True)

       print("[DEBUG-CORGS-9] Computing point disagreement", flush=True)
       fitness, rmse = compute_point_disagreement(xyz_1, xyz_2, threshold)
       print(f"[DEBUG-CORGS-10] Point metrics: {fitness:.4f}, {rmse:.6f}", flush=True)

       print("[DEBUG-CORGS-11] Rendering model 1", flush=True)
       render_pkg_1 = render(test_camera, gaussians_1, pipe, background)

       print("[DEBUG-CORGS-12] Rendering model 2", flush=True)
       render_pkg_2 = render(test_camera, gaussians_2, pipe, background)

       print("[DEBUG-CORGS-13] Computing PSNR", flush=True)
       psnr_diff = compute_rendering_disagreement(...)

       print(f"[DEBUG-CORGS-14] Done: {psnr_diff:.2f}", flush=True)
       return metrics
   ```

2. **优化方案（如果确认太慢）**:
   - **减少采样点数**: `max_points=10000` (当前 100000)
   - **跳过渲染**: 先只测试 Point Disagreement
   - **降低日志频率**: `log_freq=1000` (当前 500)
   - **使用训练相机**: 避免 100 个测试相机的开销

3. **临时 Workaround**:
   ```python
   # 在 train.py 中暂时跳过耗时计算
   if iteration % log_freq == 0:
       # 临时：只记录虚拟数据验证流程
       metrics = {
           'point_fitness': 0.5,
           'point_rmse': 0.1,
           'render_psnr_diff': 25.0
       }
       for metric_name, metric_value in metrics.items():
           tb_writer.add_scalar(f"corgs/{metric_name}", metric_value, iteration)
   ```

---

## 📊 基准数据

**数据集**: `data/cone_ntrain_50_angle_360/0_foot_cone`
- 训练视图: 50
- 测试视图: 100
- 初始 Gaussian 点数: 50,000 (每个模型)
- 训练后点数: ~100,000 (每个模型)

**R² Baseline 性能** (foot 3 views):
- PSNR: 28.547
- SSIM: 0.9008

**目标**: 使用 CoR-GS 超越上述基准

---

## 📝 关键配置参数

### CoR-GS 参数
```python
--gaussiansN 2              # 双模型
--corgs_tau 0.3             # KNN 阈值（CT 场景调整）
--corgs_log_freq 500        # 日志频率
--corgs_coprune_freq 500    # 剪枝频率（阶段 2）
--corgs_pseudo_weight 1.0   # 伪视图权重（阶段 3）
```

### 训练参数
```python
--source_path data/cone_ntrain_50_angle_360/0_foot_cone
--model_path output/[实验名称]
--iterations 10000          # 完整训练
--test_iterations 1000 5000 10000
```

---

## 🎯 后续实验计划

### 阶段 1（当前）: Disagreement Metrics Logging
**状态**: 🔧 调试中
**任务**:
1. ✅ 实现 Point Disagreement (KNN)
2. ✅ 实现 Rendering Disagreement (PSNR)
3. ✅ 集成到 train.py
4. ⏳ 验证指标正确记录
5. ⏳ 生成相关性分析图

**验证指标**:
- TensorBoard 中出现 `corgs/point_fitness`, `corgs/point_rmse`, `corgs/render_psnr_diff`
- Fitness 理论范围: [0, 1]，期望 >0.7（高一致性）
- RMSE 理论范围: [0, ∞]，期望 <0.2（低误差）
- PSNR_diff 理论范围: [0, ∞]，期望 >25 dB（相似）

---

### 阶段 2: Co-Pruning Implementation
**状态**: ⏸️ 等待阶段 1 完成
**任务**:
1. 实现 KNN-based Co-Pruning
2. 集成到密化循环（每 500 iterations）
3. 验证剪枝效果

**关键代码位置**: `train.py` density control block

---

### 阶段 3: Pseudo-View Co-Regularization
**状态**: ⏸️ 等待阶段 2 完成
**任务**:
1. CT 角度插值策略
2. 伪视图渲染
3. Co-regularization loss

---

### 阶段 4: Full Integration & Evaluation
**状态**: ⏸️ 等待阶段 3 完成
**任务**:
1. 完整系统测试
2. Ablation 实验
3. 性能对比 vs R² baseline

---

## 🚀 快速恢复指南

如果清除对话后需要继续，请执行：

### 1. 检查代码状态
```bash
cd /home/qyhu/Documents/r2_ours/r2_gaussian

# 查看已修改的文件
git status

# 查看 CoR-GS 模块是否存在
ls -lh r2_gaussian/utils/corgs_metrics.py
```

### 2. 查看最新调试日志
```bash
tail -100 /tmp/corgs_final.log | grep "DEBUG"
```

### 3. 继续调试
根据 "待解决问题" 部分的方案继续：
- 如果看到 `DEBUG-CORGS-5` 但无后续输出 → 在 `corgs_metrics.py` 添加内部 DEBUG
- 如果看到 `DEBUG-CORGS-14` → 说明计算完成，检查 TensorBoard
- 如果无任何 DEBUG → 清理缓存重新运行

---

## 📚 相关文档

- **论文**: `cc-agent/论文/reading/corgs/`
- **实现日志**: `cc-agent/code/stage1_implementation_log.md` (430 行)
- **创新点分析**: `cc-agent/3dgs_expert/corgs_innovation_analysis.md`
- **医学评估**: `cc-agent/medical_expert/corgs_medical_feasibility_report.md`

---

## ⚠️ 已知限制

1. **PyTorch 版本**: 1.12.1（部分新特性不可用，但不影响核心功能）
2. **显存占用**: 100k×100k KNN 即使批处理也需 ~4GB
3. **计算时间**: 每 500 iterations 的指标计算可能需 10-30 秒
4. **TensorBoard 延迟**: 指标可能不会立即显示，需刷新

---

**最后更新**: 2025-11-16 21:00
**下次调试**: 在 `corgs_metrics.py:248` 添加 DEBUG-CORGS-6 到 DEBUG-CORGS-14
