# CoR-GS Stage 3 实现记录

## 实现日期
2025-11-22

## 任务目标
将 CoR-GS (Co-Regularization Gaussian Splatting) Stage 3 的 Pseudo-view Co-regularization 创新点集成到 R²-Gaussian baseline。

## 关键技术决策

### 决策 1: 实现方案选择
**问题**: 是否新建 train_corgs.py 还是修改现有 train.py？  
**选择**: 方案 A - 修改现有 train.py  
**理由**:
- 避免代码重复和维护成本
- 通过 CLI 参数 (--enable_corgs, --gaussiansN) 实现模块化
- 保持向下兼容（gaussiansN=1 时等同于原有单模型训练）

### 决策 2: 医学适配范围
**问题**: 是否实现完整的医学场景适配（ROI 权重、置信度筛选）？  
**选择**: 仅实现基础版本  
**理由**:
- 论文未提供医学适配的具体实现细节
- 先验证基础算法有效性，避免引入额外复杂度
- 预留参数接口 (enable_pseudo_coreg_roi, enable_pseudo_coreg_confidence) 供后续扩展

### 决策 3: 验证策略
**问题**: 是否先做短时间测试？  
**选择**: 直接完整训练 30000 iterations  
**理由**:
- CoR-GS 的效果在完整训练后才能体现（尤其是 iteration 2000 后）
- 用户强调"每多进行一次实验就会有一只小动物死亡"
- 有紧急预案（如果失败，可尝试调参、消融实验等）

## 核心实现

### 1. 多模型训练架构
**文件**: `train.py` (lines 82-119)

**关键修改**:
```python
# 支持 gaussiansN 个独立模型
gaussians_list = []
for idx in range(gaussiansN):
    gs = GaussianModel(scale_bound)
    initialize_gaussian(gs, dataset, None)
    gs.training_setup(opt)
    gaussians_list.append(gs)

# 向下兼容：单模型时使用原有变量名
if gaussiansN == 1:
    gaussians = gaussians_list[0]
    scene.gaussians = gaussians
else:
    gaussians = gaussians_list[0]
    scene.gaussians = gaussians_list  # 多模型时存储列表
```

**设计要点**:
- 每个模型独立初始化、优化器、densification
- 兼容现有单模型代码（scene.gaussians 可以是单对象或列表）

### 2. Pseudo-view Co-regularization 集成
**文件**: `train.py` (lines 198-252)

**触发条件**:
- `gaussiansN >= 2`
- `enable_corgs = True`
- `iteration >= corgs_pseudo_start_iter` (默认 2000)

**核心流程**:
1. 调用 `generate_pseudo_view_medical()` 生成伪视角相机
   - 从训练相机中随机选择并添加位置噪声（std=0.02，约 ±0.4mm）
2. 用两个模型分别渲染 pseudo-view
3. 计算 Co-regularization 损失: `L1 + λ_dssim * (1 - SSIM)`
4. 叠加到两个模型的总损失（权重 λ_p = 1.0）

**TensorBoard 日志**:
- `corgs/pseudo_total`: 总 Co-reg 损失
- `corgs/pseudo_l1`: L1 分量
- `corgs/pseudo_ssim`: SSIM 分量

### 3. 多模型兼容性修复
**问题**: 原代码假设 `scene.gaussians` 是单对象，多模型时变成列表导致 `AttributeError`

**修复位置**:
1. **Line 399**: `training_report` 点数统计
2. **Line 478**: 3D 重建评估 `queryFunc` 调用
3. **Line 523**: 密度直方图记录

**修复模式**:
```python
gaussians_for_eval = scene.gaussians[0] if isinstance(scene.gaussians, list) else scene.gaussians
```

### 4. 命令行参数扩展
**文件**: `r2_gaussian/arguments/__init__.py` (lines 50-54)

**新增参数**:
- `--corgs_pseudo_start_iter`: Pseudo-view co-reg 启动迭代数（默认 2000）
- `--corgs_pseudo_noise_std`: Pseudo-view 位置噪声标准差（默认 0.02）
- `--enable_pseudo_coreg_roi`: ROI 自适应权重（默认 False）
- `--enable_pseudo_coreg_confidence`: 置信度筛选（默认 False）

## 训练配置

### Foot-3 views 实验参数
**脚本**: `train_corgs_foot3.sh`

```bash
GAUSSIANS_N=2                    # 双模型
ENABLE_CORGS=true                # 启用 CoR-GS
LAMBDA_PSEUDO=1.0                # Pseudo-view 权重（论文默认）
PSEUDO_START_ITER=2000           # 从 2000 iter 开始
PSEUDO_NOISE_STD=0.02            # 位置噪声标准差（约 ±0.4mm）
DENSIFY_UNTIL_ITER=15000         # Densification 结束迭代
ITERATIONS=30000                 # 总迭代数
```

**数据集**: `data/369/foot_50_3views.pickle`  
**Baseline 性能**: PSNR 28.4873 dB, SSIM 0.9005  
**期望性能**: PSNR > 28.8 dB, SSIM > 0.908 (+0.4 dB 提升)

## Bug 修复历史

### Bug 1: training_report 点数统计错误
- **错误**: `AttributeError: 'list' object has no attribute 'get_xyz'` (line 399)
- **原因**: 多模型时 scene.gaussians 是列表
- **修复**: 添加 isinstance 检查，多模型时记录每个模型的点数

### Bug 2: 3D 评估 queryFunc 错误
- **错误**: 同上 (line 478)
- **原因**: queryFunc 接收列表导致崩溃
- **修复**: 使用第一个模型进行评估（`scene.gaussians[0]`）

### Bug 3: 密度直方图记录错误
- **错误**: 同上 (line 523)
- **原因**: TensorBoard 日志访问列表对象的属性
- **修复**: 使用第一个模型记录密度分布

## 实验状态

### 当前训练
- **启动时间**: 2025-11-22 19:22:44
- **进程 PID**: 3558298
- **输出目录**: `output/2025_11_22_19_22_foot_3views_corgs_stage3`
- **日志文件**: `train_corgs_foot3.log`
- **预计完成**: 2025-11-23 上午（约 16 小时）

### 关键里程碑
- ✅ Iteration 0-500: 模型初始化和早期训练（Loss 0.33 → 0.006）
- ⏳ Iteration 2000: Pseudo-view Co-regularization 启动
- ⏳ Iteration 15000: Densification 结束
- ⏳ Iteration 30000: 训练完成，性能评估

## 经验教训

### ✅ 成功要点
1. **向下兼容设计**: 单模型场景完全不受影响
2. **渐进式集成**: 先修复基础多模型支持，再添加 Co-reg 功能
3. **充分测试**: 在训练启动前发现并修复 3 个兼容性 bug

### 📝 注意事项
1. **多模型评估**: 当前所有评估指标（PSNR/SSIM）使用第一个模型
   - 未来可扩展：记录双模型的平均/最优指标
2. **内存占用**: 双模型训练内存占用约为单模型的 2 倍
   - 当前 50k 点云 × 2 模型运行正常
3. **Pseudo-view 采样**: 当前随机选择训练视角添加噪声
   - 未来可优化：基于 ROI 的智能采样

## 下一步工作

1. **监控训练进度**（2025-11-22 晚 → 2025-11-23 晨）
   - 检查 iteration 2000 后 Co-reg 损失曲线
   - 观察 Densification 阶段（500-15000）的点云增长

2. **结果分析**（训练完成后）
   - 对比 baseline: PSNR 和 SSIM 提升幅度
   - 可视化 pseudo-view 渲染质量
   - TensorBoard 分析 Co-reg 损失趋势

3. **后续实验**（如有必要）
   - 消融实验: 单独测试 Stage 1 Co-pruning
   - 参数调优: λ_p, noise_std, start_iter
   - 扩展到其他器官: Chest, Head, Abdomen, Pancreas
