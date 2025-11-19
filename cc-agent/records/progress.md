# R²-Gaussian 项目进度记录

本文件记录项目的所有重要工作节点，包括已完成、进行中和待完成的任务。

---

## 2025-11-19 DropGaussian 集成

### 已完成

**1. DropGaussian 论文分析**
- 论文来源：arXiv:2411.06175 "DropGaussian: Randomly Dropping Gaussians for Avoiding Overfitting in 3D Gaussian Splatting"
- 核心创新点：
  - 随机丢弃 Gaussian primitives 防止过拟合
  - 渐进式调整策略（drop rate 从 0 线性增长到 γ）
  - 零计算开销（仅 5 行代码实现）
- 核心公式：`drop_rate = γ × (iteration / total_iterations)`
- 论文推荐超参数：γ = 0.2

**2. 官方代码研究**
- GitHub 仓库：https://github.com/DCVL-3D/DropGaussian_release
- 核心实现位置：`gaussian_renderer/__init__.py` 第 86-97 行
- 关键实现细节：
  - 在训练时应用随机 mask 过滤 Gaussian
  - 推理时不启用（保持完整模型）
  - 与原始 3DGS 完全兼容

**3. R²-Gaussian 代码集成**
- 修改文件 1：`r2_gaussian/arguments/__init__.py`
  - 新增参数 `use_drop_gaussian`（bool，默认 False）
  - 新增参数 `drop_gamma`（float，默认 0.2）

- 修改文件 2：`r2_gaussian/gaussian/render_query.py`
  - 新增函数参数：`is_train`, `iteration`, `model_params`
  - 实现 DropGaussian 逻辑（第 152-168 行，共 17 行代码）
  - 保持向下兼容（默认不启用）

- 修改文件 3：`train.py`
  - 修改 render 调用，传递训练状态和迭代数
  - 确保 test 和 eval 阶段不启用 DropGaussian

**4. 实验方案设计**
- 目标器官：Foot（3 视角）
- Baseline PSNR：28.4873，SSIM：0.9005
- 实验组设置：
  - 对照组：Baseline（不启用 DropGaussian）
  - 实验组：DropGaussian（γ=0.2）
- 成功标准：PSNR > 28.98, SSIM > 0.905
- 训练配置：30000 iterations

**5. 技术文档生成**
- `cc-agent/code/dropgaussian_implementation.md` - 核心实现分析（152 行）
- `cc-agent/code/integration_report.md` - 集成完成报告（包含修改清单）
- `cc-agent/experiments/dropgaussian_foot3_experiment_plan.md` - 实验方案详细说明

### 进行中
- 暂无

### 待完成

**1. 执行训练实验**


- DropGaussian 训练命令：
  ```bash
  conda activate r2_gaussian_new
  python train.py -s data/369/foot_50_3views.pickle -m ./output/2025_11_19_foot_3views_dropgaussian --iterations 30000 --use_drop_gaussian --drop_gamma 0.2 --eval
  ```

**2. 实验结果分析**
- 对比 PSNR/SSIM 定量指标
- 分析 TensorBoard 训练曲线
- 可视化渲染质量对比
- 生成结果分析报告

**3. （可选）参数调优**
- 如果 γ=0.2 效果显著，尝试其他 gamma 值：
  - γ=0.1（更保守的 dropout）
  - γ=0.3（更激进的 dropout）
- 寻找最优超参数组合

---

### Git 记录
- 当前分支：`drop-gaussian`
- 相关提交：待完成实验后创建 tag（建议：`v1.1-dropgaussian`）

### 关键决策点
- ✅ 采用论文推荐的 γ=0.2 作为初始实验参数
- ✅ 选择 Foot-3 视角作为测试场景（baseline 结果稳定）
- ⏸️ 等待实验结果后决定是否扩展到其他器官

---
