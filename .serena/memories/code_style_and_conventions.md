# R²-Gaussian 代码风格和约定

## Python 代码风格

### 命名约定
- **模块名**：小写下划线分隔 (snake_case)
  - 例如：`gaussian_model.py`, `loss_utils.py`
- **类名**：大驼峰命名 (PascalCase)
  - 例如：`GaussianModel`, `DepthEstimator`
- **函数名**：小写下划线分隔 (snake_case)
  - 例如：`render_query()`, `compute_loss()`
- **变量名**：小写下划线分隔 (snake_case)
  - 例如：`depth_loss_weight`, `num_points`
- **常量**：大写下划线分隔
  - 例如：`MAX_ITERATIONS`, `DEFAULT_DENSITY_THRESH`

### 文件组织
1. **导入顺序**
   ```python
   # 标准库
   import os
   import sys
   
   # 第三方库
   import torch
   import numpy as np
   
   # 本地模块
   from r2_gaussian.utils import loss_utils
   ```

2. **模块结构**
   - 核心逻辑放在 `r2_gaussian/` 模块下
   - 工具函数放在 `r2_gaussian/utils/` 下
   - 新增功能优先作为工具模块添加到 `utils/`

## 训练命名规范

### 实验命名格式
**格式**：`yyyy_MM_dd_organ_{{nums}}views_{{technique}}`

**示例**：
- `2024_11_17_foot_3views_fsgs`
- `2024_11_18_chest_6views_sss`
- `2024_11_19_abdomen_9views_corgs`

### 输出目录命名
- 简化格式：`{organ}dd_{views}_{MMDD}`
- 示例：`footdd_3_1117`, `chestdd_6_1118`

## Git 约定

### 提交信息格式
```
type: 简短描述（50字符内）

详细描述（可选，72字符换行）

相关 issue 或 PR（可选）
```

**Type 类型**：
- `feat`: 新功能
- `fix`: Bug 修复
- `refactor`: 重构
- `docs`: 文档
- `test`: 测试
- `chore`: 构建/工具

**示例**：
```
feat: 集成 SSS (Student Splatting and Scooping) 完整实现

- 添加 sss_helpers.py 和 sss_utils.py
- 在 train.py 中集成 SSS 训练逻辑
- 更新配置参数支持
```

### 分支命名
- 主分支：`main` 或 `fsgs-hqy`
- 功能分支：`feature/功能名`
- 修复分支：`fix/问题描述`

## 参数配置约定

### 命令行参数
- 使用 `--` 前缀的长参数名
- 布尔参数使用 `store_true` 动作
- 提供清晰的 `help` 说明

**示例**：
```python
parser.add_argument('--enable_depth', action='store_true', 
                    help='启用深度约束')
parser.add_argument('--depth_loss_weight', type=float, default=0.05,
                    help='深度损失权重')
```

### 配置文件
- 使用 YAML 格式 (`cfg_args.yml`)
- 保存到输出目录中
- 包含所有训练参数以确保可复现性

## 代码注释约定

### 文档字符串
```python
def compute_depth_loss(depth_pred, depth_gt, loss_type='pearson'):
    """
    计算深度损失
    
    Args:
        depth_pred: 预测深度图 (B, H, W)
        depth_gt: 真实深度图 (B, H, W)
        loss_type: 损失类型 ('pearson', 'l1', 'l2')
    
    Returns:
        torch.Tensor: 深度损失标量值
    """
    pass
```

### 行内注释
- 中文注释优先（根据 CLAUDE.md 要求）
- 解释"为什么"而不是"是什么"
- 复杂算法需要详细注释

**示例**：
```python
# 使用 Pearson 相关系数衡量深度一致性
# 相比 L1/L2，能更好地处理尺度变化
if loss_type == 'pearson':
    loss = pearson_correlation_loss(depth_pred, depth_gt)
```

## 新功能集成约定

### 向下兼容原则
```python
# 使用 try-except 确保新功能不破坏旧代码
try:
    if args.enable_new_feature:
        result = apply_new_feature(data)
except AttributeError:
    # 旧版本没有 enable_new_feature 参数
    result = data
```

### 默认参数设置
- 新功能默认**关闭**
- 需要显式启用（`--enable_xxx`）
- 向下兼容已有实验配置

## TensorBoard 日志约定

### 日志命名
```python
# 层级化命名
tb_writer.add_scalar("loss/total", total_loss, iteration)
tb_writer.add_scalar("loss/depth", depth_loss, iteration)
tb_writer.add_scalar("metrics/psnr_2d", psnr_2d, iteration)
tb_writer.add_scalar("metrics/psnr_3d", psnr_3d, iteration)
tb_writer.add_scalar("gaussian/opacity_mean", opacity.mean(), iteration)
```

### 可视化频率
- **Scalars**：每个 iteration 记录
- **Images**：每 100-500 iterations
- **Histograms**：每 1000 iterations

## 测试约定

### 单元测试命名
- 文件名：`test_*.py`
- 函数名：`test_功能描述()`

**示例**：
```python
# test_depth.py
def test_depth_loss_computation():
    """测试深度损失计算"""
    pass

def test_pearson_correlation():
    """测试 Pearson 相关系数"""
    pass
```

## AI 生成文档约定

**重要**：所有 AI 生成的文档必须在 `cc-agent/` 对应文件夹下

```
cc-agent/
├── 3dgs_expert/
│   └── innovation_analysis.md        # 创新点分析
├── medical_expert/
│   └── medical_feasibility_report.md # 医学可行性评估
├── code/
│   └── code_review.md                # 代码审查文档
└── experiments/
    └── experiment_plan.md            # 实验计划
```

**禁止**在项目根目录或其他位置创建 AI 文档。