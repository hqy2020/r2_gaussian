# AI辅助科研方法论：从"写代码"到"真正提点"

## 核心洞察

**问题**：Claude Code 在前后端开发中效果显著，但在科研算法优化中往往"写了代码却没提点"。

**原因**：
1. 前后端开发有明确的对错标准（能跑/不能跑），而科研算法的"好坏"是相对的
2. 复现论文代码 ≠ 复现论文效果，因为场景差异、数据特性、超参敏感性
3. 深度学习是黑盒，需要诊断和分析底层原因，而不是盲目堆代码

**解决方案**：诊断驱动的迭代工作流

---

## 诊断驱动的科研工作流

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: 诊断 - "发生了什么？"                          │
│  ├─ 可视化关键中间变量                                  │
│  ├─ 分析训练曲线和日志                                  │
│  └─ 对比启用/禁用模块的差异                             │
├─────────────────────────────────────────────────────────┤
│  Step 2: 假设 - "为什么会这样？"                        │
│  ├─ AI辅助：列出可能的原因                              │
│  ├─ 领域知识：判断哪些假设最可能                        │
│  └─ 优先级排序                                          │
├─────────────────────────────────────────────────────────┤
│  Step 3: 实验 - "验证假设"                              │
│  ├─ 设计最小化验证实验                                  │
│  ├─ 控制变量                                            │
│  └─ 快速迭代（小数据集/短训练）                         │
├─────────────────────────────────────────────────────────┤
│  Step 4: 分析 - "结果说明什么？"                        │
│  ├─ 假设被验证 → 修复问题                               │
│  ├─ 假设被否定 → 下一个假设                             │
│  └─ 发现新问题 → 回到Step 1                             │
└─────────────────────────────────────────────────────────┘
```

---

## 与 Claude Code 的正确交互方式

### 错误方式：直接要求实现

```
❌ "帮我实现FSGS的邻近密化"
❌ "把X²-Gaussian的K-Planes加到我的项目里"
❌ "帮我提升0.5 dB的PSNR"
```

**问题**：
- 代码可以写出来，但效果未必好
- Claude 不知道你的具体场景和数据特性
- 缺乏验证机制，可能写了很多无用代码

### 正确方式：诊断-假设-验证循环

```
✅ "我实现了GAR模块但PSNR没提升，帮我分析可能的原因"
✅ "这是我的邻近分数分布图，77%的点分数>阈值，这正常吗？"
✅ "帮我设计一个实验来验证是否是阈值设置的问题"
✅ "对比这两个训练曲线，你觉得问题出在哪里？"
```

**优势**：
- 基于真实数据做分析
- AI 帮助发现人类可能忽略的模式
- 快速排除不可能的假设

---

## 实战案例：SPAGS 调试过程

### 背景
- 集成了 GAR（FSGS邻近密化）和 ADM（K-Planes调制）
- 预期提升 0.3+ dB，实际**下降** 0.12 dB

### Step 1: 诊断

**创建诊断工具**：
```bash
# GAR诊断
python cc-agent/scripts/diagnosis/diagnose_gar.py \
    --checkpoint output/xxx/point_cloud/iteration_30000/point_cloud.pickle \
    --output_dir diagnosis/gar/

# ADM诊断
python cc-agent/scripts/diagnosis/diagnose_adm.py \
    --checkpoint output/xxx/chkpnt30000.pth \
    --output_dir diagnosis/adm/

# 训练对比
python cc-agent/scripts/diagnosis/analyze_training.py \
    --baseline_dir output/baseline/ \
    --spags_dir output/spags/ \
    --output_dir diagnosis/comparison/
```

**关键发现**：
| 诊断项 | 期望值 | 实际值 | 问题 |
|--------|--------|--------|------|
| GAR阈值筛选比例 | 5-15% | **77.63%** | 严重过度密化 |
| Baseline训练迭代 | 30000 | **10000** | 对比不公平 |
| 相同迭代(10000)对比 | SPAGS > Baseline | SPAGS **< 0.12 dB** | 方法有问题 |

### Step 2: 假设生成

**与Claude的对话**：
```
Q: "GAR诊断显示77%的点被标记为需要密化，这正常吗？"

A: "这不正常。邻近密化应该只针对边界区域的稀疏点，通常只有5-15%。
    77%意味着几乎所有点都被密化，这会：
    1. 大幅增加计算量
    2. 引入大量初始化不良的新高斯
    3. 可能破坏已经收敛的区域

    可能原因：
    1. 阈值设置太低
    2. 坐标归一化不一致
    3. K值不合适"
```

**假设排序**：
1. **阈值/归一化问题** (最可能) - 配置写5.0，但代码归一化到[-1,1]³
2. 密化时机不对
3. 新高斯初始化策略问题

### Step 3: 验证实验

**设计控制实验**：
```bash
# 实验1：提高阈值
./run_spags_ablation.sh spags foot 3 0 --proximity_threshold 0.15

# 实验2：禁用GAR，只用ADM
./run_spags_ablation.sh sps_adm foot 3 0

# 实验3：禁用ADM，只用GAR
./run_spags_ablation.sh sps_gar foot 3 0
```

### Step 4: 分析与迭代

根据控制实验结果，确定：
- 哪个模块是问题根源
- 正确的超参范围
- 是否需要重新设计

---

## AI辅助科研的角色分工

| 场景 | AI 的角色 | 人的角色 |
|------|-----------|----------|
| **问题诊断** | 列出所有可能原因 | 判断哪个最可能 |
| **工具开发** | 快速编写诊断脚本 | 定义需要诊断什么 |
| **实验设计** | 建议控制变量方案 | 确定优先级和资源分配 |
| **代码审查** | 找潜在bug和不一致 | 确认是否是真正的bug |
| **结果分析** | 解读数据趋势和异常 | 提出新假设 |
| **文献调研** | 总结相关工作的做法 | 判断适用性 |

---

## 诊断工具套件

已创建的诊断工具位于 `cc-agent/scripts/diagnosis/`：

### 1. GAR诊断 (`diagnose_gar.py`)
```bash
python diagnose_gar.py \
    --checkpoint <点云文件.pickle> \
    --k 5 \
    --threshold 0.05 \
    --output_dir diagnosis/gar/
```

**输出**：
- `gar_proximity_histogram.png`: 邻近分数分布图
- `gar_3d_visualization.html`: 交互式3D可视化
- `gar_diagnosis_report.json`: 诊断统计

**关键指标**：
- 邻近分数分布是否有区分度
- 阈值筛选比例是否合理（期望5-15%）

### 2. ADM诊断 (`diagnose_adm.py`)
```bash
python diagnose_adm.py \
    --checkpoint <模型检查点.pth> \
    --output_dir diagnosis/adm/
```

**输出**：
- `kplanes_*.png`: K-Planes平面热力图
- `adm_diagnosis_report.json`: 特征统计

**关键指标**：
- K-Planes是否学到了有意义的空间模式
- 特征值分布是否合理

### 3. SPS诊断 (`diagnose_sps.py`)
```bash
# 单文件分析
python diagnose_sps.py \
    --init_file data/369/init_foot_50_3views.npy \
    --output_dir diagnosis/sps/

# 对比分析
python diagnose_sps.py \
    --baseline_init data/369/init_foot_50_3views.npy \
    --sps_init data/density-369/init_foot_50_3views.npy \
    --output_dir diagnosis/sps/
```

**输出**：
- `sps_spatial_distribution.png`: 空间分布可视化
- `sps_comparison.png`: Baseline vs SPS对比
- `sps_diagnosis_report.json`: 诊断统计

**关键指标**：
- 点云空间分布是否均匀
- 密度分布是否有区分度
- 占用率和变异系数

### 4. 训练曲线分析 (`analyze_training.py`)
```bash
python analyze_training.py \
    --baseline_dir output/baseline_exp/ \
    --spags_dir output/spags_exp/ \
    --output_dir diagnosis/comparison/
```

**输出**：
- `training_curves.png`: 多指标曲线对比
- `training_analysis_report.json`: 详细对比

**关键指标**：
- 相同迭代数下的公平对比
- 收敛速度差异
- 最终性能差距

### 5. 综合报告 (`generate_diagnosis_report.py`)
```bash
python generate_diagnosis_report.py \
    --gar_report diagnosis/gar/gar_diagnosis_report.json \
    --adm_report diagnosis/adm/adm_diagnosis_report.json \
    --training_report diagnosis/comparison/training_analysis_report.json \
    --output diagnosis/full_report.md
```

---

## 检查清单：每次实验后

```markdown
## 实验名称：xxx
## 日期：yyyy-mm-dd

### 1. 基本信息
- [ ] 改动了什么？
- [ ] 预期效果是什么？
- [ ] 实际PSNR是多少？

### 2. 诊断检查
- [ ] GAR执行了吗？新增了多少高斯？
- [ ] ADM的调制范围是多少？
- [ ] 训练曲线是否正常？
- [ ] 是否与baseline做了公平对比（相同迭代数）？

### 3. 分析
- [ ] 为什么结果是这样？
- [ ] 下一步应该尝试什么？
```

---

## 常见陷阱与解决方案

### 陷阱1：不公平对比
**问题**：SPAGS训练30000次 vs Baseline训练10000次

**解决**：
- 始终在相同迭代数下对比
- 或者对比最优迭代的结果

### 陷阱2：配置不一致
**问题**：配置文件写的阈值和代码实际用的不一样（归一化问题）

**解决**：
- 在代码中打印实际使用的参数值
- 使用诊断工具验证中间结果

### 陷阱3：只看最终结果
**问题**：只关注最终PSNR，不看中间过程

**解决**：
- 监控训练曲线
- 保存中间checkpoint用于诊断
- 分析各模块的独立贡献

### 陷阱4：过早放弃
**问题**：第一次没效果就认为方法不行

**解决**：
- 先诊断是"方法不行"还是"实现有bug"
- 很多时候是超参不对，不是方法本身的问题

---

## 总结：AI辅助科研的正确姿势

1. **不要期望AI直接给出提点的代码**，而是用AI来：
   - 快速编写诊断工具
   - 分析可能的原因
   - 设计验证实验

2. **建立诊断驱动的工作流**：
   - 每次实验后先诊断，再改进
   - 用数据说话，不凭感觉调参

3. **保持迭代思维**：
   - 科研没有一步到位
   - 每次实验都是学习机会
   - 记录失败的尝试，它们同样有价值

4. **AI是放大器，不是替代品**：
   - 领域知识仍然是核心
   - AI帮助加速迭代，但方向需要人来把握
