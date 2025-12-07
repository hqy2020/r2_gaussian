# MCP 工具使用指南

## serena MCP 使用

### 基本用法

我需要将以下论文创新点移植到 r2_gaussian：

论文创新点：[在此粘贴创新点描述]

请帮我：
1. 使用 find_symbol 定位相关的类和函数
2. 使用 find_referencing_symbols 分析使用位置
3. 找到所有相关的代码文件和行号
4. 分析修改的影响范围

### 具体操作示例

- 搜索关键词：[创新点相关关键词]
- 分析文件：gaussian_model.py, render_query.py, train.py
- 输出格式：文件路径:行号:符号名称:用途

### 查找结果示例

- gaussian_model.py:45:setup_functions: 激活函数配置点
- gaussian_model.py:123:densify_and_split: 密度控制核心逻辑
- render_query.py:67:render: 渲染函数入口
- train.py:234:training_step: 训练主循环

### 影响分析示例

- 高影响：gaussian_model.py（核心模型）
- 中影响：render_query.py（渲染逻辑）
- 低影响：test.py（测试文件）

---

## Neo4j-Memory MCP 使用指南

**Neo4j-Memory 是您的长期记忆系统，用于存储记忆、决策和知识。**

### 开始任何任务之前

**始终先搜索：** 在开始工作之前，使用 `memory_find` 工具查找相关的记忆、决策和知识。

**使用关键词搜索：** 通过关键词、主题或时间范围搜索可能相关的记忆内容。

**审查所有匹配项：** 仔细检查与当前任务匹配的任何记忆、决策或观察记录。

**检查记忆类型：** 注意记忆的类型（knowledge/decision/implementation/architecture），选择最相关的信息。

### 始终保存新的或更新的信息

**立即捕获重要信息：** 当用户表达需求、偏好、决策或重要知识时，立即使用 `memory_store` 存储它。

**明确记忆类型：** 根据内容选择合适的记忆类型：

- **knowledge**：知识、观察、用户画像、行为模式
- **decision**：决策、选择、权衡结果
- **implementation**：实施方法、工作流程、技术方案
- **architecture**：系统架构、组织结构、设计模式

**添加观察记录：** 使用观察记录（observations）记录具体的细节、时间、上下文信息。

**明确标识更新：** 如果某些内容是对现有记忆的更新，先使用 `memory_find` 找到原记忆，然后使用 `memory_modify` 进行更新。

### 工作过程中

**遵循发现的记忆：** 使您的工作与找到的任何记忆保持一致，特别是决策和偏好相关的记忆。

**应用相关知识：** 使用知识类记忆来指导您的建议和操作。

**尊重历史决策：** 如果找到相关的决策记录，理解其背景和依据，不要轻易推翻。

**保持一致性：** 与先前存储的记忆、决策和知识保持一致。

### 最佳实践

**建议前先搜索：** 在提出建议之前，始终使用 `memory_find` 检查是否存在相关记忆。

**多关键词搜索：** 对于复杂任务，尝试多个相关关键词进行搜索，构建完整图景。

**优先考虑具体匹配：** 更具体、更相关的记忆优先于一般性记忆。

**主动识别模式：** 如果您注意到用户行为中的模式，考虑将其存储为 knowledge 类型的记忆。

**及时更新记忆：** 当发现信息变化或需要补充时，及时使用 `memory_modify` 更新。

**数据库管理：** 如需切换数据库，使用 `database_switch` 工具，但通常保持默认数据库即可。

**记忆结构化：** 将长内容拆分为多个观察记录，使记忆更易检索和理解。

**重要提醒：** Neo4j 记忆库是您的长期记忆。持续使用它来提供个性化协助，尊重用户既定的决策、偏好和知识背景。每次对话都是学习和记忆的机会。

---

## 记忆颗粒度标准 ⚡

### 核心理念

**Neo4j Memory 是"知识图谱"，不是"工作日志"**

- ✅ **正确理解**：每个 memory 存储一个**原子化的知识点**
- ❌ **错误理解**：每次工作会话记录一条 memory

### 颗粒度规则

#### 📏 长度标准

| 指标 | 标准 | 说明 |
|------|------|------|
| **单个 observation** | < 150 字（中文约 100 字） | 超过此长度需考虑拆分 |
| **observations 数量** | ≤ 3 条/memory | 每条聚焦单一方面 |
| **memory 总长度** | < 500 字 | 超过需拆分为多个 memory |

#### 🎯 类型纯度

**每个 memory 保持单一类型，不混合！**

- ✅ **推荐**：单一类型（decision 或 implementation 或 issue）
- ❌ **避免**：混合类型（诊断+修复+验证+配置混在一起）

#### 🧩 原子化原则

**一个 memory = 一个独立的知识点**

- ✅ **推荐**：可独立理解和复用的知识单元
- ❌ **避免**：需要上下文才能理解的片段

### 案例对比

#### ❌ 错误示例（颗粒度过大）

```markdown
Memory Name: "FSGS v2修复验证成功"
Memory Type: implementation
Observations:
  1. "根因：FSGS邻近密集化阈值设置错误，proximity_threshold 过小...（270字）...导致过度密集化..."
  2. "修复方案：3行代码修改...测试结果...训练配置...（300字）"
  3. "验证流程：启动训练...参数范围正常...（200字）"
```

**问题诊断**：
- ⚠️ 混合类型：issue（根因）+ decision（修复）+ implementation（验证）
- ⚠️ 单个 observation 过长：270/300/200 字
- ⚠️ 应拆分为 4 个独立 memory

#### ✅ 正确示例（合适颗粒度）

**Memory 1: 根因诊断**
```markdown
Name: "FSGS邻近密集化阈值设置错误"
Type: issue
Observations:
  1. "proximity_threshold 默认值 10.0 过大，导致几乎所有高斯点都被判定为邻近区域，密集化过度。"（52字）
  2. "症状：训练 5000 步后高斯点数量爆炸，VRAM 溢出。"（28字）
```

**Memory 2: 修复决策**
```markdown
Name: "调整 FSGS 邻近阈值为 5.0"
Type: decision
Observations:
  1. "决定将 proximity_threshold 从 10.0 调整为 5.0，限制密集化范围。"（42字）
  2. "原因：阈值过大会导致过度密集化，5.0 是平衡精度和效率的经验值。"（36字）
  3. "后续：可根据具体场景调整，稀疏视角建议 3.0-7.0 范围。"（32字）
```

**Memory 3: 代码实现**
```markdown
Name: "FSGS 邻近阈值参数化修复"
Type: implementation
Observations:
  1. "修改位置：r2_gaussian/innovations/fsgs/proximity_densifier.py:89。默认阈值从 10.0 改为 5.0。"（65字）
  2. "新增命令行参数 --proximity_threshold，支持运行时配置。"（38字）
```

**Memory 4: 训练验证**
```markdown
Name: "FSGS 阈值调整验证结果"
Type: implementation
Observations:
  1. "Foot 3views 测试：高斯点数量稳定在 80k-120k，VRAM 占用 8GB 内。"（41字）
  2. "PSNR 提升 0.3dB，密集化控制在合理范围，训练稳定。"（32字）
```

**关系连接**：
```
Memory1(issue) --[DIAGNOSED_BY]--> Memory2(decision)
Memory2(decision) --[IMPLEMENTS]--> Memory3(implementation)
Memory3(implementation) --[VALIDATES_BY]--> Memory4(implementation)
```

### 存储前检查清单 ✓

在调用 `memory_store` 之前，自问：

- [ ] **是否可拆分？** 这个 observation 包含多个并列要点吗？
- [ ] **长度是否超标？** 单个 observation 超过 150 字吗？
- [ ] **类型是否纯粹？** 混合了诊断、决策、实现等多种类型吗？
- [ ] **是否原子化？** 可以独立理解和复用吗？
- [ ] **是否关联？** 需要用 relations 连接其他相关 memory 吗？

### 避免的反模式 🚫

#### 反模式 1：日志式记录
```markdown
❌ "2025-11-24 初始化方法调研会话总结"
   → 包含待做项、已完成任务、关键输出物、讨论过程...（800字）
```

**改进方法**：拆分为多个独立记忆
- Memory 1: 关键决策（decision）
- Memory 2: 技术调研结论（knowledge）
- Memory 3: 代码修改清单（implementation）

#### 反模式 2：列表式堆叠
```markdown
❌ Observation: "核心改进：(1) sigmoid 调制...(2) TV 正则化...(3) Decoder 学习率...(4) 单模型训练..."
```

**改进方法**：每个改进点创建一个独立的 observation 或 memory

#### 反模式 3：混合类型
```markdown
❌ Type: implementation
   Observation: "发现 bug...修复方案...测试结果...部署配置..."
```

**改进方法**：
- issue memory：bug 诊断
- decision memory：修复方案决策
- implementation memory：代码变更
- knowledge memory：部署配置

### 重构超大记忆指南

**识别条件**（满足任一即需重构）：
- 单个 memory 总长度 > 500 字
- observations 数量 > 4 条
- 混合 3 种以上信息类型

**重构步骤**：
1. 识别独立的知识点（通常 3-6 个）
2. 为每个知识点创建独立 memory
3. 确定 memory 之间的关系（SOLVES, LEADS_TO, IMPLEMENTS, DEPENDS_ON）
4. 使用 `memory_modify` 删除原超大记忆
5. 使用 `memory_store` 创建新的原子化记忆，并设置 relations

