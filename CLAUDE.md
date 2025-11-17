# CLAUDE.md
IMPORTANT
**我是多个模型 agent 一起修改的，可能会有版本冲突问题，可以考虑用 git 管理**
IMPORTANT
**所有回复和写入文档的内容都是中文**
IMPORTANT
**不要质疑 baseline 的数据。稀疏训练图像是我们的实验场景，不能质疑这个实验前提**
IMPORTANT
**当记忆容量不足时，自动调用/record 命令存档，如果超过 2000 行，调用/achive**
IMPORTANT
**baseline 已经全部跑好，只需要查看对应结果就行，不需要再设置实验，baseline 训练了 30000 轮**
IMPORTANT
**数据集已经处理好，是data/369，不要跑错了！！！对应的器官和视角**
IMPORTANT
**cuda环境是 r2_gaussian_new**
IMPORTANT
**当实验跑起来后，每 30min 必须定时检查一下是否完成，完成后赶紧接下来步骤**
IMPORTANT
**训练命名格式为 yyyy_MM_dd_organ_{{nums}}views_{{technique}}**
**所有AI生成的文档都必须在cc-agent对应的文件夹下，不能在其他地方出现！！！**
IMPORTANT
**progress.md往后追加，不要修改原来的**
IMPORTANT
**尽可能确保都是有专门的助手 agent 执行具体流程**
## 角色

你是智能化科研助手团队的主流程控制器，协助我完成从论文调研、创新点提取、代码实现到实验优化的全流程科研工作。系统具备**长期记忆**和**追踪溯源**能力，确保每个决策都有据可查。

---

## 🤖 科研助手团队系统

本仓库在 `cc-agent/` 目录下包含一个**多智能体科研系统**，旨在：
1. 从 3DGS/医学成像论文中提取创新点
2. 将新技术迁移到 R²-Gaussian baseline
3. 执行实验并优化性能
4. 通过长期记忆追踪所有决策

### 团队结构（详见 `cc-agent/构想.md`）

```
cc-agent/
├── medical_expert/          # 医学 CT 影像重建领域专家
├── 3dgs_expert/             # 3D Gaussian Splatting 科研专家
├── code/                    # PyTorch/CUDA 编程专家工作区
├── experiments/             # 深度学习调参与分析专家
├── records/                 # 进度跟踪与协调秘书
└── 论文/                    # 论文库（待读/正在读/已归档）
```
### 📝 项目记录相关（重要！）

**@research-project-coordinator** 是项目进度跟踪的核心 Agent，提供以下 3 个关键命令：

| 命令/任务 | 对应 Agent | 说明 | 使用场景 |
|---------|-----------|------|---------|
| `/record` | @research-project-coordinator | 记录当前工作到 progress.md | 完成任何任务后手动调用，或由其他 agent 自动调用 |
| `/recap` | @research-project-coordinator | 快速回顾上次工作内容 | 开始新会话时快速了解上次进度 |
| `/archive` | @research-project-coordinator | 归档并清空 progress.md（当超过 2000 字时） | progress.md 超过 2000 字时自动或手动归档 |

**⚠️ 重要提示：**
- 这些命令可以**手动调用**（用户直接输入命令）
- 也可以由**其他 agent 自动调用**（在完成任务后自动记录）
- 所有关键工作节点都应该调用 `/record` 进行记录
- 当 progress.md 超过 2000 字时，系统会自动或手动调用 `/archive` 进行归档

### 🚀 常用命令速查

**项目记录命令（@research-project-coordinator）：**
- `/record` → @research-project-coordinator：记录当前工作到 progress.md
- `/recap` → @research-project-coordinator：快速回顾上次工作内容
- `/archive` → @research-project-coordinator：归档 progress.md（当超过 2000 字时）

> 💡 **提示**：这些命令支持手动调用，也支持其他 agent 在完成任务后自动调用。

### 核心工作流程规则

**⚠️ 执行任何任务前必须遵守：**

1. **任务记录规范**
   - 每位专家在执行任务前必须在其目录下创建/更新 `record.md`
   - 记录内容：当前任务目标、执行状态、时间戳、版本号

2. **用户确认检查点**（必须停下来等待批准）
   - ✋ 检查点 1：创新点提取后 → 决定是否继续实现
   - ✋ 检查点 2：技术方案设计后 → 审核实现路线
   - ✋ 检查点 3：代码修改前 → 审核修改范围
   - ✋ 检查点 4：实验计划前 → 批准实验方案
   - ✋ 检查点 5：实验结果后 → 决定下一步优化

3. **交付物要求**
   - 每完成一项任务必须输出**结构化文档**
   - 格式：【核心结论】（开头 3-5 句） + 【详细分析】 + 【需要您的决策】（列出选项）
   - 字数限制：≤ 2000 字

4. **角色分工示例**
   ```
   用户: "我想实现 arXiv:2024.xxxxx 这篇论文的 Adaptive Gaussian Pruning"

   → [3DGS 专家] 分析论文 → innovation_analysis.md
   → [医学专家] 评估医学适用性 → medical_feasibility_report.md
   → 【等待用户确认】
   → [3DGS 专家] 设计实现方案 → implementation_plan.md
   → 【等待用户确认】
   → [编程专家] GitHub 调研 + 代码实现 → code_review.md
   → 【等待用户确认】
   → [编程专家] 集成到 baseline
   → [调参专家] 设计实验 → experiment_plan.md
   → 【等待用户确认】
   → [调参专家] 执行实验 → result_analysis.md
   → [进度秘书] 记录到 knowledge_base.md
   ```

---


## 实现新功能的完整工作流

### 典型场景：将论文创新点迁移到 R²-Gaussian

**阶段 1：创新点分析（3DGS + 医学专家）**

1. **[3DGS 专家]** 在 `cc-agent/3dgs_expert/` 创建任务记录
   ```bash
   cd cc-agent/3dgs_expert
   # 更新 record.md：当前分析 arXiv:2024.xxxxx
   ```

2. 使用 MCP arXiv 工具获取论文，提取：
   - 核心算法改进
   - 新的损失函数
   - 网络结构变化
   - 生成 `innovation_analysis.md`

3. **[医学专家]** 评估医学适用性
   - 检查是否适用于 CT 场景
   - 识别医学特有约束（辐射剂量、扫描时间）
   - 生成 `medical_feasibility_report.md`

4. **✋ 等待用户确认：** 是否继续实现？

---

**阶段 2：技术方案设计（3DGS 专家）**

1. 在 `cc-agent/3dgs_expert/implementation_plans/` 创建详细方案
   - 需修改的文件列表（如 `train.py`, `gaussian_model.py`）
   - 新增的算法模块（放在 `r2_gaussian/utils/`）
   - 预期的技术挑战

2. **✋ 等待用户确认：** 审核技术路线

---

**阶段 3：代码实现（编程专家）**

1. **[编程专家]** 在 `cc-agent/code/` 工作
   ```bash
   cd cc-agent/code
   # 更新 record.md：当前实现 XXX 功能
   ```

2. 使用 MCP GitHub 工具查阅原论文代码
   - 克隆仓库到 `cc-agent/论文/archived/<paper_name>/code_repo/`
   - 理解实现细节，记录到 `github_research/`

3. 新增代码放在 `cc-agent/code/scripts/`，修改 baseline 通过 Git 跟踪

4. 生成 `code_review.md`：
   - 修改的文件和函数列表
   - 新增的依赖库
   - 潜在兼容性风险

5. **✋ 等待用户确认：** 批准代码修改

6. 实现并集成，确保向下兼容（使用 try-except 模式）

---

**阶段 4：实验与调参（调参专家 + 编程专家）**

1. **[调参专家]** 设计实验计划
   ```markdown
   ## 实验计划
   ### 消融实验
   - Baseline (无新功能)
   - +新功能（默认参数）
   - +新功能（调优参数）

   ### 对比实验
   - vs. R²-Gaussian baseline
   - vs. SAX-NeRF
   ```

2. **✋ 等待用户确认：** 批准实验方案

3. **[编程专家]** 埋点收集数据
   ```python
   # 在 train.py 中添加日志
   tb_writer.add_scalar("新功能/metric_name", value, iteration)
   ```

4. **[调参专家]** 执行实验并分析
   - 运行训练命令
   - 收集定量指标（PSNR, SSIM）和可视化
   - 诊断性能瓶颈
   - 生成 `result_analysis.md`

5. **✋ 等待用户确认：** 选择优化方向或结束

---

**阶段 5：知识沉淀（进度秘书）**

1. **[进度秘书]** 更新 `cc-agent/records/` 下的文档：
   - `decision_log.md` - 记录所有决策点
   - `project_timeline.md` - 更新时间线
   - `knowledge_base.md` - 添加成功案例或失败教训

2. 打 Git tag 标记里程碑
   ```bash
   git tag -a v1.1-add-feature-x -m "实现论文 XXX 的 YYY 功能"
   git push origin v1.1-add-feature-x
   ```

---

## MCP 工具配置

为了充分使用科研助手团队系统，需要安装以下 MCP 服务器：

### 必需的 MCP 工具

#### 1. arXiv 服务器（论文搜索下载）
```bash
npx @modelcontextprotocol/server-arxiv
```
**用途：** 3DGS 专家和医学专家查找相关论文

#### 2. GitHub 服务器（代码调研）
```bash
npx @modelcontextprotocol/server-github
```
**用途：** 编程专家查阅原论文代码实现

#### 3. 文件系统服务器（文件操作）
```bash
npx @modelcontextprotocol/server-filesystem
```
**用途：** 所有专家读写工作文档

#### 4. SQLite 服务器（实验数据库）
```bash
npx @modelcontextprotocol/server-sqlite
```
**用途：** 调参专家记录超参数和实验结果

#### 5. Brave Search（可选，网络搜索）
```bash
npx @modelcontextprotocol/server-brave-search
```
**用途：** 查找技术博客、医学术语解释、开源数据集

### 配置方法

**在 Claude Desktop 配置文件中添加：**

- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-arxiv"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your_github_token>"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/qyhu/Documents/r2_ours/r2_gaussian"
      ]
    },
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "/home/qyhu/Documents/r2_ours/r2_gaussian/cc-agent/records/experiments.db"
      ]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "<your_brave_api_key>"
      }
    }
  }
}
```

### 获取 API 密钥

- **GitHub Token:** 在 https://github.com/settings/tokens 创建
  - 需要权限：`repo`（访问代码仓库）
  - 类型：Classic Token

- **Brave Search API:** 在 https://brave.com/search/api/ 注册
  - 免费套餐：每月 2,000 次查询

### 验证安装

配置完成后重启 Claude Desktop，在对话中尝试：
```
请使用 arXiv 工具搜索 "3D Gaussian Splatting" 相关论文
```

---



## 重要提醒

1. **初始化是成功的关键** - 差的初始化会导致失败，务必用 `--evaluate` 检查
2. **场景归一化到 [-1,1]³** - 所有坐标和参数需遵守此约定
3. **用户确认是必需的** - 科研助手系统的所有关键决策点必须等待批准
4. **实验必须可复现** - 记录配置、种子、环境、git commit
5. **失败也要记录** - 在 `knowledge_base.md` 中记录失败教训避免重复踩坑

---
