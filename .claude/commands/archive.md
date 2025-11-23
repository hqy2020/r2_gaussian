---
description: 归档并压缩 progress.md（当超过 2000 字时，由 @research-project-coordinator 处理）
---

请使用 Task 工具调用 @research-project-coordinator 执行以下任务：

**任务类型：** 归档并压缩 progress.md 文件

**执行流程：**

### 第一步：检查文件大小
1. 检查 `cc-agent/records/progress.md` 文件大小（字数统计）
2. 如果文件字数不足 2000 字，提示用户无需归档，直接返回

### 第二步：创建归档文件
1. 确保 `cc-agent/records/archives/` 目录存在，如不存在则创建
2. 生成归档文件名：`progress_YYYY_MM_DD_HHMMSS.md`（使用当前时间戳）
3. 将当前 progress.md 的**完整内容**复制到归档文件
4. 在归档文件顶部添加元信息：
   ```markdown
   ---
   归档时间：YYYY-MM-DD HH:MM:SS
   原始文件：progress.md
   归档原因：文件内容超过 2000 字
   ---
   ```

### 第三步：压缩 progress.md
1. 从原始内容中提取关键信息：
   - 最近 1-2 条完整的工作记录
   - 当前状态摘要（进行中、待处理）
   - 关键决策和重要发现
2. 创建新的轻量级 progress.md，格式：
   ```markdown
   # 工作进度记录

   > **历史记录已归档到：** archives/progress_YYYY_MM_DD_HHMMSS.md
   > **最后更新：** YYYY-MM-DD HH:MM:SS

   ## 当前状态摘要
   - **最近完成：** [从归档内容中提取的最近完成工作]
   - **进行中：** [当前正在处理的任务]
   - **待处理：** [需要后续处理的事项]

   ## 最近工作记录
   [保留最近 1-2 条完整记录作为参考]
   ```

**重要提示：**
- 归档前必须确保原内容已完整复制到归档文件
- 归档后的 progress.md 应该是"轻量级"的（建议控制在 500-800 字），方便快速查看最近进度
- 压缩时保留最重要的上下文信息，确保 @recap 能够快速理解当前状态
- 使用中文记录所有操作
- 归档完成后向用户确认：
  - 归档文件路径
  - 压缩后的 progress.md 字数
  - 保留的关键信息摘要
