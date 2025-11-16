---
description: 记录当前工作到 progress.md（由 @research-project-coordinator 处理）
---

请使用 Task 工具调用 @research-project-coordinator 执行以下任务：

**任务类型：** 记录当前工作进度

**要求：**
1. 分析当前对话中完成的工作内容
2. 提取关键信息：任务目标、执行状态、完成时间、重要决策
3. 将记录追加到 `cc-agent/records/progress.md` 文件末尾
4. 使用中文记录
5. 格式要求：
   ```markdown
   ## [时间戳] 任务名称
   - **任务目标：** XXX
   - **执行状态：** 已完成/进行中/待定
   - **关键决策：** XXX
   - **下一步行动：** XXX
   ```

**重要提示：**
- 往 progress.md 文件**末尾追加**内容，不要修改原有内容
- 如果 progress.md 不存在，则创建新文件
- 记录完成后向用户确认已记录的内容
