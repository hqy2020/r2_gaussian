---
description: 快速回顾上次工作内容（由 @research-project-coordinator 处理）
---

请使用 Task 工具调用 @research-project-coordinator 执行以下任务：

**任务类型：** 回顾上次工作进度

**要求：**
1. 读取 `cc-agent/records/progress.md` 文件
2. 提取最近 3-5 条工作记录
3. 生成简洁的工作摘要（中文）
4. 格式要求：
   ```markdown
   # 上次工作回顾

   ## 最近完成的任务
   1. [时间] 任务名称 - 状态
   2. [时间] 任务名称 - 状态
   ...

   ## 当前状态
   - 正在进行：XXX
   - 待处理：XXX

   ## 建议下一步
   - XXX
   ```

**重要提示：**
- 如果 progress.md 不存在或为空，提示用户这是新的工作会话
- 重点关注最近 2-3 天的工作记录
- 突出显示未完成的任务和待决策的事项
