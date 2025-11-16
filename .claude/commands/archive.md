---
description: 归档并清空 progress.md（当超过 2000 字时，由 @research-project-coordinator 处理）
---

请使用 Task 工具调用 @research-project-coordinator 执行以下任务：

**任务类型：** 归档 progress.md 文件

**要求：**
1. 检查 `cc-agent/records/progress.md` 文件大小
2. 如果文件字数超过 2000 字，执行归档操作：
   - 在 `cc-agent/records/archives/` 目录下创建归档文件
   - 归档文件命名格式：`progress_YYYY_MM_DD_HHMMSS.md`
   - 将当前 progress.md 的完整内容复制到归档文件
   - 在新的 progress.md 中保留标题和最近 1-2 条记录作为参考
   - 在 progress.md 顶部添加归档提示：
     ```markdown
     > **历史记录已归档到：** archives/progress_YYYY_MM_DD_HHMMSS.md
     ```
3. 如果文件字数不足 2000 字，提示用户无需归档

**重要提示：**
- 归档前必须确保原内容已完整复制
- 归档后的 progress.md 应该是"轻量级"的，方便快速查看最近进度
- 使用中文记录所有操作
- 归档完成后向用户确认归档文件路径
