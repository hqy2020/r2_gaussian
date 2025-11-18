# R²-Gaussian 任务完成清单

## 完成任何任务后必须执行的步骤

### 1. 代码质量检查

#### 语法检查
```bash
# Python 语法检查（如果安装了 flake8）
python -m py_compile <修改的文件>.py
```

#### 导入检查
```bash
# 确保所有导入都正确
python -c "import r2_gaussian; print('Import OK')"
```

### 2. Git 版本控制

#### 查看修改
```bash
git status
git diff
```

#### 提交代码
```bash
# 添加修改的文件
git add <文件路径>

# 提交（遵循 Git 约定）
git commit -m "type: 简短描述"

# 可选：打标签
git tag -a v1.x-feature-name -m "描述"
```

### 3. 文档更新

#### 必须更新的文档
1. **cc-agent/records/progress.md**
   - 调用 `/record` 记录当前工作
   - 或手动追加进度（禁止修改原有内容）

2. **相关专家的 record.md**
   - `cc-agent/3dgs_expert/record.md`
   - `cc-agent/medical_expert/record.md`
   - `cc-agent/code/record.md`
   - `cc-agent/experiments/record.md`

3. **决策日志**
   - `cc-agent/records/decision_log.md`
   - 记录关键决策和理由

4. **知识库**
   - `cc-agent/records/knowledge_base.md`
   - 记录成功案例或失败教训

### 4. 实验相关（如果涉及训练）

#### 训练前检查
- [ ] 确认环境已激活：`conda activate r2_gaussian_new`
- [ ] 检查数据文件存在
- [ ] 检查初始化文件存在
- [ ] 验证配置参数正确

#### 训练后检查
- [ ] TensorBoard 日志正常生成
- [ ] 检查点文件保存成功
- [ ] 评估指标在合理范围内
- [ ] 生成深度图（如果启用深度约束）

#### 实验记录
```bash
# 运行评估
python test.py -m <输出路径>

# 生成深度图
python generate_depth_maps.py <输出路径>

# 启动 TensorBoard 分析
tensorboard --logdir <输出路径> --port 6006
```

### 5. 代码审查（创新点移植后）

#### 必须检查项
- [ ] **向下兼容**：旧代码仍能正常运行
- [ ] **参数默认值**：新功能默认关闭
- [ ] **错误处理**：使用 try-except 防止崩溃
- [ ] **文档注释**：关键函数有中文文档字符串
- [ ] **TensorBoard 日志**：新功能的指标已记录

#### 影响分析
使用 serena MCP 工具分析：
```python
# 1. 找到修改的符号
find_symbol(name_path="函数或类名", relative_path="文件路径")

# 2. 找到所有引用
find_referencing_symbols(name_path="符号名", relative_path="文件路径")

# 3. 评估影响范围
# 输出：文件路径:行号:符号名称:用途
```

### 6. 测试验证

#### 单元测试
```bash
# 如果有测试文件
python test_<功能名>.py
```

#### 集成测试
```bash
# 运行完整训练流程（短迭代）
python train.py -s <数据路径> -m <输出路径> --iterations 100
```

#### 初始化质量测试
```bash
# 如果修改了初始化逻辑
python initialize_pcd.py --data <数据路径> --evaluate
```

### 7. 清理工作

#### 删除临时文件
```bash
# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

#### 整理输出目录
```bash
# 删除失败的训练输出
rm -rf output/failed_experiment_*
```

### 8. AI Agent 专用检查

#### 文档位置检查
- [ ] **禁止**在项目根目录创建 AI 文档
- [ ] 所有 AI 文档必须在 `cc-agent/` 对应文件夹

#### progress.md 维护
- [ ] 新内容**追加**到文件末尾
- [ ] **禁止**修改已有内容
- [ ] 超过 2000 字时调用 `/archive`

### 9. 重要提醒检查

#### 初始化检查
- [ ] 初始化质量直接影响训练成功率
- [ ] 使用 `--evaluate` 标志检查初始化 PSNR

#### 坐标归一化
- [ ] 场景归一化到 [-1, 1]³
- [ ] 所有坐标和参数遵守此约定

#### 可复现性
- [ ] 记录训练配置到 `cfg_args.yml`
- [ ] 记录随机种子
- [ ] 记录 git commit hash
- [ ] 记录环境信息

## 完成清单模板

任务完成后，复制以下模板到 progress.md：

```markdown
## [日期] 任务名称

### 完成内容
- [ ] 代码修改：<文件列表>
- [ ] 文档更新：<文档列表>
- [ ] 测试验证：<测试结果>

### Git 提交
- Commit: <commit hash>
- Tag: <tag name>（可选）

### 实验结果（如适用）
- 训练时间：X 分钟
- PSNR: XX.XX dB
- SSIM: 0.XXXX
- 输出路径：<路径>

### 决策记录
- 选择方案：<方案描述>
- 理由：<决策理由>

### 下一步
- [ ] 待办事项 1
- [ ] 待办事项 2
```