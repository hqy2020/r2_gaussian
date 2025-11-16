# MCP 工具配置完整指南

## 📌 前提条件检查

### 1. 确认 Claude Desktop 已安装

**检查方法：**
```bash
# Linux
which claude-desktop

# 或者查看是否有 Claude 配置目录
ls ~/.config/Claude/
```

**如果未安装：**
- 访问 https://claude.ai/download 下载 Claude Desktop
- Linux 用户也可以使用 Claude Code (CLI 版本)

---

## 🔧 配置步骤

### 第一步：创建配置目录和文件

```bash
# 创建 Claude 配置目录（如果不存在）
mkdir -p ~/.config/Claude

# 创建或编辑配置文件
nano ~/.config/Claude/claude_desktop_config.json
# 或使用您喜欢的编辑器：vim、code 等
```

### 第二步：获取必需的 API 密钥

#### 2.1 获取 GitHub Personal Access Token

**步骤：**
1. 访问 https://github.com/settings/tokens
2. 点击 **"Generate new token"** → 选择 **"Classic"**
3. 设置名称：`Claude MCP GitHub Access`
4. 设置过期时间：建议选 **"No expiration"**（永不过期）
5. **勾选权限：**
   - ✅ `repo` - 完整仓库访问权限（这会自动勾选所有 repo 子权限）
6. 滚动到底部，点击 **"Generate token"**
7. **⚠️ 立即复制 token**（只显示一次！）- 格式类似：`ghp_xxxxxxxxxxxxxxxxxxxx`

**保存 token：**
```bash
# 临时保存到文件（稍后会用到）
echo "ghp_你的token" > ~/.github_token_temp
chmod 600 ~/.github_token_temp
```

#### 2.2 获取 Brave Search API Key（可选）

**步骤：**
1. 访问 https://brave.com/search/api/
2. 点击 **"Get Started"** 或 **"Sign Up"**
3. 注册账号并登录
4. 在 Dashboard 中找到您的 API Key
5. 免费套餐：每月 2,000 次查询

**保存 API key：**
```bash
echo "BSA_你的key" > ~/.brave_api_key_temp
chmod 600 ~/.brave_api_key_temp
```

---

### 第三步：编写配置文件

**打开配置文件：**
```bash
nano ~/.config/Claude/claude_desktop_config.json
```

**完整配置内容（复制以下内容）：**

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
        "GITHUB_PERSONAL_ACCESS_TOKEN": "替换为你的GitHub_Token"
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
        "BRAVE_API_KEY": "替换为你的Brave_API_Key"
      }
    }
  }
}
```

**⚠️ 重要替换：**
- 将 `"替换为你的GitHub_Token"` 改为您从 GitHub 复制的 token（如 `ghp_xxxx...`）
- 将 `"替换为你的Brave_API_Key"` 改为您的 Brave API key（如 `BSA_xxxx...`）
- 如果不需要 Brave Search，可以删除整个 `"brave-search"` 部分

**使用命令自动替换（推荐）：**
```bash
# 读取保存的 token
GITHUB_TOKEN=$(cat ~/.github_token_temp)
BRAVE_KEY=$(cat ~/.brave_api_key_temp 2>/dev/null || echo "")

# 创建配置文件
cat > ~/.config/Claude/claude_desktop_config.json <<EOF
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
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
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
    }
  }
}
EOF

# 删除临时文件
rm -f ~/.github_token_temp ~/.brave_api_key_temp

echo "✅ 配置文件已创建：~/.config/Claude/claude_desktop_config.json"
```

---

### 第四步：验证配置

**检查配置文件语法：**
```bash
# 使用 jq 检查 JSON 格式是否正确
cat ~/.config/Claude/claude_desktop_config.json | jq .

# 如果没有 jq，安装它
sudo apt-get install jq  # Ubuntu/Debian
```

**检查 Node.js 和 npx：**
```bash
# MCP 服务器需要 Node.js
node --version  # 应该 >= 16.x
npx --version

# 如果未安装
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**测试 MCP 服务器是否可以运行：**
```bash
# 测试 arXiv 服务器
npx -y @modelcontextprotocol/server-arxiv &
sleep 3
pkill -f server-arxiv
echo "✅ arXiv 服务器测试完成"

# 测试 GitHub 服务器
export GITHUB_PERSONAL_ACCESS_TOKEN="你的token"
npx -y @modelcontextprotocol/server-github &
sleep 3
pkill -f server-github
echo "✅ GitHub 服务器测试完成"
```

---

### 第五步：重启 Claude Desktop

**Linux:**
```bash
# 如果 Claude Desktop 正在运行，重启它
pkill claude-desktop
claude-desktop &

# 或者使用系统托盘重启
```

**重启后验证：**
1. 打开 Claude Desktop
2. 在对话中输入：
   ```
   请使用 arXiv 工具搜索 "3D Gaussian Splatting" 相关论文
   ```
3. 如果工具正常工作，会显示搜索结果

---

## 🎯 最小配置（仅必需工具）

如果暂时不需要 Brave Search，使用这个简化配置：

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
        "GITHUB_PERSONAL_ACCESS_TOKEN": "你的GitHub_Token"
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
    }
  }
}
```

---

## ❓ 常见问题

### 1. 找不到配置文件？

**可能的位置：**
```bash
# Linux
~/.config/Claude/claude_desktop_config.json

# macOS
~/Library/Application Support/Claude/claude_desktop_config.json

# Windows
%APPDATA%\Claude\claude_desktop_config.json
```

### 2. MCP 服务器启动失败？

**检查日志：**
```bash
# Claude Desktop 日志位置（Linux）
~/.config/Claude/logs/

# 查看最新日志
tail -f ~/.config/Claude/logs/main.log
```

**常见原因：**
- Node.js 未安装或版本过低（需要 >= 16.x）
- JSON 配置格式错误（缺少逗号、引号等）
- GitHub Token 权限不足或已过期
- 网络问题导致无法下载 MCP 包

### 3. GitHub 工具无法访问私有仓库？

**解决方案：**
- 确认 Token 勾选了 `repo` 权限
- 如果是组织仓库，需要额外勾选 `read:org`
- Token 没有过期

### 4. 如何更新 MCP 服务器版本？

```bash
# npx 会自动使用最新版本，但可以清除缓存
npx clear-npx-cache

# 或者手动清除
rm -rf ~/.npm/_npx/
```

---

## 🔐 安全建议

1. **保护配置文件权限：**
   ```bash
   chmod 600 ~/.config/Claude/claude_desktop_config.json
   ```

2. **不要将配置文件提交到 Git：**
   ```bash
   echo "claude_desktop_config.json" >> ~/.config/Claude/.gitignore
   ```

3. **定期轮换 GitHub Token：**
   - 每 3-6 个月更新一次 Token
   - 发现泄露立即撤销

4. **使用最小权限原则：**
   - GitHub Token 只勾选必需的权限
   - Brave API Key 不要分享给他人

---

## 📚 参考资源

- **MCP 官方文档：** https://modelcontextprotocol.io/
- **Claude Desktop 下载：** https://claude.ai/download
- **GitHub Token 管理：** https://github.com/settings/tokens
- **Brave Search API：** https://brave.com/search/api/

---

**配置完成后，您就可以：**
✅ 使用 arXiv 工具搜索和下载论文
✅ 使用 GitHub 工具浏览代码仓库
✅ 访问本地文件系统（在项目目录范围内）
✅ 使用 SQLite 数据库记录实验
✅ 使用 Brave Search 进行网络搜索

**下一步：** 开始使用科研助手团队系统，按照 `cc-agent/构想.md` 中的工作流程进行论文实现！
