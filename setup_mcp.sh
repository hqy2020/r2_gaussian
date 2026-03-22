#!/bin/bash
# MCP 工具快速配置脚本

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MCP 工具配置向导"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 配置文件路径
CONFIG_DIR="$HOME/.config/Claude"
CONFIG_FILE="$CONFIG_DIR/claude_desktop_config.json"

# 创建配置目录
mkdir -p "$CONFIG_DIR"

# 获取 GitHub Token
echo "步骤 1/3：配置 GitHub Token"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "请访问 https://github.com/settings/tokens 获取 Personal Access Token"
echo ""
echo "提示："
echo "  1. 点击 'Generate new token' → 选择 'classic'"
echo "  2. 勾选权限：✅ repo"
echo "  3. 复制生成的 token (格式: ghp_xxxx...)"
echo ""
read -p "请粘贴您的 GitHub Token: " GITHUB_TOKEN

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GitHub Token 不能为空"
    exit 1
fi

echo "✅ GitHub Token 已保存"
echo ""

# 询问是否配置 Brave Search
echo "步骤 2/3：配置 Brave Search (可选)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
read -p "是否配置 Brave Search API？(y/N): " USE_BRAVE

BRAVE_CONFIG=""
if [[ "$USE_BRAVE" =~ ^[Yy]$ ]]; then
    echo ""
    echo "请访问 https://brave.com/search/api/ 获取 API Key"
    echo ""
    read -p "请粘贴您的 Brave API Key: " BRAVE_KEY

    if [ ! -z "$BRAVE_KEY" ]; then
        BRAVE_CONFIG=",
    \"brave-search\": {
      \"command\": \"npx\",
      \"args\": [\"-y\", \"@modelcontextprotocol/server-brave-search\"],
      \"env\": {
        \"BRAVE_API_KEY\": \"${BRAVE_KEY}\"
      }
    }"
        echo "✅ Brave Search 已配置"
    fi
else
    echo "⏭️  跳过 Brave Search 配置"
fi

echo ""
echo "步骤 3/3：生成配置文件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 获取当前项目路径
PROJECT_PATH="/home/qyhu/Documents/r2_ours/r2_gaussian"
DB_PATH="$PROJECT_PATH/cc-agent/records/experiments.db"

# 生成配置文件
cat > "$CONFIG_FILE" <<EOF
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
        "${PROJECT_PATH}"
      ]
    },
    "sqlite": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sqlite",
        "--db-path",
        "${DB_PATH}"
      ]
    }${BRAVE_CONFIG}
  }
}
EOF

# 设置文件权限
chmod 600 "$CONFIG_FILE"

echo "✅ 配置文件已生成：$CONFIG_FILE"
echo ""

# 验证 JSON 格式
echo "验证配置文件格式..."
if command -v jq &> /dev/null; then
    if jq empty "$CONFIG_FILE" 2>/dev/null; then
        echo "✅ JSON 格式正确"
    else
        echo "⚠️  JSON 格式可能有误，但仍可尝试使用"
    fi
else
    echo "ℹ️  未安装 jq，跳过格式验证"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  配置完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "下一步："
echo "  1. 重启 Claude Desktop"
echo "  2. 在对话中测试："
echo "     '请使用 arXiv 工具搜索 3D Gaussian Splatting'"
echo ""
echo "配置文件位置："
echo "  $CONFIG_FILE"
echo ""
echo "查看配置："
echo "  cat $CONFIG_FILE"
echo ""
