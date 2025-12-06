# 🌍 智能旅行规划助手

基于 LangGraph + MCP + RAG 的 Agentic 旅行规划系统，结合多模型协作、知识检索和实时数据查询，为用户提供智能化的旅行方案。

## 📋 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
- [系统要求](#系统要求)
- [安装部署](#安装部署)
- [使用指南](#使用指南)
- [数据库管理](#数据库管理)
- [项目结构](#项目结构)
- [API 接口](#api-接口)
- [故障排查](#故障排查)

---

## 🎯 项目简介

这是一个智能旅行规划系统，由 Python 后端（`aggentic_RAG`）和 Next.js 前端（`travel-chat-ui`）组成，通过自然语言对话为用户生成完整的旅行方案。

### 主要特性

- **双模式查询**：
  - 🔍 **简单查询模式**：只需目的地，快速获取景点推荐
  - 🎯 **完整规划模式**：提供详细信息，生成含交通、住宿、天气、黄历的完整方案

- **多模型协作**：
  - **DeepSeek R1**：处理复杂推理和优化问题（预算优化、路线规划）
  - **Qwen3**：负责信息提取和普通对话

- **实时数据集成**：
  - 🚄 12306 火车票查询
  - 🗺️ 高德地图（导航、酒店、天气、POI 搜索）
  - 📅 八字黄历服务器（查询出行吉日）

- **知识库检索**：
  - RAG 向量数据库存储旅游攻略
  - 支持 TXT、MD、PDF、CSV 格式导入

---

## 🚀 核心功能

### 1. 智能信息提取
- 从自然语言对话中提取出发地、目的地、日期、预算等关键信息
- 支持相对日期（"明天"、"下周"）自动转换
- 多轮对话上下文保持

### 2. 交通方案对比
- 自动查询火车票信息（车次、时间、票价）
- 计算自驾路线（距离、时间、过路费）
- 综合对比推荐最优方案

### 3. 住宿推荐
- 根据预算自动选择酒店等级关键词
  - 预算 > 500元：五星/豪华
  - 预算 300-500元：品牌连锁
  - 预算 < 300元：经济型/快捷
- 提供酒店名称、价格、地址信息

### 4. 天气与黄历
- 查询旅行日期的天气预报（最多4天）
- 查询农历黄历，分析是否适合出行
- 展示宜忌事项

### 5. 行程规划
- 结合 RAG 知识库和实时 POI 数据
- 生成每日详细行程
- 计算预算分配（交通、住宿、餐饮、门票）

---

## 🏗️ 技术架构

### 后端架构 (`aggentic_RAG`)

```
LangGraph Workflow
├── planner_node          # 信息提取（Qwen3）
├── route_after_plan      # 模式路由（简单/完整）
├── rag_search_node       # RAG 检索 + 酒店查询
├── route_after_rag       # 跳过不需要的节点
├── train_query_node      # 12306 火车票查询
├── lucky_day_query_node  # 黄历吉日查询
├── weather_query_node    # 天气预报查询
├── r1_analysis_node      # DeepSeek R1 深度分析（按需）
└── synthesizer_node      # 方案合成（Qwen3）
```

### 核心技术栈

**后端**：
- **LangGraph**：工作流编排
- **LangChain**：LLM 抽象层
- **ChromaDB**：向量数据库（存储 RAG 数据）
- **DashScope**：阿里云模型服务（Qwen3、文本嵌入）
- **DeepSeek API**：复杂推理模型
- **MCP**：模型上下文协议（12306、高德地图、八字服务器）

**前端**：
- **Next.js 15**：React 框架
- **Tailwind CSS**：样式
- **LangGraph SDK**：流式响应
- **Turbo**：Monorepo 构建工具

---

## 💻 系统要求

### 后端
- Python >= 3.11
- 8GB+ RAM（用于向量数据库）
- Windows/Linux/macOS

### 前端
- Node.js >= 18
- pnpm >= 10.6.3

### API 密钥
- **DeepSeek API Key**（用于 DeepSeek R1 模型）
- **DashScope API Key**（用于 Qwen3 和文本嵌入）
- **MCP 服务器 URL**（12306、高德地图、八字服务器等）

---

## 🔑 API 密钥获取

### 1. DeepSeek API Key

**用途**：DeepSeek R1 模型用于复杂推理和优化任务

**获取步骤**：

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册账号并登录
3. 进入「API Keys」页面
4. 点击「创建新密钥」
5. 复制生成的 API Key（格式：`sk-xxxxxxxxxxxxxxxx`）

**费用**：按 Token 使用量计费，新用户通常有免费额度

### 2. DashScope API Key（阿里云）

**用途**：Qwen3 模型和文本嵌入（text-embedding-v3）

**获取步骤**：
1. 访问 [阿里云 DashScope](https://dashscope.aliyun.com/)
2. 使用阿里云账号登录（需要实名认证）
3. 进入「API-KEY 管理」
4. 创建新的 API Key
5. 复制生成的 API Key（格式：`sk-xxxxxxxxxxxxxxxx`）

**费用**：

- Qwen3 模型：按 Token 计费，有免费额度
- 文本嵌入：按调用次数计费，新用户有免费额度

### 3. MCP 服务器配置

**MCP（Model Context Protocol）** 是连接外部工具的协议。本项目使用以下 MCP 服务器：

#### 可用的 MCP 服务器：

1. **12306 Server** - 火车票查询
   - 提供商：ModelScope
   - 功能：查询火车车次、票价、时刻表

2. **Gaode Map Server** - 高德地图
   - 提供商：ModelScope
   - 功能：路线规划、酒店查询、天气预报、POI 搜索

3. **Bazi Server** - 八字黄历服务器
   - 提供商：ModelScope
   - 功能：查询农历、黄历宜忌、出行吉日

4. **Bing Search Server** - 必应搜索（可选）
   - 提供商：ModelScope
   - 功能：搜索最新旅游资讯

5. **Flight Server** - 航班查询（可选）
   - 提供商：ModelScope
   - 功能：查询航班信息

#### 如何获取 MCP 服务器 URL：

**方式1：使用 ModelScope 提供的公开服务**

1. 访问 [ModelScope MCP 广场](https://www.modelscope.cn/)
2. 搜索对应的 MCP 服务（如「12306 MCP」、「高德地图 MCP」）
3. 获取服务的 SSE 接口地址

**方式2：自己部署 MCP 服务器**
1. 从 GitHub 获取 MCP 服务器源码
2. 按照服务器文档部署到自己的服务器
3. 使用自己的服务器地址

**注意**：
- MCP 服务器 URL 通常以 `/sse` 结尾（Server-Sent Events）
- 某些 MCP 服务可能需要额外的 API Key（如高德地图需要高德开放平台 Key）
- 建议使用稳定的服务提供商，避免服务中断

---

## 📦 安装部署

### 1. 克隆项目

```bash
git clone <repository-url>
cd "agentic RAG"
```

### 2. 后端安装

#### 2.1 安装依赖

```bash
cd aggentic_RAG
pip install -e .
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

#### 2.2 配置环境变量

在 `aggentic_RAG` 目录下创建 `.env` 文件：

```bash
# 模型 API 密钥（必填）
DEEPSEEK_API_KEY=sk-your-deepseek-api-key-here
DASHSCOPE_API_KEY=sk-your-dashscope-api-key-here

# LangChain 追踪（可选，用于调试）
LANGCHAIN_TRACING_V2=false

# MCP 配置文件路径（默认值）
MCP_CONFIG_PATH=travel_agent/config/servers_config.json

# ChromaDB 向量数据库路径（默认值）
CHROMA_PERSIST_DIR=./data/travel_vectordb
```

**重要**：将 `sk-your-xxx-key-here` 替换为你从上一步获取的真实 API Key。

#### 2.3 配置 MCP 服务器

编辑 `travel_agent/config/servers_config.json`：

```json
{
    "mcp_servers": [
        {
            "name": "12306 Server",
            "url": "https://your-12306-mcp-server-url/sse"
        },
        {
            "name": "Gaode Server",
            "url": "https://your-gaode-mcp-server-url/sse"
        },
        {
            "name": "bazi Server",
            "url": "https://your-bazi-mcp-server-url/sse"
        }
    ],
    "agent": {
        "name": "TravelPlannerAssistant",
        "instructions": "你是一名专业的旅行规划智能助手。你可以帮助用户通过以下工具进行旅游规划：1) 12306查询 - 查询火车票信息；2) Gaode地图 - 路线规划和导航；3) 八字工具 - 命理信息查询。请根据用户需求调用相应工具，并生成详细的旅行方案。"
    }
}
```

**配置说明**：
- `mcp_servers`：MCP 服务器列表
  - `name`：服务器名称（用于日志）
  - `url`：服务器 SSE 接口地址
- `agent`：Agent 配置
  - `name`：Agent 名称
  - `instructions`：Agent 系统提示词

**必需的 MCP 服务器**：
- ✅ **12306 Server**：火车票查询（完整规划模式必需）
- ✅ **Gaode Server**：地图、酒店、天气（完整规划模式必需）
- ✅ **Bazi Server**：黄历查询（完整规划模式必需）

**可选的 MCP 服务器**：
- ⭕ **Bing Search Server**：搜索旅游资讯
- ⭕ **Flight Server**：航班查询

将 URL 替换为你获取的真实 MCP 服务器地址。

#### 2.4 启动后端服务

```bash
langgraph dev
```

后端将在 `http://localhost:2024` 启动。

**验证安装**：
- 启动成功后，访问 `http://localhost:2024/ok` 应返回 `{"status": "ok"}`
- 检查日志确认 MCP 服务器连接成功

### 3. 前端安装

#### 3.1 安装依赖

```bash
cd travel-chat-ui
pnpm install
```

#### 3.2 配置环境变量

编辑 `.env` 文件：

```bash
# LangGraph 后端 URL
LANGGRAPH_API_URL=http://localhost:2024
```

#### 3.3 启动前端服务

```bash
pnpm dev
```

前端将在 `http://localhost:3000` 启动。

### 4. 访问应用

打开浏览器访问 `http://localhost:3000`，即可开始使用智能旅行规划助手。

---

## 📖 使用指南

### 简单查询模式

**适用场景**：快速了解某个城市的景点信息

**示例**：
```
用户：苏州有什么好玩的？
用户：推荐一下成都的景点
```

**系统行为**：
- 只调用 RAG 知识库和高德地图 POI 搜索
- 不查询火车票、天气、黄历
- 返回景点列表和简要介绍

### 完整规划模式

**适用场景**：需要完整的旅行方案

**需要提供的信息**：
- ✅ 出发地：如"上海"
- ✅ 目的地：如"苏州"
- ✅ 旅行天数：如"2天"
- ✅ 预算：如"1000元"
- ✅ 出发日期：如"12月10日" 或 "明天"

**示例**：
```
用户：我想从上海去苏州玩2天，预算1000元，12月10日出发，帮我规划一下
```

**系统行为**：
1. 提取关键信息
2. 查询 RAG 知识库
3. 查询火车票（12306）
4. 计算自驾路线（高德地图）
5. 推荐酒店（高德地图 + 预算过滤）
6. 查询天气预报（高德地图）
7. 查询黄历吉日（八字服务器）
8. 如需复杂优化，调用 DeepSeek R1 分析
9. 合成完整方案

**输出内容**：
- 📋 基本信息（路线、日期、天气、黄历）
- 🚗🚆 交通方案对比（自驾 vs 火车）
- 🏨 住宿推荐（2-3家酒店）
- 📅 每日行程安排
- 💰 预算分配明细
- 💡 特别建议（老人/儿童友好提示）

---

## 🗄️ 数据库管理

本项目使用 **ChromaDB** 作为向量数据库，存储旅游攻略文档。

### 数据库位置

```
aggentic_RAG/data/travel_vectordb/
```

### 导入数据

#### 1. 准备文档

将旅游攻略文档放入 `data/travel_docs/` 目录：

```bash
cd aggentic_RAG
mkdir -p data/travel_docs
```

支持的格式：
- `.txt` - 纯文本
- `.md` - Markdown
- `.pdf` - PDF 文档
- `.csv` - CSV 表格

#### 2. 批量导入

打开 Python REPL：

```bash
python
```

执行导入脚本：

```python
from travel_agent.tools.rag_tool import TravelRAGTool

# 创建 RAG 工具实例
rag_tool = TravelRAGTool()

# 导入数据（自动生成向量）
rag_tool.build_knowledge_base(
    data_dir="./data/travel_docs",
    force_recreate=False  # False=追加模式，True=重建数据库
)

print("数据导入完成！")
```

#### 3. 自定义分块参数（可选）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 创建自定义分块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # 每块字符数
    chunk_overlap=100,     # 重叠字符数
)

# 应用到 RAG 工具
rag_tool.text_splitter = text_splitter

# 导入数据
rag_tool.build_knowledge_base("./data/travel_docs")
```

### 查询数据

```python
# 查看数据库统计信息
stats = rag_tool.get_stats()
print(f"文档总数: {stats['total_docs']}")
print(f"数据源: {stats['sources']}")

# 搜索测试
results = rag_tool.search("苏州园林")
for result in results:
    print(result.page_content)
```

### 删除数据

#### 删除指定文件

```python
# 删除某个源文件的所有文档
rag_tool.delete_by_source("data/travel_docs/suzhou_guide.txt")
print("已删除 suzhou_guide.txt 的所有文档")
```

#### 重建数据库

```python
# 清空并重建整个数据库
rag_tool.build_knowledge_base(
    data_dir="./data/travel_docs",
    force_recreate=True  # 强制重建
)
```

### 更新数据

```python
# 追加新文档（自动跳过已存在的文档）
rag_tool.build_knowledge_base(
    data_dir="./data/travel_docs",
    force_recreate=False
)
```

### UUID 机制

每个文档块生成稳定的 UUID（基于内容和来源）：

```python
# UUID 生成规则
UUID = MD5(f"{source_file}:{chunk_index}:{content[:100]}")
```

优点：
- ✅ 自动去重
- ✅ 精确删除
- ✅ 支持增量更新

---

## 📁 项目结构

```
agentic RAG/
├── aggentic_RAG/                 # Python 后端
│   ├── travel_agent/             # 主应用代码
│   │   ├── config/               # 配置文件
│   │   │   ├── prompts.py        # Prompt 模板
│   │   │   ├── settings.py       # 全局配置
│   │   │   └── servers_config.json  # MCP 服务器配置
│   │   ├── graph/                # LangGraph 工作流
│   │   │   ├── workflow.py       # 主工作流定义
│   │   │   ├── state.py          # 状态管理
│   │   │   └── nodes.py          # 节点实现
│   │   ├── models/               # 模型封装
│   │   │   ├── llm.py            # Qwen3 模型
│   │   │   └── deepseek_r1.py   # DeepSeek R1 模型
│   │   ├── tools/                # 工具集成
│   │   │   ├── rag_tool.py       # RAG 向量检索
│   │   │   └── mcp_tools.py      # MCP 工具调用
│   │   └── app.py                # FastAPI 服务器
│   ├── data/                     # 数据目录
│   │   ├── travel_docs/          # 旅游攻略文档
│   │   └── travel_vectordb/      # ChromaDB 向量数据库
│   ├── .env                      # 环境变量
│   ├── requirements.txt          # Python 依赖
│   ├── setup.py                  # 安装脚本
│   └── langgraph.json            # LangGraph 配置
│
├── travel-chat-ui/               # Next.js 前端
│   ├── apps/                     # 应用目录
│   │   ├── web/                  # Web 前端
│   │   │   ├── src/
│   │   │   │   ├── components/   # React 组件
│   │   │   │   │   └── thread/   # 对话线程组件
│   │   │   │   ├── app/          # Next.js 页面
│   │   │   │   └── providers/    # Context Providers
│   │   │   └── package.json
│   │   └── agents/               # Agent 相关代码
│   ├── .env                      # 环境变量
│   ├── package.json              # 项目配置
│   ├── pnpm-workspace.yaml       # pnpm 工作区配置
│   └── turbo.json                # Turbo 配置
│
└── README.md                     # 本文档
```

---

## 🔌 API 接口

### 后端 API

**基础 URL**: `http://localhost:2024`

#### 1. 创建新对话

```http
POST /threads
Content-Type: application/json

{}
```

**响应**:
```json
{
  "thread_id": "uuid-string"
}
```

#### 2. 发送消息（流式）

```http
POST /threads/{thread_id}/runs/stream
Content-Type: application/json

{
  "assistant_id": "travel_agent",
  "input": {
    "messages": [
      {
        "type": "human",
        "content": "我想从上海去苏州玩2天"
      }
    ]
  },
  "stream_mode": ["values"]
}
```

**响应**（Server-Sent Events）:
```
data: {"event": "values", "data": {...}}
```

#### 3. 获取对话历史

```http
GET /threads/{thread_id}/history
```

---

## 🐛 故障排查

### 1. 后端启动失败

**问题**：`ModuleNotFoundError: No module named 'travel_agent'`

**解决**：
```bash
cd aggentic_RAG
pip install -e .
```

### 2. 向量数据库为空

**问题**：简单查询返回"未找到相关信息"

**解决**：导入旅游攻略文档到 RAG
```python
from travel_agent.tools.rag_tool import TravelRAGTool
rag = TravelRAGTool()
rag.build_knowledge_base("./data/travel_docs")
```

### 3. MCP 工具调用失败

**问题**：火车票、天气查询返回错误

**解决**：
1. 检查 MCP 服务器是否启动
2. 验证 `servers_config.json` 配置
3. 查看后端日志

```bash
langgraph dev --verbose
```

### 4. 前端连接失败

**问题**：前端显示 "Connection Error"

**解决**：
1. 确认后端已启动（`http://localhost:2024`）
2. 检查前端 `.env` 中的 `LANGGRAPH_API_URL`
3. 检查防火墙/代理设置

### 5. DeepSeek R1 调用失败

**问题**：复杂规划没有使用 R1 分析

**解决**：
1. 检查 `DEEPSEEK_API_KEY` 是否正确
2. 确认用户查询触发了 `needs_deep_analysis`
3. 查看后端日志确认 R1 节点是否被调用

---

## 📝 开发说明

### 修改 Prompt

编辑 `aggentic_RAG/travel_agent/config/prompts.py`：

```python
# 修改规划提示词
PLANNER_SYSTEM_PROMPT = """你的自定义提示词..."""

# 修改合成提示词
SYNTHESIZER_PROMPT_TEMPLATE = """你的自定义提示词..."""
```

### 调整模型参数

编辑 `aggentic_RAG/travel_agent/config/settings.py`：

```python
# RAG 分块大小
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50

# 检索数量
RAG_TOP_K = 5

# 模型温度
LLM_TEMPERATURE = 0.7
```

### 添加新节点

1. 在 `nodes.py` 中定义节点函数：
```python
def my_new_node(state: TravelPlanState) -> TravelPlanState:
    # 处理逻辑
    return state
```

2. 在 `workflow.py` 中注册节点：
```python
travel_workflow.add_node("my_new_node", my_new_node)
travel_workflow.add_edge("previous_node", "my_new_node")
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证。

---

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - 工作流编排
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [DashScope](https://dashscope.aliyun.com/) - 阿里云模型服务
- [DeepSeek](https://www.deepseek.com/) - DeepSeek R1 模型

---

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

**项目地址**: [GitHub Repository URL]

---

**祝您使用愉快！🎉**
