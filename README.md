# 🌍 智能旅行规划助手

基于 **LangChain Agent + Streamlit + MCP + RAG** 的智能旅行规划系统，结合双模型协作（DeepSeek R1 + Qwen3）、知识检索和实时数据查询，为用户提供智能化的旅行方案。

> **架构升级**: 已从 LangGraph 迁移到 **LangChain Agent**，无递归限制，支持复杂多目的地行程规划。

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

这是一个智能旅行规划系统，采用 **Streamlit UI + LangChain Agent** 架构，通过自然语言对话为用户生成完整的旅行方案。系统支持简单查询和复杂行程规划两种模式，根据场景复杂度自动选择合适的处理路径。

### 主要特性

- **双模式查询**：
  - 🔍 **简单查询模式**：只需目的地，快速获取景点推荐
  - 🎯 **完整规划模式**：提供详细信息，生成含交通、住宿、天气、黄历的完整方案

- **双模型协作**：
  - **DeepSeek R1**：处理复杂推理和多目的地路线优化（预算分配、时间安排）
  - **Qwen3**：负责信息提取、工具调用决策和方案生成
  - 系统根据场景复杂度自动选择 R1 主导或 Qwen3 主导模式

- **实时数据集成** (基于 MCP 协议)：
  - 🚄 12306 火车票查询（自动获取站点代码，查询车次时刻表）
  - 🚗 高德地图自驾路线（自动计算距离、时间、过路费）
  - 🏨 高德地图酒店搜索（根据预算自动筛选）
  - ☀️ 高德地图天气预报（支持多日预报）
  - 📅 八字黄历查询（农历、宜忌、吉日）
  - ✈️ 航班查询（可选，长途>800km 自动触发）

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
LangChain Agent + 预分析层
├── pre_analyze_query         # 预分析（场景检测、多目的地识别）
│   ├── simple_query          # 简单查询：只需景点信息
│   ├── complex_query         # 复杂查询：特殊需求、预算紧张
│   └── multi_destination     # 多目的地：2个以上城市
│
├── [R1主导模式]              # 复杂/多目的地场景
│   ├── r1_strategy_node      # R1分解行程、制定query_plan
│   ├── ReAct Loop            # 按query_plan执行工具调用
│   │   ├── train_query       # 12306查询（自动附带自驾路线）
│   │   ├── gaode_weather     # 天气查询
│   │   ├── gaode_hotel       # 酒店搜索
│   │   ├── lucky_day         # 黄历查询
│   │   └── flight_query      # 航班查询（条件触发）
│   ├── r1_optimization       # R1二次优化（仅单目的地）
│   └── synthesizer_node      # 生成最终方案
│
└── [Qwen3主导模式]           # 简单场景
    ├── ReAct Loop            # Qwen3自主决策调用工具
    └── synthesizer_node      # 生成最终方案
```

**关键特性**:
- ✅ **无递归限制**: 从 LangGraph 迁移到 LangChain Agent，支持任意长度的query_plan
- ✅ **自动重试**: MCP工具调用失败自动重试2次（SSE连接保护）
- ✅ **超时保护**: 12306查询90秒超时，防止长时间阻塞
- ✅ **错误恢复**: 单个工具失败不影响整体流程

### 核心技术栈

**后端**：
- **LangChain Agent**: 核心Agent执行引擎（ReAct模式）
- **Streamlit**: Web UI 界面
- **ChromaDB**: 向量数据库（存储旅游攻略）
- **DashScope**: 阿里云模型服务（Qwen3-plus + text-embedding-v3）
- **DeepSeek API**: 深度推理模型（deepseek-reasoner）
- **MCP (Model Context Protocol)**: 外部工具集成
  - 12306 Server（火车票）
  - Gaode Server（地图、天气、酒店）
  - Bazi Server（黄历）
  - Flight Server（航班，可选）

---

## 💻 系统要求

### 运行环境
- Python >= 3.11
- 8GB+ RAM（用于向量数据库和模型推理）
- Windows/Linux/macOS

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

**推荐方式**（支持开发模式，方便调试）：
```bash
cd aggentic_RAG
pip install -e .
```

或者使用 requirements.txt：

```bash
pip install -r requirements.txt
```

> **说明**：虽然 `app.py` 会自动添加模块路径，但 `pip install -e .` 能让其他工具脚本（如 `check_mcp_health.py`）正常运行，并支持开发模式下的代码热重载。

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

#### 2.4 启动应用

在项目根目录运行:

```bash
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。

**验证安装**：
- 启动成功后，浏览器会自动打开 Streamlit UI
- 在侧边栏可以查看已加载的工具列表
- 后台日志会显示 MCP 服务器连接状态

### 3. 使用健康检查工具（可选）

检查所有 MCP 服务器连接状态:

```bash
python check_mcp_health.py
```

输出示例:
```
🔍 MCP 服务器健康检查
============================================================

🔧 12306 Server
   状态: ✅ 正常
   工具数: 3
   示例工具: get-tickets, get-stations-code-in-city, ...

🔧 Gaode Server
   状态: ✅ 正常
   工具数: 5
   ...

✅ 所有 MCP 服务器状态正常！
```

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

### 🧠 DeepSeek R1 复杂推理触发条件

**DeepSeek R1** 仅在**复杂场景**下才会被调用，以控制成本和提高效率。

#### 什么时候会调用 R1？

系统在信息提取阶段会自动判断是否需要复杂推理，满足以下 **任意一个条件** 就会设置 `needs_deep_analysis=true`：

1. **复杂的多城市路线**
   - 示例：“上海 → 苏州 → 杭州 → 南京，5天”
   - 需要：路线优化、时间分配

2. **紧张的预算优化**
   - 示例：“4人去苏州3天，总预算1500元”（人均375元/天）
   - 需要：交通、住宿、餐饮、门票的精细优化

3. **多重冲突的约束条件**
   - 示例：“带着两个70岁老人和一个5岁孩子，时间只有1天，要去3个景点”
   - 需要：平衡老人体力、孩子兴趣、时间限制

4. **复杂的优化问题**
   - 示例：“最省钱的方案”、“最快到达的路线”、“最多景点的行程”
   - 需要：多目标优化、权衡分析

#### 什么时候不会调用 R1？

大多数普通场景只需要 **Qwen3** 就能处理，**不会调用 R1**：

- ✅ 简单查询：“苏州有什么好玩的？”
- ✅ 单城市、充裕预算：“上海去苏州2天，预算3000元”
- ✅ 没有特殊约束：“两个成年人去杭州3天”

#### 如何验证 R1 是否被调用？

查看 Streamlit 后台日志（运行 `streamlit run app.py` 的终端窗口）：

如果看到以下日志，说明 R1 被调用了：
```
📊 预分析结果: {..., 'needs_deep_analysis': True, ...}
🌍 检测到多目的地场景: comma_separated_2

🧠 R1 主导模式: multi_destination

🧠 [r1_analysis] 深度分析调用: 用户计划于...
  R1分析完成，返回 1771 字符
```

或者在 Streamlit UI 中看到：
```
🌍 检测到多目的地行程，将调用深度路线优化...
```

#### R1 调用示例

**会调用 R1 的查询**：
```
用户：我带2个老人和1个儿童，从北京出发，去苏州、4天，预算2500元，
     老人不能走太多路，孩子喜欢动物园，给我最省钱的方案。

系统：⚠️ 检测到复杂场景：
      - 紧张预算（4人4天只有2500元）
      - 特殊约束（老人体力 + 儿童兴趣）
      - 优化目标（最省钱）
      → 调用 DeepSeek R1 进行深度分析...
```

**不会调用 R1 的查询**：
```
用户：我想从上海去苏州玩2天，预算1500元，明天出发。

系统：✅ 普通场景，使用 Qwen3 处理
      → 直接查询车票、酒店、天气，生成方案
```

---

## 🗄️ 数据库管理

本项目使用 **ChromaDB** 作为向量数据库，存储旅游攻略文档。

### 数据库位置

```
aggentic_RAG/data/travel_vectordb/
```

### 导入数据

本项目提供 **两种** 文档导入方式：

#### 方式1：Streamlit UI 上传（推荐）

1. 启动应用：`streamlit run app.py`
2. 在左侧边栏找到 **"📚 知识库管理"** 区域
3. 点击 **"上传文件"** 按钮，选择文档
4. 系统自动完成:
   - ✅ 文件上传到 `data/travel_docs/`
   - ✅ 文本分块处理
   - ✅ 向量化并存入 ChromaDB
   - ✅ 实时显示处理进度

**支持的格式**：
- `.txt` - 纯文本
- `.md` - Markdown
- `.pdf` - PDF 文档
- `.csv` - CSV 表格

**优点**：
- 🎯 简单直观，无需写代码
- 📊 实时反馈处理状态
- 🔄 自动去重（基于 UUID）

#### 方式2：命令行批量导入（适合大量文档）

1. 将旅游攻略文档放入 `data/travel_docs/` 目录：
   ```bash
   cd aggentic_RAG
   mkdir -p data/travel_docs
   # 复制你的文档到 data/travel_docs/
   ```

2. 打开 Python REPL：
   ```bash
   python
   ```

3. 执行导入脚本：
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

**优点**：
- 📦 适合批量处理数百上千个文档
- ⚙️ 支持自定义分块参数
- 📄 适合脚本化和自动化场景

---

### 自定义分块参数（可选）

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
travel-planning-agent/
├── aggentic_RAG/                 # Python 后端包
│   ├── travel_agent/             # 主应用代码
│   │   ├── config/               # 配置文件
│   │   │   ├── prompts.py        # Prompt 模板（Planner, Synthesizer, R1）
│   │   │   ├── settings.py       # 全局配置（路径、参数）
│   │   │   └── servers_config.json  # MCP 服务器配置
│   │   ├── core/                 # 核心模块
│   │   │   └── agent_executor.py # Agent执行引擎（已备用）
│   │   ├── graph/                # 工作流节点（保留兼容LangGraph）
│   │   │   ├── workflow.py       # LangGraph工作流定义（已备用）
│   │   │   ├── state.py          # 状态类型定义
│   │   │   └── nodes.py          # 所有节点实现（planner, R1, tools, synthesizer）
│   │   ├── tools/                # 工具集
│   │   │   ├── rag_tool.py       # RAG 向量检索
│   │   │   ├── mcp_tools.py      # MCP 工具管理器（带重试机制）
│   │   │   ├── r1_tool.py        # DeepSeek R1 封装
│   │   │   └── tool_registry.py  # 工具注册表（所有工具定义）
│   │   └── app.py                # Agent入口（已备用）
│   ├── data/                     # 数据目录
│   │   ├── travel_docs/          # 旅游攻略文档
│   │   └── travel_vectordb/      # ChromaDB 向量数据库
│   ├── .env                      # 环境变量
│   ├── requirements.txt          # Python 依赖
│   └── setup.py                  # 安装脚本
│
├── app.py                        # ✅ Streamlit UI（主入口）
├── check_mcp_health.py           # ✅ MCP 健康检查工具
├── README.md                     # 项目文档
├── LICENSE                       # MIT 许可证
├── .gitignore                    # Git 忽略配置
└── .gitattributes                # Git 属性配置
```

**核心文件说明**:
- `app.py`: Streamlit UI 主程序，包含预分析层和 Agent 创建
- `nodes.py`: 所有节点实现（planner, r1_strategy, train_query, synthesizer 等）
- `mcp_tools.py`: MCP 工具管理器，带自动重试和超时保护
- `tool_registry.py`: 所有可用工具的定义和描述
- `prompts.py`: 所有 LLM Prompt 模板

---

## 📝 使用方式

### Streamlit UI 界面

启动应用后，浏览器自动打开 `http://localhost:8501`：

1. **主界面**: 聊天对话窗口
2. **侧边栏**:
   - 📚 上传旅游攻略文档（可选）
   - ⚙️ 系统配置（最大迭代次数）
   - 🧰 查看工具列表
   - 🗑️ 清空聊天记录

3. **对话示例**:
   ```
   用户：我想12月18日从上海出发，去徐州和青岛旅游3天，预算1500元
   
   系统：🌍 检测到多目的地行程，将调用深度路线优化...
   🚄 正在查询交通信息...
   ☀️ 正在查询天气预报...
   🏨 正在搜索酒店...
   
   [详细的旅行方案]
   ```

### 命令行工具

#### MCP 健康检查
```bash
python check_mcp_health.py
```

#### 导入旅游攻略到 RAG
```python
from aggentic_RAG.travel_agent.tools.rag_tool import TravelRAGTool
rag = TravelRAGTool()
rag.build_knowledge_base("./aggentic_RAG/data/travel_docs")
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
3. 运行健康检查工具

```bash
python check_mcp_health.py
```

4. 查看 Streamlit 后台日志（终端窗口）

### 4. Streamlit 启动失败

**问题**：Streamlit 无法启动或报错

**解决**：
1. 确认已安装 Streamlit: `pip install streamlit`
2. 检查端口 8501 是否被占用
3. 尝试指定端口：`streamlit run app.py --server.port 8502`

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

### 添加新工具

1. 在 `tool_registry.py` 中添加工具定义：
```python
ToolDefinition(
    name="my_new_tool",
    description="新工具的功能描述",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数说明"
            }
        },
        "required": ["param1"]
    },
    tool_type="mcp",  # 或 "r1", "special"
    server_name="Your Server",
    mcp_tool_name="tool_name_in_mcp"
)
```

2. 如果需要自定义处理逻辑，在 `app.py` 中添加工具处理函数

### 添加新节点（高级）

在 `nodes.py` 中添加新的处理节点：
```python
async def my_custom_node(state: TravelPlanState) -> Dict[str, Any]:
    """自定义节点逻辑"""
    # 处理逻辑
    return {
        "custom_field": "result",
        "messages": [AIMessage(content="处理完成")]
    }
```

**注意**: 当前系统使用 LangChain Agent，不需要手动编排节点。Agent 会自动决定调用顺序。

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

## 📧 联系方式

该项目Created by Alex，如有问题或建议，请提交 Issue 或联系项目维护者。

**项目地址**: [https://github.com/alexlmoney83-oss/travel-planning-agent]

---

**祝您使用愉快！🎉**
