"""
智能旅游规划助手 - Streamlit UI
基于LangChain Agent，无递归限制
"""
import streamlit as st
import tempfile
import os
import asyncio
import warnings
from typing import List, Dict, Any

# suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*LangChainDeprecationWarning.*')
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader
)
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.tools import Tool
from dotenv import load_dotenv

# 设置页面配置（必须是第一个Streamlit命令）
st.set_page_config(
    page_title="🗺️ 智能旅游规划助手",
    page_icon="🗺️",
    layout="wide"
)

# 加载环境变量（指定.env文件路径）
env_path = os.path.join(os.path.dirname(__file__), 'aggentic_RAG', '.env')
load_dotenv(dotenv_path=env_path, override=True)

# 添加项目路径到sys.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aggentic_RAG'))

# 应用标题
st.title("🗺️ 智能旅游规划助手")
st.markdown("基于RAG + Agentic AI的旅游规划系统，无递归限制")

# 侧边栏 - 文档上传
with st.sidebar:
    st.header("📚 旅游攻略文档")
    
    # 上传多种格式的文件
    uploaded_files = st.file_uploader(
        label="上传旅游攻略文档",
        type=["txt", "pdf", "csv"],
        accept_multiple_files=True,
        help="支持 TXT、PDF、CSV 格式的文档"
    )
    
    st.markdown("---")
    
    # 系统配置
    st.header("⚙️ 系统配置")
    max_iterations = st.slider(
        "最大迭代次数",
        min_value=10,
        max_value=100,
        value=50,
        help="Agent执行的最大步骤数，无递归限制！"
    )
    
    st.markdown("---")
    
    # 清空对话按钮
    if st.button("🗑️ 清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 文档上传现在是可选的
if not uploaded_files:
    st.info("💡 提示：您可以选择上传旅游攻略文档，或直接进行实时查询")
    # 不再强制要求上传文档


# 配置RAG检索器
@st.cache_resource(ttl="1h")
def configure_rag_retriever(uploaded_files):
    """配置RAG检索器 - 支持多种文件格式"""
    docs = []
    
    # 创建临时目录存储上传的文件
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        # 根据文件类型选择合适的 loader
        file_extension = os.path.splitext(file.name)[1].lower()
        
        try:
            if file_extension == ".txt":
                loader = TextLoader(temp_filepath, encoding="utf-8")
            elif file_extension == ".pdf":
                loader = PyPDFLoader(temp_filepath)
            elif file_extension == ".csv":
                loader = CSVLoader(temp_filepath, encoding="utf-8")
            else:
                st.warning(f"跳过不支持的文件格式: {file.name}")
                continue
            
            docs.extend(loader.load())
            st.success(f"✅ 已加载: {file.name}")
        except Exception as e:
            st.error(f"加载 {file.name} 失败: {e}")
    
    # 文档分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    
    # 使用Qwen的Embedding模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    
    # 创建向量数据库
    vectordb = Chroma.from_documents(splits, embeddings)
    
    # 返回检索器
    return vectordb.as_retriever(search_kwargs={"k": 5})


# 配置检索器（如果有文档）
if uploaded_files:
    with st.spinner("正在加载文档并建立索引..."):
        retriever = configure_rag_retriever(uploaded_files)
        st.success(f"✅ 已加载 {len(uploaded_files)} 个文档")
else:
    retriever = None


# 启用嵌套asyncio支持
import nest_asyncio
nest_asyncio.apply()

# 创建全局event loop用于MCP调用
import asyncio
_mcp_loop = asyncio.new_event_loop()
_mcp_manager = None


# ========== 预分析层：照搬旧版本 LangGraph 的分流逻辑 ==========

def detect_multi_destination(user_query: str, extraction: dict) -> dict:
    """检测是否为多目的地场景（排除往返/回程误判）
    
    照搬自 nodes.py 中的 detect_multi_destination 函数
    """
    # === 1) 优先排除往返场景 ===
    roundtrip_keywords = ["往返", "来回", "回程", "返程", "返回"]
    if any(kw in user_query for kw in roundtrip_keywords):
        print("  🔄 检测到往返关键词，不算多目的地")
        return {
            'is_multi_destination': False,
            'detected_keywords': [],
            'raw_destination_text': extraction.get('destination', ''),
            'detection_method': 'roundtrip_excluded'
        }
    
    # === 2) 多目的地关键词 ===
    multi_dest_keywords = [
        "再去", "然后去", "接着去", "顺便去",
        "再到", "然后到", "接着到",
        "再去看看", "再看看",
        "之后去", "之后到"
    ]
    detected_keywords = [kw for kw in multi_dest_keywords if kw in user_query]
    if detected_keywords:
        return {
            'is_multi_destination': True,
            'detected_keywords': detected_keywords,
            'raw_destination_text': extraction.get('destination', ''),
            'detection_method': 'keyword'
        }
    
    # === 3) 目的地字段中包含多个城市（逗号/顿号/“和”分隔） ===
    destination = extraction.get('destination', '') or ''
    origin = extraction.get('origin', '') or ''
    
    # 统一分隔符
    norm = destination.replace(',', '，').replace('、', '，').replace('和', '，')
    cities = [c.strip() for c in norm.split('，') if c.strip()]
    
    # 去重保持顺序
    unique_cities = []
    for c in cities:
        if c not in unique_cities:
            unique_cities.append(c)
    
    if len(unique_cities) >= 3:
        return {
            'is_multi_destination': True,
            'detected_keywords': [],
            'raw_destination_text': destination,
            'detection_method': 'comma_separated_3plus'
        }
    
    if len(unique_cities) == 2:
        # 如果两个城市中包含出发地，通常是往返（例如 上海, 南京）→ 视为单目的地
        if origin and origin in unique_cities:
            return {
                'is_multi_destination': False,
                'detected_keywords': [],
                'raw_destination_text': destination,
                'detection_method': 'origin_pair_excluded'
            }
        # 两个且都不是出发地 → 多目的地
        return {
            'is_multi_destination': True,
            'detected_keywords': [],
            'raw_destination_text': destination,
            'detection_method': 'comma_separated_2'
        }
    
    return {
        'is_multi_destination': False,
        'detected_keywords': [],
        'raw_destination_text': destination
    }


def pre_analyze_query(user_query: str, llm) -> dict:
    """预分析用户查询，判断场景类型
    
    照搬自 nodes.py 中的 planner_node 逻辑
    
    返回:
        dict: {
            'scenario_type': 'simple' | 'complex' | 'multi_destination',
            'needs_deep_analysis': bool,  # 是否需要R1主导
            'extraction': {...},  # 提取的信息
            'multi_dest_info': {...}  # 多目的地检测结果
        }
    """
    from datetime import datetime
    
    today = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year
    
    # 构建提取 prompt，参考旧版本的 PLANNER_SYSTEM_PROMPT
    extraction_prompt = f"""You are a travel planning assistant. Today's date is {today}.

Your task: Extract key information from the user query and determine if it needs deep analysis.

RULES:
- Output ONLY valid JSON. NO explanations, NO markdown code blocks.
- Date conversion: "今天"/"today" = {today}, "明天"/"tomorrow" = +1 day, etc.
- Use Chinese for city names.
- Set "needs_deep_analysis" to true if:
  * Complex multi-city routes (e.g., "A和B", "A再去B")
  * Budget optimization needed (tight budget with many requirements)
  * Multiple conflicting constraints (e.g., elderly + children, limited time + many places)
  * Special needs: 老人, 小孩, 儿童, 亲子, 残疾, etc.

Output this exact JSON structure:
{{
  "destination": "extracted destination city or cities (comma separated if multiple)",
  "origin": "extracted origin city",
  "travel_days": 0,
  "budget": 0,
  "travel_date": "YYYY-MM-DD",
  "preferences": ["preference1"],
  "needs_deep_analysis": false,
  "has_special_needs": false
}}

User query: {user_query}
"""
    
    try:
        response = llm.invoke(extraction_prompt)
        content = response.content.strip()
        
        # 移除可能的markdown代码块
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
            if content.startswith('json'):
                content = content[4:]
        
        import json
        extraction = json.loads(content)
        print(f"\n📊 预分析结果: {extraction}")
        
    except Exception as e:
        print(f"⚠️ 预分析失败: {e}")
        # 默认值
        extraction = {
            "destination": "",
            "origin": "",
            "travel_days": 0,
            "budget": 0,
            "travel_date": "",
            "preferences": [],
            "needs_deep_analysis": False,
            "has_special_needs": False
        }
    
    # 检测多目的地场景
    multi_dest_info = detect_multi_destination(user_query, extraction)
    
    # 确定场景类型和是否需要R1
    if multi_dest_info.get('is_multi_destination', False):
        scenario_type = 'multi_destination'
        needs_r1 = True
        print(f"  🌍 检测到多目的地场景: {multi_dest_info.get('detection_method')}")
    elif extraction.get('needs_deep_analysis', False) or extraction.get('has_special_needs', False):
        scenario_type = 'complex'
        needs_r1 = True
        print(f"  🧠 检测到复杂场景（特殊需求/预算紧张）")
    else:
        scenario_type = 'simple'
        needs_r1 = False
        print(f"  ✅ 简单场景，Qwen3主导")
    
    return {
        'scenario_type': scenario_type,
        'needs_deep_analysis': needs_r1,
        'extraction': extraction,
        'multi_dest_info': multi_dest_info
    }

def get_mcp_manager_sync():
    """Sync wrapper to get MCP manager"""
    global _mcp_manager
    if _mcp_manager is None:
        from aggentic_RAG.travel_agent.tools.mcp_tools import MCPToolManager
        _mcp_manager = MCPToolManager()
        _mcp_loop.run_until_complete(_mcp_manager.initialize())
    return _mcp_manager

def call_mcp_tool_sync(server_name: str, tool_name: str, **kwargs) -> str:
    """Sync wrapper to call MCP tool"""
    manager = get_mcp_manager_sync()
    return _mcp_loop.run_until_complete(manager.call_tool(server_name, tool_name, **kwargs))


# 创建工具列表
def create_tools(retriever) -> List[Tool]:
    """创建Agent可用的工具列表"""
    from langchain.tools.retriever import create_retriever_tool
    from aggentic_RAG.travel_agent.tools.tool_registry import AVAILABLE_TOOLS
    
    tools = []
    
    # 1. RAG检索工具（如果有文档）
    if retriever is not None:
        rag_tool = create_retriever_tool(
            retriever=retriever,
            name="rag_search",
            description="用于查询旅游攻略、景点信息、美食推荐等。输入城市名或景点名。"
        )
        tools.append(rag_tool)
    
    # 2. 初始化MCP管理器
    try:
        get_mcp_manager_sync()  # 预初始化MCP连接
    except Exception as e:
        st.warning(f"MCP初始化失败: {e}")
    
    # 3. 从tool_registry获取所有MCP工具定义
    import json
    from datetime import datetime
    
    # 特殊处理函数：查询火车票（需要先获取站点代码）
    def query_train_tickets(origin: str, destination: str, date: str) -> str:
        """Query train tickets - first get station codes, then query tickets"""
        print(f"\n🚂 [train_query] 调用: {origin} -> {destination}, {date}")
        try:
            # 1. 修正日期年份
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                current_year = datetime.now().year
                if date_obj.year < current_year:
                    date = date.replace(str(date_obj.year), str(current_year))
                    print(f"  日期修正为: {date}")
            except:
                pass
            
            # 2. 获取站点代码
            from_code = None
            to_code = None
            
            # 分别查询每个城市的站点代码
            for city, is_origin in [(origin, True), (destination, False)]:
                try:
                    print(f"  获取{city}站点代码...")
                    codes_result = call_mcp_tool_sync(
                        "12306 Server",
                        "get-stations-code-in-city",
                        city=city
                    )
                    print(f"  {city}站点查询结果: {str(codes_result)[:150]}")
                    
                    if codes_result and "error" not in str(codes_result).lower():
                        codes_data = json.loads(codes_result) if isinstance(codes_result, str) else codes_result
                        
                        # 处理数组格式 - 找主站（站名与城市名匹配的）
                        if isinstance(codes_data, list) and len(codes_data) > 0:
                            code = None
                            station_name = None
                            
                            # 优先找完全匹配的主站（如"上海"->"上海", "徐州"->"徐州"）
                            for station in codes_data:
                                name = station.get('station_name', '')
                                if name == city:  # 完全匹配
                                    code = station.get('station_code') or station.get('code') or station.get('telecode')
                                    station_name = name
                                    break
                            
                            # 其次找包含城市名的站（如"上海"->"上海虹桥", "上海南"）
                            if not code:
                                for station in codes_data:
                                    name = station.get('station_name', '')
                                    if city in name and '东' not in name and '南' not in name and '西' not in name and '北' not in name:
                                        code = station.get('station_code') or station.get('code') or station.get('telecode')
                                        station_name = name
                                        break
                            
                            # 最后尝试第一个包含城市名的站
                            if not code:
                                for station in codes_data:
                                    name = station.get('station_name', '')
                                    if city in name:
                                        code = station.get('station_code') or station.get('code') or station.get('telecode')
                                        station_name = name
                                        break
                            
                            if code:
                                if is_origin:
                                    from_code = code
                                else:
                                    to_code = code
                                print(f"  ✅ {city}站点: {station_name} (代码: {code})")
                            else:
                                print(f"  ⚠️ {city}未找到匹配的主站")
                except Exception as e:
                    print(f"  {city}站点查询异常: {e}")
            
            print(f"  站点代码汇总: {origin}={from_code}, {destination}={to_code}")
            
            # 3. 查询车票
            train_result = None
            if from_code and to_code:
                print(f"  步骤2: 使用站点代码查询车票...")
                train_result = call_mcp_tool_sync(
                    "12306 Server",
                    "get-tickets",
                    fromStation=from_code,
                    toStation=to_code,
                    date=date
                )
                print(f"  车票查询结果: {str(train_result)[:300]}")
            else:
                # 备选：直接使用城市名查询
                print(f"  步骤2(备选): 直接使用城市名查询...")
                train_result = call_mcp_tool_sync(
                    "12306 Server",
                    "get-tickets",
                    fromStation=origin,
                    toStation=destination,
                    date=date
                )
                print(f"  车票查询结果: {str(train_result)[:300]}")
            
            # 4. 同时查询自驾路线（照搬旧版本行为）
            driving_result = None
            try:
                print(f"\n🚗 步骤3: 自动查询自驾路线信息")
                
                # 先获取城市经纬度
                origin_coords = None
                dest_coords = None
                
                try:
                    origin_geo = call_mcp_tool_sync("Gaode Server", "maps_geo", address=origin)
                    if origin_geo and "error" not in str(origin_geo).lower():
                        geo_data = json.loads(origin_geo) if isinstance(origin_geo, str) else origin_geo
                        if isinstance(geo_data, dict) and 'return' in geo_data:
                            if isinstance(geo_data['return'], list) and len(geo_data['return']) > 0:
                                origin_coords = geo_data['return'][0].get('location')
                    
                    dest_geo = call_mcp_tool_sync("Gaode Server", "maps_geo", address=destination)
                    if dest_geo and "error" not in str(dest_geo).lower():
                        geo_data = json.loads(dest_geo) if isinstance(dest_geo, str) else dest_geo
                        if isinstance(geo_data, dict) and 'return' in geo_data:
                            if isinstance(geo_data['return'], list) and len(geo_data['return']) > 0:
                                dest_coords = geo_data['return'][0].get('location')
                    
                    if origin_coords and dest_coords:
                        print(f"  ✅ 经纬度: {origin}={origin_coords}, {destination}={dest_coords}")
                        driving_result = call_mcp_tool_sync(
                            "Gaode Server",
                            "maps_direction_driving",
                            origin=origin_coords,
                            destination=dest_coords
                        )
                        print(f"  ✅ 自驾路线查询成功: {str(driving_result)[:200]}")
                    else:
                        print(f"  ⚠️ 无法获取经纬度，跳过自驾路线")
                except Exception as geo_err:
                    print(f"  ⚠️ 地理编码失败: {geo_err}")
            except Exception as driving_err:
                print(f"  ⚠️ 自驾路线查询异常: {driving_err}")
            
            # 5. 组合返回结果（包含火车和自驾信息）
            combined_result = {}
            if train_result:
                combined_result["train"] = train_result
            if driving_result:
                combined_result["driving"] = driving_result
            
            if combined_result:
                return json.dumps(combined_result, ensure_ascii=False)
            else:
                return "交通查询失败"
        except Exception as e:
            print(f"  火车票查询异常: {e}")
            return f"火车票查询失败: {str(e)}"
    
    for tool_def in AVAILABLE_TOOLS:
        if tool_def.tool_type == "mcp":
            # 特殊处理train_query
            if tool_def.name == "train_query":
                def train_tool_func(tool_input: str) -> str:
                    try:
                        if tool_input.strip().startswith('{'):
                            kwargs = json.loads(tool_input)
                        else:
                            return "请提供JSON格式的参数"
                        
                        origin = kwargs.get("origin") or kwargs.get("from") or kwargs.get("fromStation", "")
                        destination = kwargs.get("destination") or kwargs.get("to") or kwargs.get("toStation", "")
                        date = kwargs.get("date", "")
                        
                        return query_train_tickets(origin, destination, date)
                    except Exception as e:
                        return f"参数解析失败: {str(e)}"
                
                tools.append(Tool(
                    name=tool_def.name,
                    description=tool_def.description,
                    func=train_tool_func
                ))
                continue
            
            # 其他MCP工具的通用处理
            def make_sync_tool(server_name, mcp_tool_name, tool_name):
                def call_tool(tool_input: str) -> str:
                    """Parse input string and call MCP tool"""
                    print(f"\n🛠️ [{tool_name}] 调用: {tool_input[:100]}")
                    try:
                        if tool_input.strip().startswith('{'):
                            kwargs = json.loads(tool_input)
                        else:
                            kwargs = {"query": tool_input}
                        
                        # 修正日期年份
                        if "date" in kwargs:
                            date_str = kwargs["date"]
                            try:
                                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                                current_year = datetime.now().year
                                if date_obj.year < current_year:
                                    kwargs["date"] = date_str.replace(str(date_obj.year), str(current_year))
                            except:
                                pass
                        
                        # 黄历工具特殊处理 - 转换为ISO时间格式
                        if tool_name == "lucky_day":
                            date_val = kwargs.pop('date', None) or kwargs.pop('query', '')
                            # 确保日期格式正确
                            if date_val:
                                date_val = date_val.strip()
                                kwargs = {"solarDatetime": f"{date_val}T12:00:00+08:00"}
                        
                        # 航班查询工具 - 参数映射
                        if tool_name == "flight_query":
                            dep = kwargs.pop('dep', None) or kwargs.pop('from', None) or kwargs.pop('origin', '')
                            arr = kwargs.pop('arr', None) or kwargs.pop('to', None) or kwargs.pop('destination', '')
                            date_val = kwargs.pop('date', '')
                            # 修正日期年份
                            if date_val:
                                try:
                                    date_obj = datetime.strptime(date_val.strip(), "%Y-%m-%d")
                                    if date_obj.year < datetime.now().year:
                                        date_val = date_val.replace(str(date_obj.year), str(datetime.now().year))
                                except:
                                    pass
                            kwargs = {"dep": dep, "arr": arr, "date": date_val}
                        
                        # 天气工具 - 确保city参数
                        if tool_name == "gaode_weather":
                            city = kwargs.pop("city", None) or kwargs.pop("location", None) or kwargs.pop("query", "")
                            # 清理城市名：去除日期、空格、逗号后的内容
                            if city:
                                city = city.strip()
                                # 如果包含逗号，取第一部分
                                if ',' in city:
                                    city = city.split(',')[0].strip()
                                if '，' in city:
                                    city = city.split('，')[0].strip()
                                # 去除"市"后缀保留纯城市名
                                city = city.replace('市', '')
                            kwargs = {"city": city}
                        
                        # 地理编码工具 - 确保address参数
                        if tool_name == "gaode_geo":
                            address = kwargs.pop("address", None) or kwargs.pop("location", None) or kwargs.pop("query", "")
                            kwargs = {"address": address}
                        
                        # 酒店搜索工具 - 确保keywords和city参数
                        if tool_name == "gaode_hotel_search":
                            keywords = kwargs.pop("keywords", None) or kwargs.pop("keyword", None) or kwargs.pop("query", "")
                            city = kwargs.pop("city", None) or kwargs.pop("location", "")
                            # 如果keywords没有包含"酒店"或"民宿"，自动补充
                            if keywords and "酒店" not in keywords and "民宿" not in keywords:
                                keywords = f"{keywords} 酒店"
                            kwargs = {"keywords": keywords, "city": city} if city else {"keywords": keywords}
                        
                        # POI搜索工具 - 确保keywords参数
                        if tool_name == "gaode_poi_search":
                            keywords = kwargs.pop("keywords", None) or kwargs.pop("keyword", None) or kwargs.pop("query", "")
                            city = kwargs.pop("city", None) or kwargs.pop("location", "")
                            kwargs = {"keywords": keywords, "city": city} if city else {"keywords": keywords}
                        
                        # 自驾路线工具 - 确保origin和destination参数（应为坐标）
                        if tool_name == "gaode_driving":
                            origin = kwargs.pop("origin", None) or kwargs.pop("from", "")
                            destination = kwargs.pop("destination", None) or kwargs.pop("to", "")
                            kwargs = {"origin": origin, "destination": destination}
                        
                        print(f"  参数: {kwargs}")
                        return call_mcp_tool_sync(server_name, mcp_tool_name, **kwargs)
                    except json.JSONDecodeError:
                        # 对于简单字符串输入，根据工具类型处理
                        print(f"  简单字符串输入: {tool_input}")
                        tool_input_clean = tool_input.strip()
                        
                        if tool_name == "gaode_weather":
                            city = tool_input_clean
                            if ',' in city:
                                city = city.split(',')[0].strip()
                            if '，' in city:
                                city = city.split('，')[0].strip()
                            city = city.replace('市', '')
                            return call_mcp_tool_sync(server_name, mcp_tool_name, city=city)
                        elif tool_name == "gaode_geo":
                            return call_mcp_tool_sync(server_name, mcp_tool_name, address=tool_input_clean)
                        elif tool_name == "lucky_day":
                            iso_dt = f"{tool_input_clean}T12:00:00+08:00"
                            return call_mcp_tool_sync(server_name, mcp_tool_name, solarDatetime=iso_dt)
                        elif tool_name == "gaode_hotel_search":
                            # 如果输入是城市名，补充酒店关键词
                            keywords = f"{tool_input_clean} 酒店" if "酒店" not in tool_input_clean else tool_input_clean
                            return call_mcp_tool_sync(server_name, mcp_tool_name, keywords=keywords)
                        elif tool_name == "gaode_poi_search":
                            return call_mcp_tool_sync(server_name, mcp_tool_name, keywords=tool_input_clean)
                        else:
                            return call_mcp_tool_sync(server_name, mcp_tool_name, query=tool_input_clean)
                    except Exception as e:
                        return f"工具调用失败: {str(e)}"
                return call_tool
            
            # 创建工具
            mcp_tool = Tool(
                name=tool_def.name,
                description=tool_def.description,
                func=make_sync_tool(tool_def.server_name, tool_def.mcp_tool_name, tool_def.name)
            )
            tools.append(mcp_tool)
        
        elif tool_def.tool_type == "r1":
            # R1 深度分析工具 - 使用 DeepSeek R1 API
            def make_r1_tool():
                def call_r1_analysis(tool_input: str) -> str:
                    """使用DeepSeek R1进行深度分析"""
                    print(f"\n🧠 [r1_analysis] 深度分析调用: {tool_input[:200]}")
                    try:
                        if tool_input.strip().startswith('{'):
                            kwargs = json.loads(tool_input)
                        else:
                            kwargs = {"problem": tool_input, "context": {}}
                        
                        problem = kwargs.get("problem", "")
                        context = kwargs.get("context", {})
                        
                        # 调用DeepSeek R1 API（使用正确的 DeepSeek 配置）
                        from openai import OpenAI
                        
                        # DeepSeek R1 使用其自己的 API
                        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
                        if not deepseek_api_key:
                            return "深度分析不可用: 未配置 DEEPSEEK_API_KEY"
                        
                        r1_client = OpenAI(
                            api_key=deepseek_api_key,
                            base_url="https://api.deepseek.com"
                        )
                        
                        # 构建分析请求 - 参考旧版本 nodes.py 中 R1 的 prompt 风格
                        analysis_prompt = f"""你是一个专业的旅行规划优化师。请对以下旅行规划问题进行深度分析和优化：

问题描述：{problem}

已收集的信息：
{json.dumps(context, ensure_ascii=False, indent=2) if context else '无额外上下文'}

你的任务：
1. **路线优化**：分析多段行程的最优顺序和连接方式
2. **时间安排**：每个目的地的合理停留时间
3. **预算分配**：根据各段行程的物价和景点密度分配预算
4. **风险评估**：识别天气、交通、时间等风险
5. **备选方案**：提供经济型、舒适型等不同方案

输出JSON格式（纯JSON，不要markdown代码块）：
{{
  "route_optimization": "路线优化建议",
  "time_arrangement": "时间安排建议",
  "budget_allocation": {{
    "城市名": 预算金额
  }},
  "risk_warnings": ["风险1", "风险2"],
  "alternative_plans": [
    {{
      "name": "方案名称",
      "description": "方案描述",
      "total_cost": 总费用,
      "pros": ["优点1"],
      "cons": ["缺点1"]
    }}
  ],
  "final_recommendation": "最终建议"
}}
"""
                        
                        response = r1_client.chat.completions.create(
                            model="deepseek-reasoner",  # 使用正确的模型名
                            messages=[{"role": "user", "content": analysis_prompt}],
                            max_tokens=4000
                        )
                        
                        result = response.choices[0].message.content
                        print(f"  R1分析完成，返回 {len(result)} 字符")
                        return result
                    except Exception as e:
                        print(f"  R1分析失败: {e}")
                        import traceback
                        traceback.print_exc()
                        return f"深度分析暂时不可用: {str(e)}"
                return call_r1_analysis
            
            tools.append(Tool(
                name=tool_def.name,
                description=tool_def.description,
                func=make_r1_tool()
            ))
    
    return tools


# 创建工具
try:
    tools = create_tools(retriever)
    if tools:
        tool_names = [t.name for t in tools]
        st.sidebar.success(f"✅ 已加载 {len(tools)} 个工具")
        # 显示工具列表以便调试
        with st.sidebar.expander("查看工具列表"):
            for t in tools:
                st.write(f"- {t.name}")
    else:
        st.sidebar.warning("⚠️ 未加载任何工具")
except Exception as e:
    st.sidebar.error(f"❌ 工具加载失败: {e}")
    tools = []


# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是智能旅游规划助手🗺️\n\n我可以帮您：\n- 📍 查询景点攻略和美食推荐\n- 🚆 查询火车票和航班信息\n- 🏨 推荐酒店和住宿\n- ☀️ 查询天气预报\n- 🗓️ 查询黄历吉日\n- 🚗 规划自驾路线\n\n请告诉我您的旅行需求吧！"}
    ]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 创建聊天历史管理
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output"
)


# ReAct Prompt模板
from datetime import datetime
current_date = datetime.now().strftime("%Y-%m-%d")
current_year = datetime.now().year

instructions = f"""你是一个专业的旅游规划助手。当前日期是{current_date}。

**完整旅行规划必须包含以下信息，请按顺序调用相应工具：**

1. **交通信息**：
   - 调用 train_query 查询火车票（必须）
   - 如果火车不便，调用 gaode_driving 查询自驾路线
   - 长途（>800km）可调用 flight_query 查询航班

2. **天气预报**：
   - 对每个目的地城市调用 gaode_weather 查询天气（必须）
   - 例如去徐州和青岛，需要分别查询两个城市的天气

3. **住宿推荐**：
   - 调用 gaode_hotel_search 搜索酒店（必须）
   - 根据预算选择关键词：预算紧张用"经济型酒店"，预算充足用"酒店"

4. **黄历吉日**：
   - 调用 lucky_day 查询出发日期的黄历信息

5. **复杂行程优化**（多城市、多段行程）：
   - 调用 r1_analysis 进行深度路线优化

**重要规则：**
- 日期必须使用{current_year}年，例如"{current_year}-12-18"
- 每个工具可以针对不同参数调用多次（如查询多个城市的天气）
- 用中文回答，提供详细的行程规划
"""

base_prompt_template = """
{instructions}

TOOLS:
------
You have access to the following tools:
{tools}

To use a tool, you MUST strictly follow this EXACT format:

Thought: Do I need to use a tool? Yes
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST strictly follow this EXACT format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

CRITICAL RULES:
1. For complete travel planning, you MUST call: train_query, gaode_weather (for each destination), gaode_hotel_search, lucky_day
2. You CAN call the same tool multiple times with DIFFERENT parameters (e.g. weather for different cities)
3. NEVER call the same tool with the SAME parameters twice
4. If a tool returns an error, explain the error to the user and move on
5. For multi-city complex trips, call r1_analysis to optimize the route
6. Use Chinese to respond to users
7. Provide detailed and complete travel plans

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""

# 创建Prompt
base_prompt = PromptTemplate.from_template(base_prompt_template)
prompt = base_prompt.partial(instructions=instructions)


# 创建LLM（使用Qwen3）
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7
    )

llm = get_llm()


# 创建ReAct Agent
agent = create_react_agent(llm, tools, prompt)

# 创建Agent执行器
max_iterations = 30  # 复杂行程需要更多迭代
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=False,  # 关闭详细日志，避免UI显示调试信息
    handle_parsing_errors=True,
    max_iterations=max_iterations
)


# 聊天输入框
if user_query := st.chat_input(placeholder="请输入您的旅行需求，例如：我想12月去杭州玩3天"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # 显示助手响应
    with st.chat_message("assistant"):
        # 隐藏调试信息，不显示工具调用过程
        # st_container = st.container()
        # st_cb = StreamlitCallbackHandler(st_container)
        
        # 配置回调（空配置，不显示执行过程）
        config = {}
        
        # ========== 预分析层：判断场景类型 ==========
        with st.spinner("🔍 正在分析您的旅行需求..."):
            pre_analysis = pre_analyze_query(user_query, llm)
        
        scenario_type = pre_analysis['scenario_type']
        needs_r1 = pre_analysis['needs_deep_analysis']
        extraction = pre_analysis['extraction']
        
        # 显示场景类型提示（隐藏技术细节）
        if scenario_type == 'multi_destination':
            st.info("🌍 检测到多目的地行程，将调用深度路线优化...")
        elif scenario_type == 'complex':
            st.info("🧠 检测到复杂场景（特殊需求/预算紧张），将进行深度分析...")
        else:
            st.info("✅ 正在为您规划旅行...")
        
        # ========== 根据场景类型选择不同的处理路径 ==========
        with st.spinner("正在规划您的旅行..."):
            try:
                if needs_r1:
                    # 复杂场景 / 多目的地：R1 主导
                    # 先调用 R1 进行深度分析和规划
                    print(f"\n🧠 R1 主导模式: {scenario_type}")
                    
                    # 构建 R1 分析的上下文
                    r1_context = {
                        "user_query": user_query,
                        "extraction": extraction,
                        "scenario_type": scenario_type,
                        "multi_dest_info": pre_analysis.get('multi_dest_info', {})
                    }
                    
                    # 构建增强的输入，强制 Agent 先调用 r1_analysis
                    enhanced_input = f"""用户查询: {user_query}

ℹ️ **重要提示**：系统检测到这是一个{'多目的地' if scenario_type == 'multi_destination' else '复杂'}场景。

请按以下顺序处理：
1. **首先调用 r1_analysis** 进行深度路线规划和优化，输入应包含完整的用户需求
2. 然后根据 R1 的建议，调用其他工具查询具体信息（火车票、天气、酒店等）
3. 最后综合所有信息生成完整的旅行规划

已提取的信息：
- 目的地: {extraction.get('destination', '未知')}
- 出发地: {extraction.get('origin', '未知')}
- 旅行天数: {extraction.get('travel_days', 0)}
- 预算: {extraction.get('budget', 0)}元
- 出发日期: {extraction.get('travel_date', '未知')}
- 偏好: {extraction.get('preferences', [])}
"""
                    response = agent_executor.invoke({"input": enhanced_input}, config=config)
                else:
                    # 简单场景：Qwen3 主导
                    print(f"\n✅ Qwen3 主导模式: {scenario_type}")
                    response = agent_executor.invoke({"input": user_query}, config=config)
                
                # 提取回答
                answer = response.get("output", "抱歉，我无法生成回答。")
                
                # 添加助手消息
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # 显示回答
                st.markdown(answer)
                
            except Exception as e:
                error_msg = f"抱歉，处理您的请求时出现错误：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# 侧边栏 - 显示当前配置
with st.sidebar:
    st.markdown("---")
    st.header("📊 当前配置")
    st.write(f"- 文档数量: {len(uploaded_files)}")
    st.write(f"- 工具数量: {len(tools)}")
    st.write(f"- 最大迭代: {max_iterations}")
    st.write(f"- 对话轮数: {len(st.session_state.messages)}")
