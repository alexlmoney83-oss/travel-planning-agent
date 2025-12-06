"""
LangGraph工作流节点实现
"""
from typing import Dict, Any
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from travel_agent.graph.state import TravelPlanState
from travel_agent.config.settings import (
    DASHSCOPE_API_KEY,
    QWEN3_MODEL,
    QWEN3_API_BASE,
    QWEN3_TEMPERATURE,
)
from travel_agent.config.prompts import (
    PLANNER_SYSTEM_PROMPT,
    SIMPLE_QUERY_PROMPT_TEMPLATE,
    SYNTHESIZER_PROMPT_TEMPLATE,
)

# Pydantic model for structured output
class TravelPlanExtraction(BaseModel):
    """Extracted travel plan information"""
    destination: str = Field(description="Destination city in Chinese")
    origin: str = Field(description="Origin city in Chinese")
    travel_days: int = Field(description="Number of travel days")
    budget: float = Field(description="Budget in yuan")
    travel_date: str = Field(description="Departure date in YYYY-MM-DD format")
    preferences: list[str] = Field(description="Travel preferences")
    needs_deep_analysis: bool = Field(default=False)
    tools_needed: list[str] = Field(default_factory=lambda: ["旅游攻略检索", "12306查询"])

# 初始化Qwen3 LLM（使用DashScope API）
qwen3_llm = ChatOpenAI(
    model=QWEN3_MODEL,
    api_key=DASHSCOPE_API_KEY,
    base_url=QWEN3_API_BASE,
    temperature=QWEN3_TEMPERATURE,
)

# Structured output LLM
try:
    qwen3_structured = qwen3_llm.with_structured_output(TravelPlanExtraction)
except Exception:
    # Fallback if structured output not supported
    qwen3_structured = None
    print("⚠️ Structured output not supported, using JSON parsing")

async def planner_node(state: TravelPlanState) -> Dict[str, Any]:
    """规划节点 - 分析用户需求"""
    from datetime import datetime
    
    print(f"\n{'='*60}")
    print("▶️ Planner 节点开始执行")
    print(f"State keys: {list(state.keys())}")
    
    # 立即返回一条提示消息，让用户知道系统在处理
    status_msg = AIMessage(content="🔎 正在分析您的旅行需求，请稍候…")
    
    # 优先从 user_query 读取；若为空，则从对话历史中取最后一条用户消息
    user_query = state.get("user_query", "") or ""
    if not user_query:
        msgs = state.get("messages") or []
        print(f"Messages count: {len(msgs)}")
        for i, m in enumerate(reversed(msgs)):
            print(f"Message {i}: type={type(m)}, isinstance(HumanMessage)={isinstance(m, HumanMessage)}")
            try:
                # LangChain message objects
                if isinstance(m, HumanMessage):
                    user_query = m.content or ""
                    print(f"Extracted from HumanMessage: {user_query[:50]}")
                    break
                # Dict format
                elif isinstance(m, dict):
                    print(f"Dict keys: {m.keys()}")
                    if m.get("type") == "human" or m.get("role") == "user":
                        user_query = m.get("content", "")
                        print(f"Extracted from dict: {user_query[:50]}")
                        break
                # Try to access as object with attributes
                elif hasattr(m, 'type') and m.type == 'human':
                    user_query = m.content
                    print(f"Extracted from object.type: {user_query[:50]}")
                    break
                elif hasattr(m, 'role') and m.role == 'user':
                    user_query = m.content
                    print(f"Extracted from object.role: {user_query[:50]}")
                    break
            except Exception as e:
                print(f"Error extracting from message {i}: {e}")
                continue
    
    print(f"提取到的用户查询: {user_query[:100] if user_query else '[空]'}")
    
    if not user_query:
        print("⚠️ 错误：未找到用户输入")
        return {
            "destination": "",
            "origin": "",
            "travel_days": 0,
            "budget": 0,
            "travel_date": "",
            "preferences": [],
            "needs_deep_analysis": False,
            "tools_needed": [],
        }
    
    # 动态获取当前日期
    today = datetime.now().strftime("%Y-%m-%d")
    dynamic_prompt = PLANNER_SYSTEM_PROMPT.replace("{{TODAY}}", today)
    print(f"当前日期: {today}")
    
    # 使用完整的对话历史，而不只是最后一条消息
    # 这样 LLM 可以看到之前的上下文
    conversation_messages = state.get("messages") or []
    messages = [
        SystemMessage(content=dynamic_prompt),
    ]
    # 添加所有历史消息（保留上下文）
    for msg in conversation_messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            messages.append(msg)
        elif isinstance(msg, dict):
            if msg.get("type") == "human" or msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("type") == "ai" or msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))
    
    # 优先尝试结构化输出
    if qwen3_structured is not None:
        try:
            extraction = await qwen3_structured.ainvoke(messages)
            result = {
                "destination": extraction.destination,
                "origin": extraction.origin,
                "travel_days": extraction.travel_days,
                "budget": extraction.budget,
                "travel_date": extraction.travel_date,
                "preferences": extraction.preferences,
                "needs_deep_analysis": extraction.needs_deep_analysis,
                "tools_needed": extraction.tools_needed,
            }
            # 如果上一轮需要 clarification，说明用户是在补充信息
            # 关键逻辑：只填充缺失的字段，不覆盖已有的字段
            if state.get("needs_clarification"):
                prev_destination = state.get("destination") or ""
                prev_origin = state.get("origin") or ""
                prev_days = state.get("travel_days") or 0
                prev_budget = state.get("budget") or 0
                prev_date = state.get("travel_date") or ""
                prev_prefs = state.get("preferences") or []
                
                print(f"  🔄 检测到 clarification 状态，合并上一轮信息")
                print(f"    上一轮: dest={prev_destination}, origin={prev_origin}")
                print(f"    这一轮: dest={result['destination']}, origin={result['origin']}")
                
                # 保留已有的字段，只填充缺失的
                result['destination'] = prev_destination or result['destination']
                result['origin'] = prev_origin or result['origin']
                result['travel_days'] = prev_days or result['travel_days']
                result['budget'] = prev_budget or result['budget']
                result['travel_date'] = prev_date or result['travel_date']
                result['preferences'] = prev_prefs or result['preferences']
                
                print(f"    合并后: dest={result['destination']}, origin={result['origin']}")
            
            print(f"\n✅ Planner 提取结果:")
            print(f"  目的地: {result['destination']}")
            print(f"  出发地: {result['origin']}")
            print(f"  旅行天数: {result['travel_days']}")
            print(f"  预算: {result['budget']}")
            print(f"  出发日期: {result['travel_date']}")
            print(f"  偏好: {result['preferences']}")
            print(f"{'='*60}\n")
            
            # ==== 智能判断查询模式 ====
            # 简单查询：只有目的地，没有旅行天数/预算/日期
            is_simple_query = (
                result['destination'] and 
                not result['travel_days'] and 
                not result['budget'] and 
                not result['travel_date']
            )
            
            if is_simple_query:
                print(f"  💡 检测到简单查询模式：用户只想了解景点")
                return {
                    **result,
                    "query_mode": "simple",
                    "needs_clarification": False,
                    "tools_needed": ["旅游攻略检索"],  # 只用RAG和高德POI
                    "messages": [status_msg]
                }
            
            # ==== 完整规划模式：检查关键信息是否缺失 ====
            print(f"  📋 检测到完整规划模式：需要收集完整信息")
            missing_fields = []
            if not result['destination']:
                missing_fields.append("目的地")
            # 如果有目的地但没有出发地，则需要询问（用于查询交通）
            if result['destination'] and not result['origin']:
                missing_fields.append("出发地")
            
            if missing_fields:
                clarification = f"请问您的{''.join(missing_fields)}是哪里？这样我才能为您查询具体的交通和行程信息。"
                return {
                    **result,
                    "query_mode": "full",
                    "needs_clarification": True,
                    "clarification_question": clarification,
                    "messages": [status_msg, AIMessage(content=clarification)],
                }
            
            # 信息完整，清除之前的 clarification 标记
            return {
                **result,
                "query_mode": "full",
                "needs_clarification": False,
                "clarification_question": None,
                "messages": [status_msg]
            }
        except Exception as e:
            print(f"⚠️ Structured output 失败: {e}，回退到 JSON 解析")
    
    response = await qwen3_llm.ainvoke(messages)
    
    try:
        # 处理markdown代码块包装的JSON
        content = response.content.strip()
        
        # 提取JSON代码块
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end]
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end]
        
        content = content.strip()
        
        # 如果还有多余文本，只取第一个完整的JSON对象
        if content.startswith("{"):
            # 找到JSON对象的结束位置
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
        
        # 解析JSON
        result = json.loads(content)
        return {
            "destination": result.get("destination", ""),
            "origin": result.get("origin", ""),
            "travel_days": result.get("travel_days", 0),
            "budget": result.get("budget", 0),
            "travel_date": result.get("travel_date", ""),
            "preferences": result.get("preferences", []),
            "needs_deep_analysis": result.get("needs_deep_analysis", False),
            "tools_needed": result.get("tools_needed", []),
        }
    except (json.JSONDecodeError, Exception) as e:
        print(f"\n{'='*60}")
        print(f"⚠️ 解析规划结果失败: {e}")
        print(f"\nLLM完整响应：")
        print(response.content)
        print(f"\n处理后的 content：")
        print(repr(content))
        print(f"{'='*60}\n")
        
        # 返回默认值，确保工作流继续
        return {
            "destination": "",
            "origin": "",
            "travel_days": 0,
            "budget": 0,
            "travel_date": "",
            "preferences": [],
            "needs_deep_analysis": False,
            "tools_needed": [],
        }


async def rag_search_node(state: TravelPlanState) -> Dict[str, Any]:
    """检索节点 - 智能检索相关信息（RAG + 高德地图）"""
    print(f"\n{'='*60}")
    print(f"🔍 开始执行 RAG_SEARCH_NODE")
    print(f"{'='*60}")
    
    from travel_agent.tools.rag_tool import get_rag_instance
    from travel_agent.tools.mcp_tools import get_mcp_manager
    
    destination = state.get("destination", "")
    preferences = state.get("preferences", [])
    
    status_msg = AIMessage(content=f"📚 正在检索{destination}的景点信息和旅游攻略（RAG + 高德地图）…")
    
    if not destination:
        print("⚠️ 未提供目的地，跳过 RAG 检索")
        return {"rag_results": "未提供目的地"}
    
    print(f"📚 步骤 1: 开始 RAG 知识库检索 - {destination}")
    try:
        rag = get_rag_instance()
        
        # 构建多个检索查询，提高召回率
        queries = [
            f"{destination}",  # 基本查询
            f"{destination} 景点",  # 景点信息
            f"{destination} 旅游",  # 旅游攻略
        ]
        
        # 根据偏好添加特定查询
        if any("老人" in p or "亲子" in p or "儿童" in p for p in preferences):
            queries.append(f"{destination} 亲子 老人适合")
        
        # 执行多个检索并合并结果
        all_results = []
        for query in queries[:2]:  # 限制检索次数避免过慢
            result = await rag.search(query, k=3)  # 每个查询返回3条
            if result and "未找到" not in result and "失败" not in result:
                all_results.append(f"[{query}]\n{result}")
        
        # 合并 RAG 结果
        rag_content = "\n\n---\n\n".join(all_results) if all_results else f"知识库中没有{destination}的攻略"
        print(f"✅ RAG 检索完成，共 {len(all_results)} 条结果")
    except Exception as e:
        rag_content = f"RAG检索失败: {str(e)}"
        print(f"❌ RAG 检索异常: {e}")
    
    # 同时调用高德地图搜索实时景点
    print(f"\n🗺️ 步骤 2: 开始高德地图 POI 搜索 - {destination}")
    gaode_pois = []
    try:
        manager = await get_mcp_manager()
        print(f"  ✅ MCP Manager 获取成功")
        
        # 搜索关键词
        search_keywords = [
            f"{destination} 景点",
            f"{destination} 旅游",
        ]
        
        if any("老人" in p or "亲子" in p or "儿童" in p for p in preferences):
            search_keywords.append(f"{destination} 亲子 公园")
        
        for keyword in search_keywords[:2]:  # 限制搜索次数
            try:
                # 尝试多种参数组合
                result = None
                param_combinations = [
                    {"keywords": keyword, "city": destination},
                    {"keyword": keyword, "city": destination},
                    {"keywords": keyword},
                ]
                
                for params in param_combinations:
                    try:
                        result = await manager.call_tool("Gaode Server", "maps_text_search", **params)
                        if result and "MCP error" not in str(result):
                            print(f"  ✅ {keyword} - 搜索成功")
                            print(f"  📦 高德返回类型: {type(result)}, 前500字符: {str(result)[:500]}")
                            gaode_pois.append(f"[高德地图 - {keyword}]\n{result}")
                            break
                    except Exception as ex:
                        print(f"  参数 {params} 失败: {ex}")
                        continue
            except Exception as e:
                print(f"  ⚠️ {keyword} 搜索失败: {e}")
        
        print(f"\n📊 高德 POI 搜索结果：共 {len(gaode_pois)} 条")
    except Exception as e:
        print(f"❌ 高德地图搜索异常: {e}")
        import traceback
        traceback.print_exc()
    
    # 步骤 3: 搜索酒店/民宿
    print(f"\n🏨 步骤 3: 开始高德地图酒店搜索 - {destination}")
    gaode_hotels = []
    budget = state.get("budget", 0)
    
    try:
        manager = await get_mcp_manager()
        
        # 根据预算确定搜索关键词
        hotel_keywords = []
        if budget and budget > 0:
            per_day_budget = budget / state.get("travel_days", 1) if state.get("travel_days") else budget
            if per_day_budget > 500:
                hotel_keywords.append(f"{destination} 高端酒店")
                hotel_keywords.append(f"{destination} 豪华酒店")
            elif per_day_budget > 300:
                hotel_keywords.append(f"{destination} 酒店")
                hotel_keywords.append(f"{destination} 品牌酒店")
            else:
                hotel_keywords.append(f"{destination} 经济型酒店")
                hotel_keywords.append(f"{destination} 民宿")
        else:
            hotel_keywords.append(f"{destination} 酒店")
            hotel_keywords.append(f"{destination} 民宿")
        
        # 根据偏好调整
        if any("亲子" in p or "儿童" in p for p in preferences):
            hotel_keywords.insert(0, f"{destination} 亲子酒店")
        
        print(f"  酒店搜索关键词: {hotel_keywords[:2]}")
        
        for keyword in hotel_keywords[:2]:  # 限制搜索次数
            try:
                result = None
                param_combinations = [
                    {"keywords": keyword, "city": destination},
                    {"keyword": keyword, "city": destination},
                    {"keywords": keyword},
                ]
                
                for params in param_combinations:
                    try:
                        result = await manager.call_tool("Gaode Server", "maps_text_search", **params)
                        if result and "MCP error" not in str(result):
                            print(f"  ✅ {keyword} - 搜索成功")
                            gaode_hotels.append(f"[高德地图 - {keyword}]\n{result}")
                            break
                    except Exception:
                        continue
            except Exception as e:
                print(f"  ⚠️ {keyword} 搜索失败: {e}")
        
        print(f"\n🏨 酒店搜索结果：共 {len(gaode_hotels)} 条")
    except Exception as e:
        print(f"❌ 酒店搜索异常: {e}")
    
    # 合并三个数据源
    combined_results = []
    
    if rag_content and "失败" not in rag_content:
        combined_results.append(f"## 📚 知识库攻略\n{rag_content}")
    
    if gaode_pois:
        combined_results.append(f"## 🗺️ 实时景点（高德地图）\n{'\n\n'.join(gaode_pois)}")
    
    # 酒店信息单独存储
    hotel_results = ""
    if gaode_hotels:
        hotel_results = f"## 🏨 酒店/民宿推荐（高德地图）\n{'\n\n'.join(gaode_hotels)}"
    
    print(f"\n📦 合并结果: RAG={bool(rag_content and '失败' not in rag_content)}, 高德POI={len(gaode_pois)}, 酒店={len(gaode_hotels)}")
    print(f"{'='*60}")
    print(f"✅ RAG_SEARCH_NODE 执行完成")
    print(f"{'='*60}\n")
    
    if combined_results:
        return {
            "rag_results": "\n\n".join(combined_results),
            "hotel_info": hotel_results,
            "messages": [status_msg]
        }
    else:
        return {
            "rag_results": f"{destination}景点信息检索失败，将使用 LLM 通用知识",
            "hotel_info": hotel_results,
            "messages": [status_msg]
        }


async def train_query_node(state: TravelPlanState) -> Dict[str, Any]:
    """交通方案查询节点 - 根据距离智能选择查询火车票或自驾路线"""
    from travel_agent.tools.mcp_tools import get_mcp_manager
    
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    travel_date = state.get("travel_date", "")
    
    status_msg = AIMessage(content=f"🚆 正在查询 {origin or '未知'} → {destination or '未知'} 的交通方案…")
    
    if not all([origin, destination, travel_date]):
        return {
            "train_info": {
                "error": "缺少必要信息",
                "origin": origin,
                "destination": destination,
                "date": travel_date
            },
            "messages": [status_msg]
        }
    
    try:
        manager = await get_mcp_manager()
        
        # 首先列出所有可用工具，找出正确的工具名
        try:
            tools = await manager.list_tools("12306 Server")
            print(f"\n🔧 12306 Server 可用工具: {tools}")
            
            # 尝试获取 get-tickets 工具的详细 schema
            try:
                server = manager.mcp_servers.get("12306 Server")
                if server:
                    tools_list = await server.list_tools()
                    for tool in tools_list:
                        if hasattr(tool, 'name') and tool.name == 'get-tickets':
                            print(f"\n📋 get-tickets 工具详细信息:")
                            if hasattr(tool, 'inputSchema'):
                                import json
                                print(json.dumps(tool.inputSchema, ensure_ascii=False, indent=2))
                            elif hasattr(tool, 'parameters'):
                                print(tool.parameters)
                            break
            except Exception as schema_err:
                print(f"⚠️ 无法获取工具 schema: {schema_err}")
        except Exception as e:
            print(f"\n⚠️ 无法获取工具列表: {e}")
        
        print(f"\n🚆 正在调用 MCP 12306: {origin} → {destination}, {travel_date}")
        
        # 先获取站点代码（必须！get-tickets 要求 station_code）
        from_code = None
        to_code = None
        
        try:
            print(f"  步骤 1: 获取站点代码")
            
            # 尝试 get-station-code-of-citys （按城市查询）
            try:
                result = await manager.call_tool(
                    "12306 Server",
                    "get-station-code-of-citys",
                    citys=f"{origin},{destination}"
                )
                print(f"  get-station-code-of-citys 结果: {result}")
                
                if result and "error" not in str(result).lower():
                    # 解析结果提取站点代码
                    try:
                        import json
                        codes_data = json.loads(result) if isinstance(result, str) else result
                        # 假设返回格式为 {'城市名': [{'station_name': 'xxx', 'station_code': 'xxx'}]}
                        if isinstance(codes_data, dict):
                            # 提取第一个站点代码
                            for city in [origin, destination]:
                                if city in codes_data and isinstance(codes_data[city], list) and len(codes_data[city]) > 0:
                                    code = codes_data[city][0].get('station_code') or codes_data[city][0].get('code')
                                    if city == origin:
                                        from_code = code
                                    else:
                                        to_code = code
                            
                            if from_code and to_code:
                                print(f"  ✅ 站点代码: {origin}={from_code}, {destination}={to_code}")
                    except Exception as parse_err:
                        print(f"  ⚠️ 解析站点代码失败: {parse_err}")
            except Exception as e:
                print(f"  ⚠️ get-station-code-of-citys 失败: {e}")
            
            # 如果上面失败，尝试 get-stations-code-in-city（单个城市）
            if not from_code or not to_code:
                for city, var_name in [(origin, "from_code"), (destination, "to_code")]:
                    if (var_name == "from_code" and from_code) or (var_name == "to_code" and to_code):
                        continue
                    
                    try:
                        result = await manager.call_tool(
                            "12306 Server",
                            "get-stations-code-in-city",
                            city=city
                        )
                        print(f"  get-stations-code-in-city({city}): {str(result)[:200]}")
                        
                        if result and "error" not in str(result).lower():
                            try:
                                import json
                                data = json.loads(result) if isinstance(result, str) else result
                                if isinstance(data, list) and len(data) > 0:
                                    code = data[0].get('station_code') or data[0].get('code')
                                    if var_name == "from_code":
                                        from_code = code
                                    else:
                                        to_code = code
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"  ⚠️ get-stations-code-in-city({city}) 失败: {e}")
        except Exception as e:
            print(f"  ❌ 站点代码查询异常: {e}")
        
        # 查询车次 - 必须使用 station_code
        print(f"  步骤 2: 查询车次")
        result = None
        
        if not from_code or not to_code:
            print(f"  ❌ 站点代码缺失，无法查询车票")
            result = f"MCP error: Unable to get station codes for {origin} and {destination}"
        else:
            # 使用正确的参数名称（根据 schema）
            try:
                print(f"  使用站点代码: fromStation={from_code}, toStation={to_code}, date={travel_date}")
                result = await manager.call_tool(
                    "12306 Server",
                    "get-tickets",
                    fromStation=from_code,
                    toStation=to_code,
                    date=travel_date
                )
                if result and "MCP error" not in str(result):
                    print(f"  ✅ 查询成功")
                else:
                    print(f"  ❌ 查询失败: {str(result)[:200]}")
            except Exception as e:
                print(f"  ❌ 查询异常: {e}")
                result = f"MCP error: {str(e)}"
        
        print(f"📦 MCP 返回结果类型: {type(result)}")
        print(f"📦 MCP 返回内容: {result[:500] if isinstance(result, str) else result}")
        
        # ========== 先查询自驾路线（不依赖12306结果）==========
        driving_result = None
        try:
            # 使用高德地图查询驾车距离
            print(f"\n🚗 步骤 3: 查询自驾路线信息")
            print(f"  DEBUG: origin={origin}, destination={destination}")
            
            # 先查看 maps_direction_driving 的参数 schema
            try:
                gaode_tools = await manager.list_tools("Gaode Server")
                print(f"  🔧 Gaode Server 可用工具: {gaode_tools}")
                
                server = manager.mcp_servers.get("Gaode Server")
                if server:
                    tools_list = await server.list_tools()
                    for tool in tools_list:
                        if hasattr(tool, 'name') and 'direction' in tool.name.lower() and 'driving' in tool.name.lower():
                            print(f"\n📋 {tool.name} 工具详细信息:")
                            if hasattr(tool, 'inputSchema'):
                                import json
                                print(json.dumps(tool.inputSchema, ensure_ascii=False, indent=2))
                            break
            except Exception as schema_err:
                print(f"  ⚠️ 无法获取工具 schema: {schema_err}")
            
            # 先将城市名转换为经纬度
            print(f"  步骤 3.1: 获取城市经纬度")
            origin_coords = None
            dest_coords = None
            
            try:
                # 查询出发地经纬度
                origin_geo = await manager.call_tool("Gaode Server", "maps_geo", address=origin)
                print(f"  {origin} 地理编码结果: {str(origin_geo)[:150]}")
                
                if origin_geo and "error" not in str(origin_geo).lower():
                    import json
                    geo_data = json.loads(origin_geo) if isinstance(origin_geo, str) else origin_geo
                    if isinstance(geo_data, dict):
                        # 高德API返回的是 {"return": [{...}]} 结构
                        if 'return' in geo_data and isinstance(geo_data['return'], list) and len(geo_data['return']) > 0:
                            location_data = geo_data['return'][0]
                            # 经纬度可能在 location 或 center 字段
                            origin_coords = location_data.get('location') or location_data.get('center')
                        else:
                            origin_coords = geo_data.get('location') or geo_data.get('geocodes', [{}])[0].get('location')
                        print(f"  ✅ {origin} 经纬度: {origin_coords}")
                
                # 查询目的地经纬度
                dest_geo = await manager.call_tool("Gaode Server", "maps_geo", address=destination)
                print(f"  {destination} 地理编码结果: {str(dest_geo)[:150]}")
                
                if dest_geo and "error" not in str(dest_geo).lower():
                    import json
                    geo_data = json.loads(dest_geo) if isinstance(dest_geo, str) else dest_geo
                    if isinstance(geo_data, dict):
                        # 高德API返回的是 {"return": [{...}]} 结构
                        if 'return' in geo_data and isinstance(geo_data['return'], list) and len(geo_data['return']) > 0:
                            location_data = geo_data['return'][0]
                            # 经纬度可能在 location 或 center 字段
                            dest_coords = location_data.get('location') or location_data.get('center')
                        else:
                            dest_coords = geo_data.get('location') or geo_data.get('geocodes', [{}])[0].get('location')
                        print(f"  ✅ {destination} 经纬度: {dest_coords}")
            except Exception as geo_err:
                print(f"  ⚠️ 地理编码失败: {geo_err}")
            
            if not origin_coords or not dest_coords:
                print(f"  ❌ 无法获取经纬度，跳过自驾路线查询")
                driving_result = None
            else:
                # 使用经纬度查询自驾路线
                print(f"  \n步骤 3.2: 查询自驾路线")
                try:
                    driving_result = await manager.call_tool(
                        "Gaode Server", 
                        "maps_direction_driving",
                        origin=origin_coords,
                        destination=dest_coords
                    )
                    print(f"  返回结果: {str(driving_result)[:200]}")
                    
                    # 检查是否有错误
                    result_str = str(driving_result).upper()
                    if "ERROR" in result_str or "INVALID" in result_str or "FAILED" in result_str:
                        print(f"  ❌ 自驾路线查询失败: {driving_result}")
                        driving_result = None
                    elif driving_result and "MCP error" not in str(driving_result):
                        print(f"  ✅ 自驾路线查询成功")
                        
                        # 解析距离，判断是否适合自驾
                        try:
                            import json
                            driving_data = json.loads(driving_result) if isinstance(driving_result, str) else driving_result
                            print(f"  📊 完整路线数据: {json.dumps(driving_data, ensure_ascii=False, indent=2)[:500]}")
                            
                            # 提取距离（可能在不同字段）
                            distance_km = None
                            if isinstance(driving_data, dict):
                                # 尝试多种可能的字段
                                distance_m = (driving_data.get('distance') or 
                                            driving_data.get('route', {}).get('distance') or
                                            driving_data.get('paths', [{}])[0].get('distance'))
                                if distance_m:
                                    distance_km = float(distance_m) / 1000
                                    print(f"  📏 距离: {distance_km:.1f} km")
                            
                            if distance_km and distance_km < 300:
                                print(f"  ✅ 距离 {distance_km:.0f}km < 300km，适合自驾，保留路线信息")
                            elif distance_km:
                                print(f"  ⚠️ 距离 {distance_km:.0f}km > 300km，不推荐自驾")
                                driving_result = None
                            else:
                                print(f"  ⚠️ 无法提取距离信息，保留原始数据")
                        except Exception as parse_err:
                            print(f"  ⚠️ 解析距离失败: {parse_err}")
                            import traceback
                            traceback.print_exc()
                except Exception as driving_err:
                    print(f"  ❌ 自驾路线查询异常: {driving_err}")
                    driving_result = None
        except Exception as e:
            print(f"  ⚠️ 自驾路线查询总体异常: {e}")
            import traceback
            traceback.print_exc()
        # ========== 解析12306结果 ==========
        # 检查是否是 MCP 错误消息
        if isinstance(result, str) and "MCP error" in result:
            print(f"⚠️ MCP 返回错误: {result}")
            return {
                "train_info": {"error": result},
                "driving_info": driving_result if driving_result and "MCP error" not in str(driving_result) else None,
                "messages": [status_msg]
            }
        
        # 尝试解析 JSON
        try:
            train_data = json.loads(result) if isinstance(result, str) else result
        except json.JSONDecodeError:
            # 12306返回的是纯文本格式，直接使用
            print(f"ℹ️ MCP 返回的是文本格式（非JSON），直接使用")
            train_data = {"tickets_text": result}
        
        # 检查是否是错误响应
        if isinstance(train_data, dict) and "error" in train_data:
            print(f"⚠️ MCP 返回错误: {train_data['error']}")
        else:
            print(f"✅ 12306 查询成功")
        
        return {
            "train_info": train_data,
            "driving_info": driving_result if driving_result and "MCP error" not in str(driving_result) else None,
            "messages": [status_msg]
        }
    except Exception as e:
        print(f"❌ 12306 调用异常: {e}")
        import traceback
        traceback.print_exc()
        return {
            "train_info": {
                "error": f"12306查询失败: {str(e)}",
                "departure": origin,
                "arrival": destination,
                "date": travel_date
            },
            "driving_info": None,
            "messages": [status_msg]
        }


async def lucky_day_query_node(state: TravelPlanState) -> Dict[str, Any]:
    """黄历吉日查询节点 - 查询出行日期的黄历宜忌"""
    from travel_agent.tools.mcp_tools import get_mcp_manager
    
    travel_date = state.get("travel_date", "")
    
    status_msg = AIMessage(content=f"🗓️ 正在查询出行日期的黄历宜忌…")
    
    if not travel_date:
        return {"lucky_day_info": None, "messages": [status_msg]}
    
    try:
        manager = await get_mcp_manager()
        print(f"\n🗓️ 调用八字黄历 API")
        print(f"  请求的日期: {travel_date}")
        
        # 先查看 getChineseCalendar 的参数 schema
        try:
            bazi_tools = await manager.list_tools("bazi Server")
            print(f"  🔧 bazi Server 可用工具: {bazi_tools}")
            
            server = manager.mcp_servers.get("bazi Server")
            if server:
                tools_list = await server.list_tools()
                for tool in tools_list:
                    if hasattr(tool, 'name') and 'calendar' in tool.name.lower():
                        print(f"\n📋 {tool.name} 工具详细信息:")
                        if hasattr(tool, 'inputSchema'):
                            import json
                            print(json.dumps(tool.inputSchema, ensure_ascii=False, indent=2))
                        break
        except Exception as schema_err:
            print(f"  ⚠️ 无法获取工具 schema: {schema_err}")
        
        # 根据 schema，需要 ISO 格式的时间字符串
        # 将 YYYY-MM-DD 转换为 ISO 格式: YYYY-MM-DDT12:00:00+08:00
        iso_datetime = f"{travel_date}T12:00:00+08:00"
        print(f"  🔄 转换为 ISO 格式: {iso_datetime}")
        
        result = None
        param_combinations = [
            {"solarDatetime": iso_datetime},
            {"date": travel_date},  # 备用
        ]
        
        for params in param_combinations:
            try:
                # 尝试调用 getChineseCalendar 获取黄历信息
                print(f"  尝试参数: {params}")
                result = await manager.call_tool("bazi Server", "getChineseCalendar", **params)
                if result and "MCP error" not in str(result):
                    print(f"  ✅ 黄历查询成功")
                    print(f"  📊 黄历数据前200字符: {str(result)[:200]}")
                    print(f"  📊 完整黄历数据: {result}")
                    
                    # 检查返回的日期是否匹配
                    try:
                        import json
                        check_data = json.loads(result) if isinstance(result, str) else result
                        print(f"  🗒️ 解析后的数据类型: {type(check_data)}")
                        print(f"  🗒️ 解析后的数据: {check_data}")
                        if isinstance(check_data, dict):
                            returned_date = check_data.get('公历', '') or check_data.get('date', '')
                            print(f"  📅 返回的公历日期: {returned_date}")
                            print(f"  🎯 请求的日期: {travel_date}")
                    except Exception as e:
                        print(f"  ⚠️ JSON解析失败: {e}")
                    break
            except Exception:
                continue
        
        if result and "MCP error" not in str(result):
            # 解析黄历数据
            try:
                import json
                calendar_data = json.loads(result) if isinstance(result, str) else result
                
                lucky_summary = f"🗓️ 出行日期: {travel_date}\n"
                
                if isinstance(calendar_data, dict):
                    # 提取关键信息
                    gongli = calendar_data.get('公历', '')
                    lunar_date = calendar_data.get('农历', '') or calendar_data.get('lunar_date', '') or calendar_data.get('lunarDate', '')
                    ganzhi = calendar_data.get('干支', '')
                    
                    # 宜和忌是逗号分隔的字符串
                    yi = calendar_data.get('宜', '')
                    ji = calendar_data.get('忌', '')
                    
                    # 构建详细的黄历信息
                    if gongli:
                        lucky_summary = f"🗓️ {gongli}\n"
                    else:
                        lucky_summary = f"🗓️ 出行日期: {travel_date}\n"
                    
                    if lunar_date:
                        lucky_summary += f"🌕 {lunar_date}\n"
                    
                    if ganzhi:
                        lucky_summary += f"🎯 干支: {ganzhi}\n"
                    
                    lucky_summary += "\n"
                    
                    # 处理“宜”
                    if yi:
                        yi_list = yi.split(',') if isinstance(yi, str) else yi
                        if '出行' in yi or '出门' in yi:
                            lucky_summary += f"✅ **吉：今日宜出行！**\n"
                        lucky_summary += f"👍 宜: {', '.join(yi_list[:8])}\n"
                    
                    # 处理“忌”
                    if ji:
                        ji_list = ji.split(',') if isinstance(ji, str) else ji
                        if '出行' in ji or '出门' in ji:
                            lucky_summary += f"\n⚠️ **忌：今日忌出行，如无法调整请多注意安全**\n"
                        lucky_summary += f"⛔ 忌: {', '.join(ji_list[:8])}\n"
                    
                    # 如果宜和忌都没有出行，给出中性提示
                    if yi and ji:
                        if '出行' not in yi and '出门' not in yi and '出行' not in ji and '出门' not in ji:
                            lucky_summary += f"\n🟡 今日对出行无特别宜忌，可正常安排\n"
                    
                    # 添加说明
                    lucky_summary += f"\n📍 注：黄历仅供参考，具体行程请根据实际情况安排"
                    
                    return {"lucky_day_info": lucky_summary, "messages": [status_msg]}
                else:
                    return {"lucky_day_info": str(result)[:300], "messages": [status_msg]}
            except Exception as parse_err:
                print(f"  ⚠️ 黄历数据解析失败: {parse_err}")
                return {"lucky_day_info": str(result)[:300], "messages": [status_msg]}
        else:
            return {"lucky_day_info": None, "messages": [status_msg]}
    except Exception as e:
        print(f"❌ 黄历查询异常: {e}")
        return {"lucky_day_info": None, "messages": [status_msg]}


async def weather_query_node(state: TravelPlanState) -> Dict[str, Any]:
    """天气查询节点 - 查询旅行期间的天气预报"""
    from travel_agent.tools.mcp_tools import get_mcp_manager
    from datetime import datetime, timedelta
    
    destination = state.get("destination", "")
    travel_days = state.get("travel_days", 1)
    travel_date = state.get("travel_date", "")
    
    status_msg = AIMessage(content=f"☀️ 正在查询{destination}的天气情况（{travel_days}天）…")
    
    if not destination:
        return {"weather_info": {"error": "未提供目的地"}, "messages": [status_msg]}
    
    try:
        manager = await get_mcp_manager()
        print(f"\n☀️ 调用高德地图天气 API: {destination}，旅行天数: {travel_days}")
        
        # 尝试多种参数组合查询天气
        result = None
        param_combinations = [
            {"city": destination},
            {"address": destination},
            {"location": destination},
        ]
        
        for params in param_combinations:
            try:
                result = await manager.call_tool("Gaode Server", "maps_weather", **params)
                if result and "MCP error" not in str(result):
                    print(f"  ✅ 天气查询成功")
                    print(f"  📊 天气数据前300字符: {str(result)[:300]}")
                    break
            except Exception:
                continue
        
        # 构建旅行期间的天气描述
        if result and "MCP error" not in str(result):
            # 尝试解析天气数据，提取多日预报
            weather_summary = f"目的地: {destination}\n"
            
            # 计算旅行日期范围
            travel_dates = []
            if travel_date:
                try:
                    start_date = datetime.strptime(travel_date, "%Y-%m-%d")
                    travel_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(travel_days)]
                    print(f"  📅 旅行日期: {', '.join(travel_dates)}")
                except Exception as date_err:
                    print(f"  ⚠️ 日期解析失败: {date_err}")
            
            try:
                import json
                weather_data = json.loads(result) if isinstance(result, str) else result
                
                # 高德天气 API 可能返回 forecasts 字段
                if isinstance(weather_data, dict):
                    # 当前天气（仅当今天就出发时显示）
                    today = datetime.now().strftime("%Y-%m-%d")
                    if travel_date == today:
                        if 'weather' in weather_data:
                            weather_summary += f"当前天气: {weather_data.get('weather', '未知')}\n"
                        if 'temperature' in weather_data:
                            weather_summary += f"当前温度: {weather_data.get('temperature', '未知')}°C\n"
                    
                    # 多日预报：筛选旅行日期范围的天气
                    if 'forecasts' in weather_data and isinstance(weather_data['forecasts'], list):
                        all_forecasts = weather_data['forecasts']
                        print(f"  📊 高德返回 {len(all_forecasts)} 天预报数据")
                        
                        # 提取预报的日期范围
                        forecast_dates = [f.get('date', '') for f in all_forecasts if f.get('date')]
                        if forecast_dates:
                            print(f"  📊 预报覆盖日期: {forecast_dates[0]} ~ {forecast_dates[-1]}")
                        
                        matched_forecasts = []
                        missing_dates = []
                        
                        # 如果有旅行日期，精确匹配
                        if travel_dates:
                            for travel_date_str in travel_dates:
                                found = False
                                for forecast in all_forecasts:
                                    forecast_date = forecast.get('date', '')
                                    if forecast_date == travel_date_str:
                                        matched_forecasts.append(forecast)
                                        found = True
                                        break
                                if not found:
                                    missing_dates.append(travel_date_str)
                        else:
                            # 如果没有日期，直接取前 N 天
                            matched_forecasts = all_forecasts[:travel_days]
                        
                        if matched_forecasts:
                            weather_summary += f"\n旅行期间天气预报（{len(matched_forecasts)}天）：\n"
                            for forecast in matched_forecasts:
                                date = forecast.get('date', '未知')
                                weather = forecast.get('dayweather') or forecast.get('weather', '未知')
                                temp_high = forecast.get('daytemp') or forecast.get('high', '')
                                temp_low = forecast.get('nighttemp') or forecast.get('low', '')
                                wind = forecast.get('daywind') or forecast.get('wind', '')
                                weather_summary += f"  {date}: {weather}, {temp_low}-{temp_high}°C"
                                if wind:
                                    weather_summary += f", {wind}"
                                weather_summary += "\n"
                        
                        # 如果有缺失的日期，给出提示
                        if missing_dates:
                            weather_summary += f"\n⚠️ 注意：{', '.join(missing_dates)} 的天气预报暂时不可用（超出高德地图 {len(all_forecasts)} 天预报范围）\n"
                            weather_summary += f"📍 建议：出行前 1-2 天再查看实时天气预报\n"
                        
                        if not matched_forecasts:
                            weather_summary += f"\n⚠️ 旅行日期（{', '.join(travel_dates)}）超出当前天气预报范围\n"
                            weather_summary += f"📍 高德地图目前只提供 {len(all_forecasts)} 天内的预报，建议出行前再查询\n"
                    
                    # 如果没有 forecasts，直接使用原始数据
                    if 'forecasts' not in weather_data:
                        weather_summary += f"\n详细信息: {str(weather_data)[:200]}"
                else:
                    weather_summary += str(result)[:300]
            except Exception as parse_err:
                print(f"  ⚠️ 天气数据解析失败: {parse_err}，使用原始数据")
                weather_summary = str(result)
            
            return {
                "weather_info": weather_summary,
                "messages": [status_msg]
            }
        else:
            return {
                "weather_info": f"{destination}天气查询失败，请出行前查看天气预报",
                "messages": [status_msg]
            }
    except Exception as e:
        print(f"❌ 天气查询异常: {e}")
        return {
            "weather_info": {
                "city": destination,
                "error": str(e)
            },
            "messages": [status_msg]
        }


async def deep_analysis_node(state: TravelPlanState) -> Dict[str, Any]:
    """深度分析节点"""
    from travel_agent.tools.r1_tool import get_r1_instance
    
    status_msg = AIMessage(content="🧠 DeepSeek R1 正在进行深度分析与优化，这可能需要 10-20 秒…")
    
    destination = state.get("destination", "未知")
    origin = state.get("origin", "未知")
    travel_days = state.get("travel_days", 0)
    budget = state.get("budget", 0)
    preferences = state.get("preferences", [])
    
    problem = f"""
    优化从{origin}到{destination}的{travel_days}天旅行方案。
    
    约束条件：
    - 预算限制: {budget}元
    - 旅行偏好: {', '.join(preferences) if preferences else '无特殊偏好'}
    - 已获取的基础信息：
      * 火车票信息: {state.get('train_info', {})}
      * 旅游攻略: {state.get('rag_results', '无')}
      * 天气信息: {state.get('weather_info', {})}
    
    请深度分析并提供：
    1. 预算分配优化建议
    2. 时间安排优化
    3. 风险评估和应对方案
    4. 多种可选方案对比
    """
    
    context = {
        "destination": destination,
        "origin": origin,
        "travel_days": travel_days,
        "budget": budget,
        "preferences": preferences,
        "train_info": state.get("train_info"),
        "rag_results": state.get("rag_results"),
        "weather_info": state.get("weather_info")
    }
    
    try:
        r1 = get_r1_instance()
        result = await r1.analyze(problem, context)
        
        # 尝试解析JSON，但如果失败则使用原始文本
        suggestions = []
        try:
            analysis = json.loads(result) if isinstance(result, str) else result
            if isinstance(analysis, dict):
                suggestions = analysis.get("suggestions", [])
        except json.JSONDecodeError:
            # R1可能返回非JSON格式的深度分析文本，这也是有效的
            print(f"⚠️ R1返回非JSON格式，使用原始文本")
        
        return {
            "reasoning_chain": result,
            "optimization_suggestions": suggestions,
            "messages": [status_msg]
        }
    except Exception as e:
        print(f"❌ R1分析异常: {e}")
        return {
            "reasoning_chain": f"R1分析失败: {str(e)}",
            "optimization_suggestions": [],
            "messages": [status_msg]
        }


async def synthesizer_node(state: TravelPlanState) -> Dict[str, Any]:
    """整合节点 - 生成最终方案"""
    status_msg = AIMessage(content="✨ 正在整合所有信息，为您生成专属旅行方案…")
    
    # 从 state 中提取结构化字段，重新构建用户需求
    # 避免使用最后一条消息（可能只是补充信息的简短回复）
    destination = state.get("destination", "")
    origin = state.get("origin", "")
    travel_days = state.get("travel_days", 0)
    budget = state.get("budget", 0)
    travel_date = state.get("travel_date", "")
    preferences = state.get("preferences", [])
    
    # 构建完整的用户需求描述
    user_query_parts = []
    if origin:
        user_query_parts.append(f"从{origin}")
    if destination:
        user_query_parts.append(f"去{destination}旅游")
    if travel_days:
        user_query_parts.append(f"{travel_days}天")
    if budget:
        user_query_parts.append(f"预算{budget}元")
    if travel_date:
        user_query_parts.append(f"出发日期{travel_date}")
    if preferences:
        user_query_parts.append(f"特殊需求: {', '.join(preferences)}")
    
    user_query = "，".join(user_query_parts)
    
    print(f"\n✨ Synthesizer 节点")
    print(f"  重构的用户需求: {user_query}")
    print(f"  目的地: {destination}")
    
    # 根据查询模式选择不同prompt
    query_mode = state.get("query_mode", "full")
    print(f"  查询模式: {query_mode}")
    
    if query_mode == "simple":
        # 简单查询模式：只显示景点信息
        prompt = SIMPLE_QUERY_PROMPT_TEMPLATE.format(
            destination=destination,
            rag_results=state.get("rag_results", "未找到相关信息")
        )
    else:
        # 完整规划模式：显示完整旅行方案
        # 提取 train_info 的完整文本
        train_info_raw = state.get("train_info", {})
        if isinstance(train_info_raw, dict):
            if "tickets_text" in train_info_raw:
                train_info_text = train_info_raw["tickets_text"]
            elif "error" in train_info_raw:
                train_info_text = f"错误: {train_info_raw['error']}"
            else:
                train_info_text = str(train_info_raw)
        else:
            train_info_text = str(train_info_raw)
        
        print(f"  train_info 长度: {len(train_info_text)} 字符")
        print(f"  train_info 前300字符: {train_info_text[:300]}")
        
        prompt = SYNTHESIZER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            rag_results=state.get("rag_results", ""),
            hotel_info=state.get("hotel_info", "未查询到酒店信息"),
            train_info=train_info_text,
            driving_info=state.get("driving_info", "未查询到自驾路线（可能距离过远）"),
            lucky_day_info=state.get("lucky_day_info", ""),
            weather_info=state.get("weather_info", {})
        )
    
    response = await qwen3_llm.ainvoke([HumanMessage(content=prompt)])
    
    # 返回 messages 格式供 Chat API 使用
    return {
        "travel_plan": response.content,
        "messages": [AIMessage(content=response.content)],
    }
