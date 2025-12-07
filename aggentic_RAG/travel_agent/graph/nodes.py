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
    REACT_THOUGHT_PROMPT,
    REACT_OBSERVATION_PROMPT,
)

# Pydantic model for structured output
class TravelPlanExtraction(BaseModel):
    """提取的旅行计划信息"""
    destination: str = Field(description="Destination city in Chinese")
    origin: str = Field(description="Origin city in Chinese")
    travel_days: int = Field(description="Number of travel days")
    budget: float = Field(description="Budget in yuan")
    travel_date: str = Field(description="Departure date in YYYY-MM-DD format")
    preferences: list[str] = Field(description="Travel preferences")
    needs_deep_analysis: bool = Field(default=False)
    tools_needed: list[str] = Field(default_factory=lambda: ["旅游攻略检索", "12306查询"])
    is_appending: bool = Field(default=False, description="Whether this is appending to existing plan")

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

# ========== 格式化辅助函数 ==========

def format_travel_segments(segments: list) -> str:
    """格式化行程段展示，支持返程段识别"""
    if not segments:
        return "无多段行程"
    
    result = []
    segment_index = 1
    
    for i, seg in enumerate(segments):
        origin = seg.get('origin', '未知')
        dest = seg.get('destination', '未知')
        days = seg.get('days', 0)
        date = seg.get('date_start', '')
        is_return = seg.get('is_return', False)
        
        if is_return:
            # 返程段特殊处理
            result.append(f"  返程：{origin} → {dest}")
            if date:
                result.append(f"    返程日期：{date}")
            result.append(f"    提示：当天返回，请预留充裕时间")
        else:
            # 普通行程段
            result.append(f"  第{segment_index}段：{origin} → {dest}")
            result.append(f"    旅行时间：{days}天")
            if date:
                result.append(f"    出发日期：{date}")
            segment_index += 1
    
    return "\n".join(result)

def format_budget_allocation(budget_alloc: dict) -> str:
    """格式化预算分配展示"""
    if not budget_alloc:
        return "未分配预算"
    
    result = []
    total = sum(budget_alloc.values())
    for city, amount in budget_alloc.items():
        percentage = (amount / total * 100) if total > 0 else 0
        result.append(f"  {city}：{amount}元 ({percentage:.1f}%)")
    
    return "\n".join(result)

def format_risk_warnings(warnings: list) -> str:
    """格式化风险警告展示"""
    if not warnings:
        return "无明显风险"
    
    result = []
    for i, warning in enumerate(warnings, 1):
        result.append(f"  {i}. ⚠️ {warning}")
    
    return "\n".join(result)

def format_alternative_plans(plans: list) -> str:
    """格式化替代方案展示"""
    if not plans:
        return "无替代方案"
    
    result = []
    for plan in plans:
        name = plan.get('name', '未命名方案')
        desc = plan.get('description', '')
        cost = plan.get('total_cost', 0)
        pros = plan.get('pros', [])
        cons = plan.get('cons', [])
        
        result.append(f"\n● **{name}**（总计约{cost}元）")
        if desc:
            result.append(f"  {desc}")
        if pros:
            result.append(f"  ✅ 优点：{', '.join(pros)}")
        if cons:
            result.append(f"  ⚠️ 缺点：{', '.join(cons)}")
    
    return "\n".join(result)

def format_value_comparison(comparisons: list) -> str:
    """格式化性价比对比展示"""
    if not comparisons:
        return "无对比数据"
    
    result = []
    for comp in comparisons:
        seg_idx = comp.get('segment', 0)
        dest = comp.get('destination', '未知')
        score = comp.get('value_score', 'N/A')
        highlights = comp.get('highlights', [])
        concerns = comp.get('concerns', [])
        
        result.append(f"\n  段{seg_idx + 1} - {dest} (性价比: {score})")
        if highlights:
            result.append(f"    🌟 亮点：{', '.join(highlights)}")
        if concerns:
            result.append(f"    ⚠️ 关注：{', '.join(concerns)}")
    
    return "\n".join(result)

# ========== 业务逻辑函数 ==========

def detect_multi_destination(user_query: str, extraction: dict) -> dict:
    """检测是否为多目的地场景（排除往返/回程误判）
    
    Args:
        user_query: 用户原始查询
        extraction: Planner提取的结果
    
    Returns:
        dict: {
            'is_multi_destination': bool,
            'detected_keywords': List[str],
            'raw_destination_text': str
        }
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
    
    # === 3) 目的地字段中包含多个城市（逗号/顿号分隔） ===
    destination = extraction.get('destination', '') or ''
    origin = extraction.get('origin', '') or ''
    norm = destination.replace(',', '，').replace('、', '，')
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
            
            # ==== 检测多目的地场景 ====
            # 注意：只在确实检测到多目的地关键词时才强制调用R1
            # 避免 Qwen3 误判普通单目的地为 needs_deep_analysis=true
            
            # ==== 禁用对话追加功能 ====
            # 如果检测到目的地变化，直接清空开始新查询
            prev_destination = state.get("destination", "")
            
            # 如果目的地变化，清空所有历史，开始全新查询
            if prev_destination and prev_destination != result.get('destination'):
                print(f"  🔄 目的地变化: {prev_destination} → {result.get('destination')}")
                print(f"  ⚒️ 对话追加功能已禁用，清空历史开始新查询")
                
                # 清空所有历史数据
                result['rag_results'] = None
                result['train_info'] = None
                result['driving_info'] = None
                result['flight_info'] = None
                result['hotel_info'] = None
                result['weather_info'] = None
                result['lucky_day_info'] = None
                result['travel_segments'] = None
                result['r1_plan'] = None
                result['iteration_count'] = 0
                result['is_complete'] = False
                result['should_continue'] = True
            
            # 正常单次查询的多目的地检测
            multi_dest_detection = detect_multi_destination(user_query, result)
            if multi_dest_detection.get('is_multi_destination', False):
                print(f"  🌍 检测到多目的地场景！")
                print(f"    关键词: {multi_dest_detection.get('detected_keywords', [])}")
                print(f"    原始目的地文本: {multi_dest_detection.get('raw_destination_text', '')}")
                # 只有真正检测到关键词时才强制设置
                result['needs_deep_analysis'] = True
                result['scenario_type'] = 'multi_destination'
                result['raw_destination_text'] = multi_dest_detection.get('raw_destination_text', result['destination'])
            else:
                # 单目的地：保持 Qwen3 原始判断，但强制设置 needs_deep_analysis=False
                # 避免 Qwen3 误判普通场景为复杂场景
                if result.get('needs_deep_analysis', False):
                    print(f"  ⚠️ Qwen3 设置 needs_deep_analysis=True，但未检测到多目的地关键词，强制设为False")
                result['needs_deep_analysis'] = False
                result['scenario_type'] = 'simple'
            
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
            
            # ==== 检测目的地是否变化，清空旧数据 ====
            # 注意：如果是追加场景，不清空历史数据，保留给R1分析
            destination_changed = prev_destination and prev_destination != result['destination']
            
            if destination_changed and not is_appending:
                print(f"  🔄 检测到目的地变化: {prev_destination} → {result['destination']}")
                print(f"  🧹 清空旧的查询结果...")
                # 清空所有查询结果
                result['rag_results'] = None
                result['train_info'] = None
                result['driving_info'] = None
                result['flight_info'] = None
                result['hotel_info'] = None
                result['weather_info'] = None
                result['lucky_day_info'] = None
                # 重置迭代计数
                result['iteration_count'] = 0
                result['is_complete'] = False
                result['should_continue'] = True
            elif is_appending:
                print(f"  📦 追加场景：保留历史数据 ({prev_destination}) 供 R1 分析")
            
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
        
        # 查询车次 - 优先使用 station_code，如果失败则尝试城市名
        print(f"  步骤 2: 查询车次")
        result = None
        
        if from_code and to_code:
            # 使用站点代码查询（最佳方式）
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
                    print(f"  ✅ 使用站点代码查询成功")
                else:
                    print(f"  ⚠️ 站点代码查询失败: {str(result)[:200]}")
                    result = None  # 重置结果，尝试备选方案
            except Exception as e:
                print(f"  ⚠️ 站点代码查询异常: {e}")
                result = None
        
        # 如果站点代码查询失败，尝试使用城市名直接查询（备选方案）
        if not result:
            print(f"  步骤 2.1: 尝试使用城市名直接查询")
            try:
                # 尝试多种参数组合
                param_combinations = [
                    {"fromStation": origin, "toStation": destination, "date": travel_date},
                    {"from": origin, "to": destination, "date": travel_date},
                    {"departure": origin, "arrival": destination, "date": travel_date},
                ]
                
                for params in param_combinations:
                    try:
                        print(f"  尝试参数: {params}")
                        result = await manager.call_tool(
                            "12306 Server",
                            "get-tickets",
                            **params
                        )
                        if result and "MCP error" not in str(result) and "error" not in str(result).lower():
                            print(f"  ✅ 使用城市名查询成功")
                            break
                        else:
                            print(f"  ⚠️ 参数 {list(params.keys())} 失败")
                            result = None
                    except Exception as param_err:
                        print(f"  ⚠️ 参数 {list(params.keys())} 异常: {param_err}")
                        continue
            except Exception as e:
                print(f"  ❌ 城市名查询异常: {e}")
        
        # 如果所有尝试都失败，返回详细错误信息
        if not result:
            print(f"  ❌ 所有查询方式都失败")
            result = {
                "error": "无法查询火车票信息",
                "reason": f"站点代码查询失败（fromStation={from_code or '空'}/toStation={to_code or '空'}）",
                "from_city": origin,
                "to_city": destination,
                "date": travel_date,
                "suggestion": f"请直接访问 12306 官网（www.12306.cn）或 APP 查询 {origin} 到 {destination} 的车次"
            }
        
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
                        
                        # 解析距离，保留所有数据并添加警告（不丢弃数据）
                        try:
                            import json
                            driving_data = json.loads(driving_result) if isinstance(driving_result, str) else driving_result
                            
                            # 🔍 详细调试：打印完整数据结构
                            print(f"\n  🔍 高德自驾返回数据结构调试:")
                            print(f"  数据类型: {type(driving_data)}")
                            
                            # 高德可能返回数组格式
                            if isinstance(driving_data, list):
                                print(f"  数组长度: {len(driving_data)}")
                                print(f"  第一个元素: {driving_data[0] if driving_data else 'N/A'}")
                            elif isinstance(driving_data, dict):
                                print(f"  顶层字段: {list(driving_data.keys())}")
                            
                            print(f"  完整数据 (JSON): {json.dumps(driving_data, ensure_ascii=False, indent=2)[:1000]}")
                            
                            # 提取距离（支持数组和dict格式）
                            distance_km = None
                            
                            # 方法0: 如果是数组，累加所有 distance
                            if isinstance(driving_data, list):
                                print(f"\n  🔎 检测到数组格式，累加所有 distance:")
                                total_distance_m = 0
                                for i, segment in enumerate(driving_data):
                                    seg_dist = segment.get('distance')
                                    if seg_dist:
                                        try:
                                            total_distance_m += float(seg_dist)
                                            print(f"    [{i}] distance = {seg_dist}m")
                                        except (ValueError, TypeError):
                                            pass
                                if total_distance_m > 0:
                                    distance_km = total_distance_m / 1000
                                    print(f"  ✅ 累计距离: {distance_km:.1f} km")
                            elif isinstance(driving_data, dict):
                                # 尝试多种可能的字段
                                print(f"\n  🔎 尝试提取距离:")
                                
                                # 方法1: 直接 distance
                                distance_m = driving_data.get('distance')
                                print(f"    driving_data.get('distance') = {distance_m}")
                                
                                # 方法2: route.distance
                                if not distance_m and 'route' in driving_data:
                                    distance_m = driving_data.get('route', {}).get('distance')
                                    print(f"    route.distance = {distance_m}")
                                
                                # 方法3: paths[0].distance
                                if not distance_m and 'paths' in driving_data:
                                    paths = driving_data.get('paths', [])
                                    if paths and len(paths) > 0:
                                        distance_m = paths[0].get('distance')
                                        print(f"    paths[0].distance = {distance_m}")
                                
                                # 方法4: return.paths[0].distance (高德MCP可能用 return 包裹)
                                if not distance_m and 'return' in driving_data:
                                    ret_data = driving_data.get('return', {})
                                    if isinstance(ret_data, dict) and 'route' in ret_data:
                                        distance_m = ret_data.get('route', {}).get('distance')
                                        print(f"    return.route.distance = {distance_m}")
                                    elif isinstance(ret_data, dict) and 'paths' in ret_data:
                                        paths = ret_data.get('paths', [])
                                        if paths and len(paths) > 0:
                                            distance_m = paths[0].get('distance')
                                            print(f"    return.paths[0].distance = {distance_m}")
                                
                                if distance_m:
                                    try:
                                        distance_km = float(distance_m) / 1000
                                        print(f"  ✅ 成功提取距离: {distance_km:.1f} km")
                                    except (ValueError, TypeError) as e:
                                        print(f"  ⚠️ 距离转换失败: {distance_m}, 错误: {e}")
                                else:
                                    print(f"  ❌ 未找到距离字段")
                            
                            # 根据距离添加警告，但始终保留数据
                            if distance_km:
                                if distance_km < 300:
                                    print(f"  ✅ 距离 {distance_km:.0f}km < 300km，适合自驾")
                                    # 保留原始数据，不需警告
                                    driving_result = {
                                        "data": driving_data,
                                        "distance_km": distance_km,
                                        "suitable": True
                                    }
                                elif distance_km < 500:
                                    print(f"  ⚠️ 距离 {distance_km:.0f}km (300-500km)，自驾较累，但可行")
                                    driving_result = {
                                        "data": driving_data,
                                        "distance_km": distance_km,
                                        "warning": f"距离较远（{distance_km:.0f}km），自驾需要 {int(distance_km/80)+1} 小时以上，请考虑体力和时间",
                                        "suitable": False
                                    }
                                else:  # > 500km
                                    print(f"  🚗✈️ 距离 {distance_km:.0f}km > 500km，强烈建议高铁/飞机")
                                    driving_result = {
                                        "data": driving_data,
                                        "distance_km": distance_km,
                                        "warning": f"距离很远（{distance_km:.0f}km），自驾需要 {int(distance_km/80)+1} 小时以上，强烈建议选择高铁或飞机",
                                        "suitable": False
                                    }
                            else:
                                print(f"  ⚠️ 无法提取距离信息，保留原始数据")
                                # 仍然保留数据，只是没有距离信息
                                driving_result = {
                                    "data": driving_data,
                                    "warning": "无法提取距离信息，请自行判断是否适合自驾"
                                }
                        except Exception as parse_err:
                            print(f"  ⚠️ 解析距离失败: {parse_err}")
                            import traceback
                            traceback.print_exc()
                            # 即使解析失败，也保留原始数据
                            if driving_result:
                                driving_result = {
                                    "data": driving_result,
                                    "warning": "数据解析失败，请自行判断"
                                }
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


async def r1_strategy_node(state: TravelPlanState) -> Dict[str, Any]:
    """
R1战略规划节点 - 分解多段行程并制定查询计划
    
    这是R1的第一次介入，负责：
    1. 理解用户的完整意图（识别多段行程）
    2. 分解旅行段（起点、终点、天数、日期）
    3. 分配预算给每一段
    4. 制定详细的查询计划（工具、参数、顺序）
    """
    from travel_agent.tools.r1_tool import get_r1_instance
    from datetime import datetime, timedelta
    
    print(f"\n{'='*60}")
    print("▶️ R1 Strategy 节点开始执行")
    
    user_query = state.get('user_query', '')
    scenario_type = state.get('scenario_type', 'multi_destination')
    scenario_label = "单目的地复杂行程" if scenario_type != "multi_destination" else "多段行程"
    status_msg = AIMessage(content=f"🧐 正在智能分析您的{scenario_label}，规划最佳路线…")
    
    # 收集Qwen3提取的初步信息
    destination = state.get('destination', '')
    origin = state.get('origin', '')
    travel_days = state.get('travel_days', 0)
    budget = state.get('budget', 0)
    travel_date = state.get('travel_date', '')
    preferences = state.get('preferences', [])
    raw_destination_text = state.get('raw_destination_text', destination)
    
    print(f"  场景类型: {scenario_type}")
    print(f"  用户查询: {user_query[:80]}...")
    print(f"  Qwen3提取的目的地: {destination}")
    print(f"  原始目的地文本: {raw_destination_text}")
    
    # 提取对话历史（只提取用户消息）
    conversation_messages = state.get('messages', [])
    user_messages = []
    for msg in conversation_messages:
        if isinstance(msg, HumanMessage):
            user_messages.append(msg.content)
        elif isinstance(msg, dict) and (msg.get('type') == 'human' or msg.get('role') == 'user'):
            user_messages.append(msg.get('content', ''))
    
    conversation_context = ""
    if len(user_messages) > 1:
        conversation_context = f"""

💬 **对话历史**（用户的多轮查询）：
{chr(10).join([f"{i+1}. {msg}" for i, msg in enumerate(user_messages)])}

⚠️ 请根据完整的对话历史理解用户意图，分解出完整的多段行程。
"""
        print(f"  对话历史: {len(user_messages)}轮")
    
    problem = f"""
用户的旅行需求：
最新查询：{user_query}
{conversation_context}

Qwen3已提取的初步信息：
- 目的地（可能是多个）：{raw_destination_text}
- 出发地：{origin}
- 总天数：{travel_days}
- 总预算：{budget}元
- 出发日期：{travel_date}
- 偏好：{', '.join(preferences) if preferences else '无'}

场景类型：{scenario_type}

你的任务：
1. **理解完整意图**：分析用户是否想去多个城市，识别每一段行程的起点和终点。
   - ⚠️ 注意：往返行程（如“上海→南京→上海”）不算多目的地，只算南京一个目的地。
   - ⚠️ 只有当用户明确表达“再去/然后去/接着去/之后到”等意图时，才判定为多目的地。
2. **分解行程段**：
   - 每段包括：origin（出发城市）、destination（目的地城市）、days（天数）、date_start（开始日期 YYYY-MM-DD）
   - 注意：第2段的origin应该是第1段的destination（连续行程）
   - ⚠️ 不要把“回程/返程/返回”单独拆成一段，它只是交通返回，不是新的目的地。
3. **预算分配**：根据每段的天数和目的地物价水平，分配总预算给所有段（budget_allocation 总和应等于总预算）
4. **制定查询计划**：为每一段制定需要查询的工具和参数

✨ **query_plan 建议**（无步骤限制！） ✨
系统已迁移到LangChain Agent，没有任何步骤限制，你可以自由规划！

【单目的地场景】
- ✅ **无步骤限制**：想查多少步都可以！
- 建议包含：rag_search + train_query + flight_query + gaode_hotel_search + gaode_weather + lucky_day + gaode_driving + 返程交通
- 优先级：rag_search > train_query > gaode_hotel_search > gaode_weather > lucky_day
- 可以同时查询多种交通方式（高铁+航班+自驾）进行对比

【多目的地场景】
- ✅ **无步骤限制**：想查多少步都可以！
- 建议每个目的地查询：rag_search + train_query + gaode_hotel_search + gaode_weather + lucky_day
- 可以为每个目的地查询完整信息
- 3个或更多目的地：每个目的地都可以详细查询

输出JSON格式（**必须是纯JSON，不要markdown代码块**）：
{{
  "travel_segments": [
    {{
      "origin": "出发城市",
      "destination": "目的地城市",
      "days": 天数,
      "date_start": "YYYY-MM-DD"
    }}
  ],
  "budget_allocation": {{
    "城市名": 预算金额
  }},
  "query_plan": [
    {{
      "segment": 段索引(0-based),
      "tool": "工具名",
      "params": {{
        "参数名": "参数值"
      }},
      "description": "这一步的目的"
    }}
  ],
  "initial_suggestions": [
    "初步建议1",
    "初步建议2"
  ]
}}

可用工具：
- rag_search: 查询景点攻略，参数 {{"query": "城市 景点"}}
- train_query: 查询火车票，参数 {{"origin": "出发地", "destination": "目的地", "date": "YYYY-MM-DD"}}
- gaode_driving: 查询自驾路线，参数 {{"origin": "起点坐标", "destination": "终点坐标"}} （需先用gaode_geo查询坐标）
- flight_query: 查询航班，参数 {{"dep": "城市名", "arr": "城市名", "date": "YYYY-MM-DD"}} （支持中文城市名，会自动转换为机场代码）
- lucky_day: 查询黄历吉日，参数 {{"date": "YYYY-MM-DD"}}
- gaode_weather: 查询天气，参数 {{"city": "城市名"}}
- gaode_hotel_search: 查询酒店，参数 {{"keywords": "城市 酒店", "city": "城市名"}}

示例1（单目的地复杂场景）：如果用户问\"月12月12日从上海到南京3天，2个老人1个孩子，预算1500元\"（预算紧张+特殊需求），应该分解为：

**travel_segments**:
[
  {{"origin": "上海", "destination": "南京", "days": 3, "date_start": "2025-12-12"}}
]

**query_plan** （无步骤限制，可以包含所有必要信息）:
[
  {{"segment": 0, "tool": "rag_search", "params": {{"query": "南京 景点 老人 儿童"}}, "description": "查询南京适合老人儿童的景点"}},
  {{"segment": 0, "tool": "train_query", "params": {{"origin": "上海", "destination": "南京", "date": "2025-12-12"}}, "description": "查询上海到南京高铁"}},
  {{"segment": 0, "tool": "gaode_hotel_search", "params": {{"keywords": "南京 经济型 酒店", "city": "南京"}}, "description": "查询南京经济型酒店"}},
  {{"segment": 0, "tool": "gaode_weather", "params": {{"city": "南京"}}, "description": "查询南京天气，提醒老人儿童添衣"}},
  {{"segment": 0, "tool": "lucky_day", "params": {{"date": "2025-12-12"}}, "description": "查询出行吉日"}},
  {{"segment": -1, "tool": "train_query", "params": {{"origin": "南京", "destination": "上海", "date": "2025-12-15"}}, "description": "查询返程交通"}}
]

✅ **无限制**：单目的地场景可以包含所有工具，没有步骤限制！

示例2（多目的地，7步精简版）：如果用户问\"上海到青岛3天再去大连2天，出发12月12日，预算3000元\"，应该分解为：

**travel_segments**:
[
  {{"origin": "上海", "destination": "青岛", "days": 3, "date_start": "2025-12-12"}},
  {{"origin": "青岛", "destination": "大连", "days": 2, "date_start": "2025-12-15"}}
]

**query_plan** （无步骤限制，每个目的地都可以查询完整信息）:
[
  {{"segment": 0, "tool": "rag_search", "params": {{"query": "青岛 景点"}}, "description": "查询青岛景点攻略"}},
  {{"segment": 0, "tool": "train_query", "params": {{"origin": "上海", "destination": "青岛", "date": "2025-12-12"}}, "description": "查询上海到青岛交通"}},
  {{"segment": 0, "tool": "gaode_hotel_search", "params": {{"keywords": "青岛 酒店", "city": "青岛"}}, "description": "查询青岛酒店"}},
  {{"segment": 1, "tool": "rag_search", "params": {{"query": "大连 景点"}}, "description": "查询大连景点攻略"}},
  {{"segment": 1, "tool": "train_query", "params": {{"origin": "青岛", "destination": "大连", "date": "2025-12-15"}}, "description": "查询青岛到大连交通"}},
  {{"segment": 1, "tool": "gaode_hotel_search", "params": {{"keywords": "大连 酒店", "city": "大连"}}, "description": "查询大连酒店"}},
  {{"segment": -1, "tool": "train_query", "params": {{"origin": "大连", "destination": "上海", "date": "2025-12-17"}}, "description": "查询返程交通"}}
]

✅ **无限制**：多目的地场景也可以为每个目的地查询天气和黄历，所有目的地都可以详细查询！

⚠️ **关键**：train_query/flight_query 的 params 必须包含 origin/destination/date，不能省略！

示例3（单目的地简单场景）：如果用户问\"我12月12日从上海到杭州2天，预算2000元\"（无特殊需求），这是往返行程，只有1个目的地：
- 第1段：上海→杭州，2天
- 不要单独拆“杭州→上海”，回程只在交通工具查询中体现，不构成新的行程段。
- 简单场景可以省略天气和黄历，关注核心信息：RAG + 交通 + 酒店 + 返程。

✨ **系统优势**：
系统已迁移到 LangChain Agent，彻底解决了递归限制问题！

✅ **无步骤限制**：query_plan 可以包含任意数量的步骤（10步、20步、甚至更多）
✅ **灵活规划**：可以为每个目的地查询完整信息
✅ **多种工具**：可以同时查询多种交通方式进行对比
✅ **完整信息**：不再需要省略天气、黄历等信息

请放心规划完整的查询流程！
    """
    
    context = {
        "user_query": user_query,
        "destination": destination,
        "origin": origin,
        "travel_days": travel_days,
        "budget": budget,
        "travel_date": travel_date,
        "scenario_type": scenario_type
    }
    
    try:
        r1 = get_r1_instance()
        print(f"  💭 R1开始深度分析...")
        result = await r1.analyze(problem, context)
        
        print(f"  ✅ R1分析完成，解析结果...")
        
        # 解析R1输出
        r1_plan = None
        try:
            # 尝试解析JSON
            if isinstance(result, str):
                # 移除可能的markdown代码块
                result_clean = result.strip()
                if result_clean.startswith('```'):
                    # 移除```json和```
                    lines = result_clean.split('\n')
                    result_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_clean
                r1_plan = json.loads(result_clean)
            else:
                r1_plan = result
            
            print(f"  ✅ JSON解析成功")
            print(f"    行程段数: {len(r1_plan.get('travel_segments', []))}")
            print(f"    查询计划步骤: {len(r1_plan.get('query_plan', []))}")
            
            # 显示行程段
            for i, segment in enumerate(r1_plan.get('travel_segments', [])):
                print(f"    段{i+1}: {segment.get('origin')} → {segment.get('destination')}, {segment.get('days')}天")
            
            # 无步骤限制！不再需要截断
            query_plan = r1_plan.get('query_plan', [])
            print(f"    query_plan 步数: {len(query_plan)} ✅ 无限制！")
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON解析失败: {e}")
            print(f"    R1返回原始文本: {result[:200]}...")
            # 如果JSON解析失败，创建一个简单的回退计划
            r1_plan = {
                "travel_segments": [],
                "budget_allocation": {},
                "query_plan": [],
                "initial_suggestions": []
            }
        print(f"{'='*60}\n")
        
        # 保持原始全局字段不变，仅更新R1相关数据
        update_dict = {
            'r1_plan': r1_plan,
            'travel_segments': r1_plan.get('travel_segments', []),
            'scenario_type': scenario_type,
            'messages': [status_msg]
        }
        
        # 显示行程段信息（调试用）
        segments = r1_plan.get('travel_segments', [])
        if segments:
            print(f"  ✅ R1生成 {len(segments)} 个行程段：")
            for i, seg in enumerate(segments):
                print(f"     段{i+1}: {seg.get('origin')} → {seg.get('destination')}, {seg.get('days')}天, 出发{seg.get('date_start')}")
        
        return update_dict
        
    except Exception as e:
        print(f"❌ R1 Strategy分析异常: {e}")
        import traceback
        traceback.print_exc()
        
        # 回退方案：返回空计划，让系统继续使用Qwen3主导
        return {
            'r1_plan': None,
            'travel_segments': [],
            'scenario_type': scenario_type,
            'messages': [AIMessage(content=f"⚠️ R1分析遇到问题，将使用标准流程处理: {str(e)}")]
        }


async def r1_optimization_node(state: TravelPlanState) -> Dict[str, Any]:
    """
R1优化节点 - 基于收集的数据优化多段方案
    
    这是R1的第二次介入，负责：
    1. 综合分析所有收集的数据（多段rag、交通、天气等）
    2. 对比不同段的性价比
    3. 优化预算分配和时间安排
    4. 生成多套方案对比
    5. 识别风险并提出建议
    """
    from travel_agent.tools.r1_tool import get_r1_instance
    
    print(f"\n{'='*60}")
    print("▶️ R1 Optimization 节点开始执行")
    
    status_msg = AIMessage(content="📊 正在综合分析行程数据，优化方案…")
    
    r1_plan = state.get('r1_plan', {})
    travel_segments = state.get('travel_segments', [])
    budget = state.get('budget', 0)
    
    if not travel_segments:
        print("  ⚠️ 没有多段行程，跳过R1优化")
        return {'messages': [status_msg]}
    
    print(f"  行程段数: {len(travel_segments)}")
    
    # 收集所有段的数据
    segment_data = []
    for i, segment in enumerate(travel_segments):
        seg_info = {
            'segment_index': i,
            'route': f"{segment.get('origin')} → {segment.get('destination')}",
            'destination': segment.get('destination'),
            'days': segment.get('days'),
            'date_start': segment.get('date_start', ''),
        }
        
        # 添加该段的查询结果
        # 从 rag_results_history 中提取对应段的结果
        rag_history = state.get('rag_results_history', [])
        if i < len(rag_history):
            seg_info['rag_results'] = rag_history[i][:500] if rag_history[i] else '未查询'
        
        # 从 segment_train_info 中提取
        segment_train = state.get('segment_train_info', {})
        if i in segment_train:
            train_data = segment_train[i]
            if isinstance(train_data, dict):
                seg_info['train_info'] = f"已查询，票价约{train_data.get('price', 'N/A')}元"
            else:
                seg_info['train_info'] = '已查询'
        else:
            # 回退：从全局train_info查找
            seg_info['train_info'] = '未查询'
        
        segment_data.append(seg_info)
        print(f"    段{i+1}: {seg_info['route']}, {seg_info['days']}天")
    
    # 构建R1 prompt
    problem = f"""
用户的多段旅行计划已经执行完成，现在需要你进行深度优化和分析。

R1初步规划：
{json.dumps(r1_plan.get('initial_suggestions', []), ensure_ascii=False, indent=2)}

预算分配：
{json.dumps(r1_plan.get('budget_allocation', {}), ensure_ascii=False, indent=2)}

实际收集的数据：
{json.dumps(segment_data, ensure_ascii=False, indent=2)}

总预算：{budget}元
全局信息：
- 天气: {str(state.get('weather_info', {}))[: 200]}
- 黄历: {str(state.get('lucky_day_info', ''))[: 200]}

你的任务：
1. **预算分析**：
   - 每段的预算分配是否合理？
   - 是否需要调整？（如果某段景点少、物价低，可以减少预算）
   - 给出优化后的预算分配建议

2. **时间安排分析**：
   - 每段的天数安排是否合理？
   - 是否过于紧凑或松散？
   - 是否需要调整？

3. **性价比对比**：
   - 每段的性价比如何？
   - 哪一段更值得多花时间？

4. **风险评估**：
   - 天气风险（雨天、高温、低温）
   - 时间风险（连接紧张、休息不足）
   - 预算风险（超支可能性）

5. **方案对比**：
   - 生成 2-3 套可选方案：
     * 经济型：最小化成本
     * 均衡型：当前方案（可微调）
     * 舒适型：提升体验（可增加预算）

输出JSON格式（**纯JSON，不markdown**）：
{{
  "budget_analysis": {{
    "original": {{}},
    "optimized": {{}},
    "adjustment_reason": ""
  }},
  "time_analysis": {{
    "issues": [],
    "suggestions": []
  }},
  "value_comparison": [
    {{
      "segment": 0,
      "destination": "",
      "value_score": "9/10",
      "highlights": ["亮点1", "亮点2"],
      "concerns": ["问题1"]
    }}
  ],
  "risk_warnings": [
    "风险1",
    "风险2"
  ],
  "alternative_plans": [
    {{
      "name": "经济方案",
      "description": "",
      "total_cost": 0,
      "pros": [],
      "cons": []
    }}
  ],
  "final_recommendation": ""
}}
    """
    
    context = {
        "r1_plan": r1_plan,
        "travel_segments": travel_segments,
        "segment_data": segment_data,
        "budget": budget
    }
    
    try:
        r1 = get_r1_instance()
        print(f"  💭 R1开始优化分析...")
        result = await r1.analyze(problem, context)
        
        print(f"  ✅ R1优化完成，解析结果...")
        
        # 解析R1输出
        optimization = None
        try:
            if isinstance(result, str):
                result_clean = result.strip()
                if result_clean.startswith('```'):
                    lines = result_clean.split('\n')
                    result_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_clean
                optimization = json.loads(result_clean)
            else:
                optimization = result
            
            print(f"  ✅ JSON解析成功")
            print(f"    风险警告数: {len(optimization.get('risk_warnings', []))}")
            print(f"    替代方案数: {len(optimization.get('alternative_plans', []))}")
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON解析失败: {e}")
            print(f"    R1返回原始文本: {result[:200]}...")
            # 使用原始文本作为回退
            optimization = {
                "raw_analysis": result,
                "risk_warnings": [],
                "alternative_plans": []
            }
        
        print(f"{'='*60}\n")
        
        return {
            'reasoning_chain': result,  # 保留原始推理链
            'optimization_suggestions': optimization.get('budget_analysis', {}),
            'alternative_plans': optimization.get('alternative_plans', []),
            'risk_warnings': optimization.get('risk_warnings', []),
            'value_comparison': optimization.get('value_comparison', []),
            'messages': [status_msg]
        }
        
    except Exception as e:
        print(f"❌ R1 Optimization异常: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'reasoning_chain': f"R1优化失败: {str(e)}",
            'optimization_suggestions': {},
            'alternative_plans': [],
            'messages': [AIMessage(content=f"⚠️ R1优化遇到问题: {str(e)}")]
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
    travel_segments = state.get("travel_segments", [])
    
    # 构建完整的用户需求描述
    # 检测是否为多段行程
    if travel_segments and len(travel_segments) > 1:
        # 多段行程：构建完整路线描述
        route_parts = [travel_segments[0].get('origin', origin)]
        for seg in travel_segments:
            route_parts.append(seg.get('destination', ''))
        route_str = " → ".join(filter(None, route_parts))
        
        user_query_parts = [f"从{route_str}旅游"]
        
        # 计算总天数
        total_days = sum(seg.get('days', 0) for seg in travel_segments)
        if total_days:
            user_query_parts.append(f"共{total_days}天")
    else:
        # 单段或无段：使用原逻辑
        user_query_parts = []
        if origin:
            user_query_parts.append(f"从{origin}")
        if destination:
            user_query_parts.append(f"去{destination}旅游")
        if travel_days:
            user_query_parts.append(f"{travel_days}天")
    
    # 添加通用信息
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
    
    # 支持累积历史（ReAct 模式）和单次结果（旧模式）
    rag_results_history = state.get("rag_results_history", [])
    if rag_results_history:
        # ReAct 模式：使用累积的多次检索结果
        rag_results = "\n\n---\n\n".join(rag_results_history)
        print(f"  使用 ReAct 累积的 RAG 结果: {len(rag_results_history)} 次检索")
    else:
        # 旧模式：向后兼容
        rag_results = state.get("rag_results", "")
        print(f"  使用单次 RAG 结果（旧模式）")
    
    if query_mode == "simple":
        # 简单查询模式：只显示景点信息
        prompt = SIMPLE_QUERY_PROMPT_TEMPLATE.format(
            destination=destination,
            rag_results=rag_results or "未找到相关信息"
        )
    else:
        # 完整规划模式：显示完整旅行方案
        # 提取 train_info 的完整文本，并增强错误处理
        train_info_raw = state.get("train_info", {})
        if isinstance(train_info_raw, dict):
            if "tickets_text" in train_info_raw:
                train_info_text = train_info_raw["tickets_text"]
            elif "error" in train_info_raw:
                # 提取详细错误信息
                error_msg = train_info_raw['error']
                reason = train_info_raw.get('reason', '')
                suggestion = train_info_raw.get('suggestion', '请使用12306官网或APP查询')
                train_info_text = f"❌ 工具调用失败：{error_msg}"
                if reason:
                    train_info_text += f"\n原因：{reason}"
                if suggestion:
                    train_info_text += f"\n建议：{suggestion}"
            else:
                train_info_text = str(train_info_raw)
        else:
            train_info_text = str(train_info_raw)
        
        # 提取 driving_info 的完整文本，并增强错误处理
        driving_info_raw = state.get("driving_info")
        if driving_info_raw is None:
            driving_info_text = "❌ 未查询到自驾路线（可能距离过远或输入异常）"
        elif isinstance(driving_info_raw, dict):
            if "error" in driving_info_raw:
                # 提取详细错误信息
                error_msg = driving_info_raw.get('error', '查询失败')
                suggestion = driving_info_raw.get('suggestion', '')
                driving_info_text = f"❌ {error_msg}"
                if suggestion:
                    driving_info_text += f"\n建议：{suggestion}"
            elif "data" in driving_info_raw:
                # 成功查询，提取数据
                data = driving_info_raw["data"]
                distance_km = driving_info_raw.get("distance_km", 0)
                warning = driving_info_raw.get("warning", "")
                driving_info_text = str(data)
                if warning:
                    driving_info_text = f"⚠️ {warning}\n\n{driving_info_text}"
            else:
                driving_info_text = str(driving_info_raw)
        else:
            driving_info_text = str(driving_info_raw)
        
        # 提取 flight_info 的完整文本
        flight_info_raw = state.get("flight_info")
        if flight_info_raw:
            flight_info_text = str(flight_info_raw)
        else:
            flight_info_text = "未查询航班信息（通常用于远距离出行 >800km）"
        
        print(f"  train_info 长度: {len(train_info_text)} 字符")
        print(f"  train_info 前300字符: {train_info_text[:300]}")
        print(f"  driving_info 长度: {len(driving_info_text)} 字符")
        print(f"  driving_info 前300字符: {driving_info_text[:300]}")
        print(f"  flight_info 长度: {len(flight_info_text)} 字符")
        print(f"  flight_info 前300字符: {flight_info_text[:300]}")
        
        prompt = SYNTHESIZER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            rag_results=rag_results or "",  # 使用上面处理后的 rag_results
            hotel_info=state.get("hotel_info", "未查询到酒店信息"),
            train_info=train_info_text,
            driving_info=driving_info_text,
            flight_info=flight_info_text,
            lucky_day_info=state.get("lucky_day_info", ""),
            weather_info=state.get("weather_info", {})
        )
    
    # ==== 添加R1分析结果展示 ====
    r1_plan = state.get('r1_plan')
    travel_segments = state.get('travel_segments', [])
    risk_warnings = state.get('risk_warnings', [])
    alternative_plans = state.get('alternative_plans', [])
    value_comparison = state.get('value_comparison', [])
    
    if r1_plan or travel_segments:
        print(f"  🧠 检测到R1分析结果，添加到prompt")
        
        r1_section = "\n\n"
        r1_section += "="*50 + "\n"
        r1_section += "🧐 **智能规划分析**\n"
        r1_section += "="*50 + "\n\n"
        
        # 1. 行程分解
        if travel_segments:
            r1_section += "🗺️ **多段行程规划**\n\n"
            r1_section += format_travel_segments(travel_segments)
            r1_section += "\n\n"
        
        # 2. 预算分配
        if r1_plan and 'budget_allocation' in r1_plan:
            budget_alloc = r1_plan.get('budget_allocation', {})
            if budget_alloc:
                r1_section += "💰 **预算分配建议**\n\n"
                r1_section += format_budget_allocation(budget_alloc)
                r1_section += "\n\n"
        
        # 3. 性价比对比
        if value_comparison:
            r1_section += "🎯 **性价比分析**\n"
            r1_section += format_value_comparison(value_comparison)
            r1_section += "\n\n"
        
        # 4. 风险警告
        if risk_warnings:
            r1_section += "⚠️ **风险提示**\n\n"
            r1_section += format_risk_warnings(risk_warnings)
            r1_section += "\n\n"
        
        # 5. 替代方案
        if alternative_plans:
            r1_section += "🔄 **可选方案对比**\n"
            r1_section += format_alternative_plans(alternative_plans)
            r1_section += "\n\n"
        
        # 6. 添加提示
        r1_section += "-" * 50 + "\n"
        r1_section += "📚 *以上分析由智能系统综合评估生成，为您的旅行决策提供参考*\n"
        r1_section += "=" * 50 + "\n\n"
        
        # 将R1分析添加到prompt之前
        prompt = r1_section + prompt
    
    response = await qwen3_llm.ainvoke([HumanMessage(content=prompt)])
    
    # 返回 messages 格式供 Chat API 使用
    return {
        "travel_plan": response.content,
        "messages": [AIMessage(content=response.content)],
    }


# ========== ReAct Agentic RAG Nodes ==========

async def thought_node(state: TravelPlanState) -> Dict[str, Any]:
    """
    ReAct 思考节点 - 分析当前状态，决定下一步行动
    这是 ReAct 循环的大脑，负责：
    1. 分析已收集的信息
    2. 识别信息缺口
    3. 决定下一步使用哪个工具
    
    **双模型协同**：
    - 如果有R1计划：按照R1的query_plan执行（R1主导）
    - 如果没有R1计划：Qwen3自主决策（Qwen3主导）
    """
    from travel_agent.tools.tool_registry import get_tools_description_for_llm
    
    print(f"\n{'='*60}")
    print("🧠 [THOUGHT NODE] 开始思考...")
    print(f"{'='*60}")
    
    iteration_count = state.get("iteration_count", 0) or 0
    max_iterations = state.get("max_iterations", 8) or 8
    r1_plan = state.get("r1_plan")
    
    print(f"当前迭代: {iteration_count}/{max_iterations}")
    print(f"R1计划状态: {'R1主导' if r1_plan else 'Qwen3主导'}")
    
    # 安全检查：如果迭代次数已达到最大值，强制结束
    if iteration_count >= max_iterations:
        print(f"  ⚠️ 达到最大迭代次数 {max_iterations}，强制结束循环")
        return {
            "current_thought": f"达到最大迭代次数 ({max_iterations})，结束循环",
            "thought_history": [f"达到max_iterations"],
            "current_action": {"tool": "final_answer", "params": {}},
            "should_continue": False,
            "is_complete": True,
            "iteration_count": iteration_count + 1,
        }
    
    # ==== R1主导模式：按照R1的计划执行 ====
    if r1_plan and 'query_plan' in r1_plan:
        query_plan = r1_plan.get('query_plan', [])
        
        # 打印完整的 query_plan 内容（用于调试）
        if iteration_count == 0:
            print(f"\n📝 R1生成的完整 query_plan ({len(query_plan)}步):")
            for i, step in enumerate(query_plan):
                print(f"  [{i}] segment={step.get('segment')}, tool={step.get('tool')}, params={step.get('params')}")
            print(f"{'='*60}\n")
        
        # 强制检查：如果已经完成所有步骤，直接返回 final_answer
        # 注意：iteration_count 在这里是已经+1后的值，所以需要比较 >= len(query_plan)
        if iteration_count >= len(query_plan):
            print(f"\n✅ R1计划已全部执行完毕（iteration={iteration_count}, plan_length={len(query_plan)}）")
            print(f"  强制返回 final_answer，准备进入Synthesizer")
            print(f"{'='*60}\n")
            return {
                "current_thought": "R1计划已全部执行完毕，信息充分",
                "thought_history": ["R1计划完成"],
                "current_action": {
                    "tool": "final_answer",
                    "params": {}
                },
                "should_continue": False,
                "is_complete": True,
                "iteration_count": iteration_count,  # 保持当前值
            }
        
        if iteration_count < len(query_plan):
            # 执行R1计划中的下一步
            next_step = query_plan[iteration_count]
            tool_name = next_step.get('tool', '')
            params = next_step.get('params', {})
            description = next_step.get('description', '')
            segment = next_step.get('segment', 0)
            
            # 检查是否是最后一步
            # iteration_count 是从0开始，所以最后一步的索引 = len(query_plan) - 1
            is_last_step = iteration_count == (len(query_plan) - 1)
            
            print(f"\n🎯 [按R1计划执行] 第{iteration_count + 1}步，共{len(query_plan)}步")
            if is_last_step:
                print(f"  ✅ 这是最后一步，执行后将结束")
            print(f"  段索引: {segment}")
            print(f"  工具: {tool_name}")
            print(f"  参数: {params}")
            print(f"  目的: {description}")
            print(f"{'='*60}\n")
            
            return {
                "current_thought": f"R1计划第{iteration_count+1}步：{description}",
                "thought_history": [f"R1计划第{iteration_count+1}步：{description}"],
                "current_action": {
                    "tool": tool_name,
                    "params": params,
                    "segment": segment,  # 保留段信息供action_node使用
                    "is_last_step": is_last_step  # 标记是否是最后一步
                },
                "should_continue": not is_last_step,  # 如果是最后一步，下次就不继续了
                "is_complete": is_last_step,  # 如果是最后一步，标记完成
                "iteration_count": iteration_count + 1,
            }
        else:
            # R1计划执行完毕
            print(f"\n✅ R1计划已全部执行完毕（{len(query_plan)}步）")
            
            # === 禁用补充查询：为了递归限制，不再追加 lucky_day 查询 ===
            # 递归计算：补充查询会增加 3 步 (thought + action + observation)
            # 单目的地 + 补充: 3 + 6×3 + 3(补充) + 1(r1_opt) + 1(synth) = 26 > 25 ❌
            # 因此必须禁用补充查询，R1 必须在 query_plan 中直接包含所有必要查询
            
            lucky_missing = not state.get("lucky_day_info")
            if lucky_missing:
                print(f"  ℹ️ 检测到缺少黄历信息，但为了递归限制，不进行补充查询")
            
            print(f"  准备进入Synthesizer生成最终方案")
            print(f"{'='*60}\n")
            
            return {
                "current_thought": "R1计划已全部执行完毕，信息充分",
                "thought_history": ["R1计划完成"],
                "current_action": {
                    "tool": "final_answer",
                    "params": {}
                },
                "should_continue": False,
                "is_complete": True,
                "iteration_count": iteration_count + 1,
            }
    
    # ==== Qwen3主导模式：自主决策 ====
    print(f"\n🤖 [Qwen3自主决策模式]")
    
    # 构建已收集信息的摘要
    collected_info = []
    
    # RAG 检索结果
    if state.get("rag_results"):
        rag_summary = state.get("rag_results", "")[:500]  # 只显示前500字符
        collected_info.append(f"• RAG检索: 已获取{len(state.get('rag_results',''))} 字符的景点信息")
    
    # 火车票信息
    if state.get("train_info"):
        collected_info.append("• 火车票: 已查询")
    
    # 自驾路线
    if state.get("driving_info"):
        collected_info.append("• 自驾路线: 已查询")
    
    # 酒店信息
    if state.get("hotel_info"):
        collected_info.append("• 酒店信息: 已获取")
    
    # 天气信息
    if state.get("weather_info"):
        collected_info.append("• 天气预报: 已查询")
    
    # 黄历信息
    if state.get("lucky_day_info"):
        collected_info.append("• 黄历吉日: 已查询")
    
    # 航班信息
    if state.get("flight_info"):
        collected_info.append("• 航班信息: 已查询")
    
    collected_info_str = "\n".join(collected_info) if collected_info else "⚠️ 暂无信息"
    
    print(f"已收集的信息:\n{collected_info_str}")
    
    # 获取可用工具描述
    tools_desc = get_tools_description_for_llm()
    
    # 构造用户查询
    user_query_parts = []
    if state.get("destination"):
        user_query_parts.append(f"目的地:{state['destination']}")
    if state.get("origin"):
        user_query_parts.append(f"出发地:{state['origin']}")
    if state.get("travel_days"):
        user_query_parts.append(f"{state['travel_days']}天")
    if state.get("budget"):
        user_query_parts.append(f"预算{state['budget']}元")
    
    user_query = ", ".join(user_query_parts) if user_query_parts else "未知需求"
    
    # 使用 REACT_THOUGHT_PROMPT
    prompt = REACT_THOUGHT_PROMPT.format(
        user_query=user_query,
        destination=state.get("destination", "未知"),
        origin=state.get("origin", "未知"),
        travel_days=state.get("travel_days", 0),
        budget=state.get("budget", 0),
        travel_date=state.get("travel_date", "未知"),
        preferences=state.get("preferences", []),
        collected_info=collected_info_str,
        iteration_count=iteration_count,
        max_iterations=max_iterations,
        available_tools=tools_desc
    )
    
    try:
        response = await qwen3_llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # 解析 JSON
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        
        # 提取第一个 JSON 对象
        if content.startswith("{"):
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
        
        decision = json.loads(content)
        
        thought = decision.get("thought", "")
        action = decision.get("action", "")
        action_input = decision.get("action_input", {})
        continue_flag = decision.get("continue", True)
        
        print(f"\n💡 思考: {thought}")
        print(f"🎯 决定行动: {action}")
        print(f"📝 行动参数: {action_input}")
        print(f"➡️  继续循环: {continue_flag}")
        print(f"{'='*60}\n")
        
        return {
            "current_thought": thought,
            "thought_history": [thought],
            "current_action": {
                "tool": action,
                "params": action_input
            },
            "should_continue": continue_flag,
            "iteration_count": iteration_count + 1,
        }
        
    except Exception as e:
        print(f"❌ 思考节点异常: {e}")
        print(f"原始响应: {response.content if 'response' in locals() else 'N/A'}")
        import traceback
        traceback.print_exc()
        
        # 失败时的默认行为：结束循环
        return {
            "current_thought": f"思考失败: {str(e)}",
            "thought_history": [f"思考失败: {str(e)}"],
            "current_action": {
                "tool": "final_answer",
                "params": {}
            },
            "should_continue": False,
            "is_complete": True,
            "iteration_count": iteration_count + 1,
        }


async def action_node(state: TravelPlanState) -> Dict[str, Any]:
    """
    ReAct 行动节点 - 执行工具调用
    根据 thought_node 的决策，调用相应的工具
    """
    print(f"\n{'='*60}")
    print("⚙️ [ACTION NODE] 执行行动...")
    print(f"{'='*60}")
    
    current_action = state.get("current_action", {})
    if not current_action:
        print("⚠️ 没有行动指令，跳过")
        return {"current_observation": "没有行动"}
    
    tool_name = current_action.get("tool", "")
    params = current_action.get("params", {})
    
    print(f"🔧 工具: {tool_name}")
    print(f"📝 参数: {params}")
    
    # 特殊工具：final_answer
    if tool_name == "final_answer":
        print("✅ 信息已充分，准备生成最终答案")
        return {
            "current_observation": "信息充分，准备生成答案",
            "is_complete": True,
            "should_continue": False,
        }
    
    # 调用对应的工具
    observation = ""
    
    try:
        # 根据工具名称路由到相应的函数
        if tool_name == "rag_search":
            # 调用 RAG 检索
            from travel_agent.tools.rag_tool import get_rag_instance
            rag = get_rag_instance()
            query = params.get("query", "")
            k = params.get("k", 3)
            result = await rag.search(query, k=k)
            observation = f"RAG检索结果: {result}"
            
            # 更新 state
            return {
                "rag_results": result,
                "current_observation": observation,
                "action_history": [current_action],
            }
        
        elif tool_name == "train_query":
            # 调用完整的 train_query_node，包含站点代码查询、自驾路线等逻辑
            print(f"  调用完整的 train_query_node...")
            
            # 检查参数来源
            origin_from_params = params.get("origin")
            dest_from_params = params.get("destination")
            date_from_params = params.get("date")
            
            origin_from_state = state.get("origin", "")
            dest_from_state = state.get("destination", "")
            date_from_state = state.get("travel_date", "")
            
            # 如果 params 不完整，尝试根据 segment 回填
            seg_idx = state.get("current_action", {}).get("segment")
            if (not origin_from_params or not dest_from_params or not date_from_params) and seg_idx is not None:
                segs = state.get("travel_segments", []) or []
                if 0 <= seg_idx < len(segs):
                    seg = segs[seg_idx]
                    origin_from_params = origin_from_params or seg.get("origin")
                    dest_from_params = dest_from_params or seg.get("destination")
                    date_from_params = date_from_params or seg.get("date_start")
                    print(f"  🔄 基于segment[{seg_idx}]回填参数: origin={origin_from_params}, dest={dest_from_params}, date={date_from_params}")
            
            print(f"  📝 参数来源调试:")
            print(f"    params中: origin={params.get('origin')}, dest={params.get('destination')}, date={params.get('date')}")
            print(f"    state中: origin={origin_from_state}, dest={dest_from_state}, date={date_from_state}")
            
            # 构造临时 state，包含必要的字段（优先使用回填后的params）
            temp_state = {
                "origin": origin_from_params or origin_from_state,
                "destination": dest_from_params or dest_from_state,
                "travel_date": date_from_params or date_from_state,
            }
            
            print(f"    最终使用: origin={temp_state['origin']}, dest={temp_state['destination']}, date={temp_state['travel_date']}")
            
            # 调用完整的 train_query_node，添加超时保护
            try:
                import asyncio
                # 设置90秒超时（12306返回数据可能很大）
                result = await asyncio.wait_for(
                    train_query_node(temp_state),
                    timeout=90.0
                )
            except asyncio.TimeoutError:
                print(f"  ⚠️ 12306查询超时90秒，返回部分结果")
                result = {
                    "train_info": {
                        "error": "12306查询超时，可能是网络不稳定或返回数据过大",
                        "origin": temp_state['origin'],
                        "destination": temp_state['destination'],
                        "date": temp_state['travel_date']
                    },
                    "driving_info": None
                }
            except Exception as e:
                print(f"  ❌ 12306查询异常: {type(e).__name__}: {str(e)}")
                result = {
                    "train_info": {
                        "error": f"12306查询失败: {str(e)}",
                        "origin": temp_state['origin'],
                        "destination": temp_state['destination'],
                        "date": temp_state['travel_date']
                    },
                    "driving_info": None
                }
            
            # 提取结果
            train_info = result.get("train_info", {})
            driving_info = result.get("driving_info")
            
            observation = f"火车票查询结果: {str(train_info)[:500]}"
            if driving_info:
                observation += f"\n自驾路线: {str(driving_info)[:200]}"
            
            return {
                "train_info": train_info,
                "driving_info": driving_info,
                "current_observation": observation,
                "action_history": [current_action],
            }
        
        elif tool_name == "gaode_weather":
            # 调用完整的 weather_query_node，包含多日预报和日期匹配
            print(f"  调用完整的 weather_query_node...")
            
            # 检查参数来源
            city_from_params = params.get("city")
            dest_from_state = state.get("destination", "")
            
            # 如果缺参数，尝试基于 segment 回填
            seg_idx = state.get("current_action", {}).get("segment")
            if not city_from_params and seg_idx is not None:
                segs = state.get("travel_segments", []) or []
                if 0 <= seg_idx < len(segs):
                    city_from_params = segs[seg_idx].get("destination")
                    print(f"  🔄 基于segment[{seg_idx}]回填 city={city_from_params}")
            
            print(f"  📝 参数来源调试:")
            print(f"    params中: city={params.get('city')}")
            print(f"    state中: destination={dest_from_state}")
            
            # 构造临时 state
            temp_state = {
                "destination": city_from_params or dest_from_state,
                "travel_days": state.get("travel_days", 1),
                "travel_date": state.get("travel_date", ""),
            }
            
            print(f"    最终使用: destination={temp_state['destination']}")
            
            # 调用完整的 weather_query_node
            result = await weather_query_node(temp_state)
            
            # 提取结果
            weather_info = result.get("weather_info", {})
            
            observation = f"天气查询结果: {str(weather_info)[:300]}"
            
            return {
                "weather_info": weather_info,
                "current_observation": observation,
                "action_history": [current_action],
            }
        
        elif tool_name == "gaode_hotel_search":
            # 查询酒店
            from travel_agent.tools.mcp_tools import get_mcp_manager
            manager = await get_mcp_manager()
            
            print(f"  📝 参数来源调试:")
            print(f"    params原始内容: {params}")
            
            # 如果缺少 city/keywords，尝试基于 segment 回填
            seg_idx = state.get("current_action", {}).get("segment")
            if seg_idx is not None and (not params.get("city") or not params.get("keywords")):
                segs = state.get("travel_segments", []) or []
                if 0 <= seg_idx < len(segs):
                    city = params.get("city") or segs[seg_idx].get("destination")
                    keywords = params.get("keywords") or f"{city} 酒店"
                    params = {**params, "city": city, "keywords": keywords}
                    print(f"  🔄 基于segment[{seg_idx}]回填: city={city}, keywords={keywords}")
            
            result = await manager.call_tool(
                "Gaode Server",
                "maps_text_search",
                **params
            )
            observation = f"酒店搜索结果: {str(result)[:500]}"
            
            return {
                "hotel_info": str(result),
                "current_observation": observation,
                "action_history": [current_action],
            }
        
        elif tool_name == "lucky_day":
            # 调用完整的 lucky_day_query_node，包含多种参数尝试和错误处理
            print(f"  调用完整的 lucky_day_query_node...")
            
            # 检查参数来源
            date_from_params = params.get("date")
            date_from_state = state.get("travel_date", "")
            
            print(f"  📝 参数来源调试:")
            print(f"    params中: date={date_from_params}")
            print(f"    state中: travel_date={date_from_state}")
            
            # 构造临时 state
            temp_state = {
                "travel_date": params.get("date", state.get("travel_date", "")),
            }
            
            print(f"    最终使用: travel_date={temp_state['travel_date']}")
            
            # 调用完整的 lucky_day_query_node
            result = await lucky_day_query_node(temp_state)
            
            # 提取结果
            lucky_day_info = result.get("lucky_day_info")
            
            if lucky_day_info:
                observation = f"黄历查询结果: {str(lucky_day_info)[:300]}"
            else:
                observation = "黄历查询失败，可能是 bazi Server 连接问题或参数不匹配"
            
            return {
                "lucky_day_info": lucky_day_info,
                "current_observation": observation,
                "action_history": [current_action],
            }
        
        elif tool_name == "flight_query":
            # 查询航班
            from travel_agent.tools.mcp_tools import get_mcp_manager
            manager = await get_mcp_manager()
            
            # 提取参数（支持两种格式）
            print(f"  📝 参数来源调试:")
            print(f"    params原始内容: {params}")
            
            dep = params.get("dep") or params.get("origin", "")
            arr = params.get("arr") or params.get("destination", "")
            date = params.get("date", "")
            
            # 如果参数缺失，基于 segment 回填
            seg_idx = state.get("current_action", {}).get("segment")
            if (not dep or not arr or not date) and seg_idx is not None:
                segs = state.get("travel_segments", []) or []
                if 0 <= seg_idx < len(segs):
                    seg = segs[seg_idx]
                    dep = dep or seg.get("origin", dep)
                    arr = arr or seg.get("destination", arr)
                    date = date or seg.get("date_start", date)
                    print(f"  🔄 基于segment[{seg_idx}]回填: dep={dep}, arr={arr}, date={date}")
            
            print(f"    提取后: dep={dep}, arr={arr}, date={date}")
            print(f"  航班查询: {dep} → {arr}, {date}")
            
            # ========== 城市名转机场三字码 ==========
            # flight Server 要求 dep/arr 必须是 3 位机场代码（IATA代码）
            city_to_airport = {
                # 主要城市机场代码
                "上海": "SHA",  # 上海浦东/虹桥
                "北京": "PEK",  # 首都机场
                "广州": "CAN",  # 白云机场
                "深圳": "SZX",  # 宝安机场
                "成都": "CTU",  # 双流机场
                "杭州": "HGH",  # 萧山机场
                "重庆": "CKG",  # 江北机场
                "西安": "XIY",  # 咸阳机场
                "武汉": "WUH",  # 天河机场
                "南京": "NKG",  # 禄口机场
                "青岛": "TAO",  # 流亭机场
                "大连": "DLC",  # 周水子机场
                "天津": "TSN",  # 滨海机场
                "厅门": "XMN",  # 高崎机场
                "福州": "FOC",  # 长乐机场
                "昆明": "KMG",  # 长水机场
                "湖南": "CSX",  # 黄花机场
                "济南": "TNA",  # 遥墙机场
                "郑州": "CGO",  # 新郑机场
                "沈阳": "SHE",  # 桃仙机场
                "哈尔滨": "HRB",  # 太平机场
                "长春": "CGQ",  # 龙嘉机场
                "长沙": "CSX",  # 黄花机场
                "南昌": "KHN",  # 昌北机场
                "拉萨": "LXA",  # 贡嘎机场
                "乌鲁木齐": "URC",  # 地窝堡机场
                "海口": "HAK",  # 美兰机场
                "三亚": "SYX",  # 凤凰机场
                "贵阳": "KWE",  # 龙洞堡机场
                "银川": "INC",  # 河东机场
                "兰州": "LHW",  # 中川机场
                "太原": "TYN",  # 武宿机场
                "石家庄": "SJW",  # 正定机场
                "苏州": "SZV",  # 光福机场
                "无锡": "WUX",  # 硕放机场
                "宁波": "NGB",  # 栗树机场
                "温州": "WNZ",  # 龙湾机场
                "南通": "NTG",  # 兴东机场
                "大理": "DLU",  # 机场
                "丽江": "LJG",  # 三义机场
                "威海": "WEH",  # 大水泊机场
                "烟台": "YNT",  # 蓬莱机场
                "唐山": "TVS",  # 三女河机场
                "南宁": "NNG",  # 吾垩机场
                "桂林": "KWL",  # 两江机场
                "呆湖": "GMQ",  # 机场
                "扬州": "YTY",  # 泰州机场
            }
            
            # 如果是中文城市名，转换为机场代码
            if dep in city_to_airport:
                original_dep = dep
                dep = city_to_airport[dep]
                print(f"  🔄 转换出发地: {original_dep} → {dep}")
            
            if arr in city_to_airport:
                original_arr = arr
                arr = city_to_airport[arr]
                print(f"  🔄 转换目的地: {original_arr} → {arr}")
            
            # 验证机场代码格式
            if len(dep) != 3 or len(arr) != 3:
                observation = f"⚠️ 航班查询失败：出发地({dep})或目的地({arr})不是有效的机场三字码，且未能自动转换。请使用火车/自驾方案。"
                print(observation)
                return {
                    "flight_info": {"error": observation},
                    "current_observation": observation,
                    "action_history": [current_action],
                }
            
            result = await manager.call_tool(
                "flight Server",
                "searchFlightsByDepArr",
                dep=dep,
                arr=arr,
                date=date
            )
            observation = f"航班查询结果: {str(result)[:500]}"
            
            return {
                "flight_info": str(result),
                "current_observation": observation,
                "action_history": [current_action],
            }
        
        elif tool_name == "r1_analysis":
            # 调用 DeepSeek R1
            problem = params.get("problem", "")
            context = params.get("context", {})
            
            # 直接调用 deep_analysis_node
            result = await deep_analysis_node(state)
            observation = f"R1分析完成: {result.get('reasoning_chain', '')[:500]}"
            
            return {
                **result,
                "current_observation": observation,
                "action_history": [current_action],
            }
        
        else:
            observation = f"⚠️ 未知工具: {tool_name}"
            print(observation)
            return {
                "current_observation": observation,
                "action_history": [current_action],
            }
    
    except Exception as e:
        observation = f"❌ 工具调用失败: {tool_name}, 错误: {str(e)}"
        print(observation)
        import traceback
        traceback.print_exc()
        
        return {
            "current_observation": observation,
            "action_history": [current_action],
        }
    
    finally:
        print(f"{'='*60}\n")


async def observation_node(state: TravelPlanState) -> Dict[str, Any]:
    """
    ReAct 观察节点 - 评估工具调用结果
    分析工具返回的结果，判断是否需要继续收集信息
    """
    print(f"\n{'='*60}")
    print("🔍 [OBSERVATION NODE] 观察结果...")
    print(f"{'='*60}")
    
    # 安全检查：如果 action 是 final_answer，直接返回完成
    current_action = state.get("current_action", {})
    if current_action.get("tool") == "final_answer":
        print("✅ 检测到 final_answer，直接结束循环")
        return {
            "observation_history": ["信息已充分，准备生成答案"],
            "is_complete": True,
            "should_continue": False,
        }
    
    latest_observation = state.get("current_observation", "")
    print(f"最新观察: {latest_observation[:200]}...")
    
    # ========== R1模式下特殊处理 ==========
    # 在 R1 主导模式下，不用 Qwen3 评估，直接信任 R1 的计划
    r1_plan = state.get("r1_plan")
    if r1_plan and 'query_plan' in r1_plan:
        query_plan = r1_plan.get('query_plan', [])
        iteration_count = state.get("iteration_count", 0) or 0
        
        print(f"  🧠 [R1模式] 进度: {iteration_count}/{len(query_plan)}")
        
        # 检查 action 是否标记为最后一步
        is_last_step_marked = current_action.get("is_last_step", False)
        if is_last_step_marked:
            print(f"  ✅ 检测到最后一步标记，直接结束")
            return {
                "observation_history": ["R1计划最后一步完成"],
                "is_complete": True,
                "should_continue": False,
            }
        
        # 优先级检查：如果 iteration_count 已达到或超过计划长度，强制结束
        if iteration_count >= len(query_plan):
            print(f"  ✅ iteration_count({iteration_count}) >= plan_length({len(query_plan)})，强制结束")
            return {
                "observation_history": ["R1计划全部完成"],
                "is_complete": True,
                "should_continue": False,
            }
        
        # 如果还有未执行的步骤，继续
        # 但需要从 thought_node 传递的 should_continue 判断
        # 如果 thought 已经设置 should_continue=False，就不要覆盖
        thought_should_continue = state.get('should_continue', True)
        print(f"  ➡️ 计划未完成，继续执行下一步 (thought_should_continue={thought_should_continue})")
        return {
            "observation_history": [f"R1计划第{iteration_count}步完成"],
            "is_complete": False,
            "should_continue": thought_should_continue,  # 使用thought节点的判断
        }
    
    # ========== 检测工具连续失败 ==========
    # 如果当前观察包含明显的错误信息，记录失败次数
    failed_tool_count = state.get("failed_tool_count", 0) or 0
    observation_str = str(latest_observation).lower()
    
    # 检测工具调用失败的特征
    is_tool_error = (
        "工具调用失败" in latest_observation or
        "mcp error" in observation_str or
        "error" in observation_str and ("站点代码查询失败" in latest_observation or "参数（dep）不符合要求" in latest_observation) or
        "无法查询" in latest_observation or
        "获取工具列表失败" in latest_observation
    )
    
    if is_tool_error:
        failed_tool_count += 1
        print(f"  ⚠️ 检测到工具失败，累计失败次数: {failed_tool_count}")
    else:
        # 成功调用，重置失败计数
        failed_tool_count = 0
    
    # 如果连续3次工具调用失败，强制结束循环
    if failed_tool_count >= 3:
        print(f"  🛑 连续 {failed_tool_count} 次工具调用失败，强制结束循环")
        return {
            "observation_history": [f"工具调用多次失败，使用已有数据生成答案"],
            "is_complete": True,
            "should_continue": False,
            "failed_tool_count": failed_tool_count,
        }
    
    # 构建所有已收集信息的摘要
    all_info = []
    if state.get("rag_results"):
        all_info.append("• RAG检索结果")
    if state.get("train_info"):
        all_info.append("• 火车票信息")
    if state.get("driving_info"):
        all_info.append("• 自驾路线")
    if state.get("hotel_info"):
        all_info.append("• 酒店信息")
    if state.get("weather_info"):
        all_info.append("• 天气信息")
    if state.get("lucky_day_info"):
        all_info.append("• 黄历信息")
    if state.get("flight_info"):
        all_info.append("• 航班信息")
    
    all_info_str = "\n".join(all_info) if all_info else "暂无信息"
    
    # 构造用户查询
    user_query_parts = []
    if state.get("destination"):
        user_query_parts.append(f"目的地:{state['destination']}")
    if state.get("origin"):
        user_query_parts.append(f"出发地:{state['origin']}")
    if state.get("travel_days"):
        user_query_parts.append(f"{state['travel_days']}天")
    if state.get("budget"):
        user_query_parts.append(f"预算{state['budget']}元")
    
    user_query = ", ".join(user_query_parts) if user_query_parts else "未知需求"
    
    # 使用 REACT_OBSERVATION_PROMPT
    prompt = REACT_OBSERVATION_PROMPT.format(
        user_query=user_query,
        all_collected_info=all_info_str,
        latest_observation=latest_observation
    )
    
    try:
        response = await qwen3_llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # 解析 JSON
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        
        if content.startswith("{"):
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
        
        evaluation = json.loads(content)
        
        eval_text = evaluation.get("evaluation", "")
        is_sufficient = evaluation.get("is_sufficient", False)
        missing_info = evaluation.get("missing_info", "")
        should_continue = evaluation.get("should_continue", True)
        
        print(f"\n📊 评估: {eval_text}")
        print(f"✅ 信息充分: {is_sufficient}")
        print(f"⚠️  缺失信息: {missing_info}")
        print(f"➡️  继续循环: {should_continue}")
        print(f"{'='*60}\n")
        
        return {
            "observation_history": [eval_text],
            "is_complete": is_sufficient,
            "should_continue": should_continue and not is_sufficient,
            "information_gaps": [missing_info] if missing_info else [],
            "failed_tool_count": failed_tool_count,  # 传递失败计数
        }
    
    except Exception as e:
        print(f"❌ 观察节点异常: {e}")
        import traceback
        traceback.print_exc()
        
        # 失败时默认继续，但不要无限循环
        iteration_count = state.get("iteration_count", 0) or 0
        max_iterations = state.get("max_iterations", 8) or 8
        
        if iteration_count >= max_iterations - 1:
            # 快达到最大迭代，强制结束
            return {
                "observation_history": [f"观察失败，已达最大迭代，结束循环"],
                "is_complete": True,
                "should_continue": False,
            }
        else:
            return {
                "observation_history": [f"观察失败: {str(e)}"],
                "is_complete": False,
                "should_continue": True,
            }
