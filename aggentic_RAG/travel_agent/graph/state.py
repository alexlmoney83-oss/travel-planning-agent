"""
LangGraph状态定义
"""
from typing import TypedDict, List, Optional, Annotated, Dict, Any
from langchain_core.messages import BaseMessage
import operator


class TravelPlanState(TypedDict):
    """旅游规划状态"""
    # 用户输入
    user_query: Optional[str]
    
    # 对话历史
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 用户需求提取
    destination: Optional[str]  # 目的地
    origin: Optional[str]  # 出发地
    travel_days: Optional[int]  # 天数
    budget: Optional[float]  # 预算
    travel_date: Optional[str]  # 出发日期
    preferences: Optional[List[str]]  # 偏好
    raw_destination_text: Optional[str]  # 原始目的地文本（用于多目的地检测）
    
    # 工具调用结果（保留原有字段以兼容现有代码）
    rag_results: Optional[str]  # RAG检索结果（单次，向后兼容）
    train_info: Optional[Dict[str, Any]]  # 火车票信息
    driving_info: Optional[str]  # 自驾路线信息
    weather_info: Optional[Dict[str, Any]]  # 天气信息
    hotel_info: Optional[str]  # 酒店/民宿信息
    lucky_day_info: Optional[str]  # 黄历吉日信息
    
    # ReAct 循环：累积的工具调用结果
    rag_results_history: Annotated[List[str], operator.add]  # RAG检索结果历史（多次检索累积）
    tool_results_history: Annotated[List[Dict[str, Any]], operator.add]  # 其他工具调用结果历史
    
    # R1分析结果
    reasoning_chain: Optional[str]  # 推理链
    optimization_suggestions: Optional[Dict[str, Any]]  # 优化建议（改为Dict以存储budget_analysis等）
    risk_warnings: Optional[List[str]]  # R1识别的风险警告
    alternative_plans: Optional[List[Dict[str, Any]]]  # R1生成的替代方案
    value_comparison: Optional[List[Dict[str, Any]]]  # R1的性价比对比
    
    # R1战略规划（多目的地支持）
    r1_plan: Optional[Dict[str, Any]]  # R1制定的执行计划（包含travel_segments, budget_allocation, query_plan）
    travel_segments: Optional[List[Dict[str, Any]]]  # 多段行程 [{"origin": "上海", "destination": "青岛", "days": 3}, ...]
    scenario_type: Optional[str]  # 场景类型: "simple" | "multi_destination" | "budget_optimization"
    current_segment_index: Optional[int]  # 当前执行到第几段（0-based）
    
    # 多段查询结果（分段存储）
    segment_rag_results: Optional[Dict[int, str]]  # 分段RAG结果 {0: "青岛景点...", 1: "大连景点..."}
    segment_train_info: Optional[Dict[int, Dict[str, Any]]]  # 分段交通 {0: {上海→青岛}, 1: {青岛→大连}}
    segment_driving_info: Optional[Dict[int, str]]  # 分段自驾信息
    segment_hotel_info: Optional[Dict[int, str]]  # 分段酒店信息
    flight_info: Optional[Dict[str, Any]]  # 航班信息（保留原有字段）
    
    # 控制流
    query_mode: Optional[str]  # 查询模式: "simple"（简单查询）或 "full"（完整规划）
    needs_deep_analysis: bool  # 是否需要R1深度分析
    tools_needed: Optional[List[str]]  # 需要的工具
    needs_clarification: Optional[bool]  # 是否需要用户澄清信息
    clarification_question: Optional[str]  # 需要询问用户的问题
    
    # ReAct 循环状态
    thought_history: Annotated[List[str], operator.add]  # 思考历史
    current_thought: Optional[str]  # 当前思考内容
    action_history: Annotated[List[Dict[str, Any]], operator.add]  # 行动历史
    current_action: Optional[Dict[str, Any]]  # 当前行动 {"tool": "rag_search", "params": {...}}
    observation_history: Annotated[List[str], operator.add]  # 观察历史
    current_observation: Optional[str]  # 当前观察结果
    iteration_count: Optional[int]  # 当前迭代次数
    max_iterations: Optional[int]  # 最大迭代次数（防止死循环）
    should_continue: Optional[bool]  # 是否继续循环
    is_complete: Optional[bool]  # 信息是否已充分
    
    # ReAct 工具管理
    available_tools: Optional[List[str]]  # 当前可用的工具列表
    tool_call_count: Optional[Dict[str, int]]  # 每个工具调用次数统计 {"rag_search": 2, "train_query": 1}
    information_gaps: Optional[List[str]]  # 识别的信息缺口
    failed_tool_count: Optional[int]  # 连续失败的工具调用次数（用于防止无限重试）
    
    # 最终输出
    travel_plan: Optional[str]  # 最终方案
