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
    
    # 工具调用结果
    rag_results: Optional[str]  # RAG检索结果
    train_info: Optional[Dict[str, Any]]  # 火车票信息
    driving_info: Optional[str]  # 自驾路线信息
    weather_info: Optional[Dict[str, Any]]  # 天气信息
    hotel_info: Optional[str]  # 酒店/民宿信息
    lucky_day_info: Optional[str]  # 黄历吉日信息
    
    # R1分析结果
    reasoning_chain: Optional[str]  # 推理链
    optimization_suggestions: Optional[List[str]]  # 优化建议
    
    # 控制流
    query_mode: Optional[str]  # 查询模式: "simple"（简单查询）或 "full"（完整规划）
    needs_deep_analysis: bool  # 是否需要R1深度分析
    tools_needed: Optional[List[str]]  # 需要的工具
    needs_clarification: Optional[bool]  # 是否需要用户澄清信息
    clarification_question: Optional[str]  # 需要询问用户的问题
    
    # 最终输出
    travel_plan: Optional[str]  # 最终方案
