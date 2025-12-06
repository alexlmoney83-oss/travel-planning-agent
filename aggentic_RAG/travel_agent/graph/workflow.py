"""
LangGraph工作流编排
"""
from langgraph.graph import StateGraph, END
from travel_agent.graph.state import TravelPlanState
from travel_agent.graph.nodes import (
    planner_node,
    rag_search_node,
    train_query_node,
    lucky_day_query_node,
    weather_query_node,
    deep_analysis_node,
    synthesizer_node,
)


def should_use_r1(state: TravelPlanState) -> str:
    """判断是否需要R1深度分析"""
    if state.get("needs_deep_analysis", False):
        return "deep_analysis"
    return "tool_execution"


def route_after_planner(state: TravelPlanState) -> str:
    """规划器之后的路由决策"""
    # 如果需要用户澄清信息，直接结束并返回问题
    if state.get("needs_clarification", False):
        return "end"
    
    if state.get("needs_deep_analysis", False):
        return "deep_analysis"
    
    # 默认进入 RAG 检索
    return "rag_search"


def route_after_rag(state: TravelPlanState) -> str:
    """检索后的路由：根据查询模式决定是否查询交通"""
    query_mode = state.get("query_mode", "full")
    
    if query_mode == "simple":
        # 简单查询：直接跳到整合节点
        return "synthesizer"
    else:
        # 完整规划：继续查询交通
        return "train_query"


def create_travel_workflow():
    """创建旅游规划工作流"""
    workflow = StateGraph(TravelPlanState)
    
    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("rag_search", rag_search_node)
    workflow.add_node("train_query", train_query_node)
    workflow.add_node("lucky_day_query", lucky_day_query_node)
    workflow.add_node("weather_query", weather_query_node)
    workflow.add_node("deep_analysis", deep_analysis_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # 设置入口点
    workflow.set_entry_point("planner")
    
    # 添加条件路由
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "end": END,
            "deep_analysis": "deep_analysis",
            "rag_search": "rag_search",
        }
    )
    
    # 深度分析后继续工具调用
    workflow.add_edge("deep_analysis", "rag_search")
    
    # RAG检索后根据模式选择路径
    workflow.add_conditional_edges(
        "rag_search",
        route_after_rag,
        {
            "synthesizer": "synthesizer",  # 简单模式：直接整合
            "train_query": "train_query",  # 完整模式：查询交通
        }
    )
    
    # 完整规划模式的工具节点链
    workflow.add_edge("train_query", "lucky_day_query")
    workflow.add_edge("lucky_day_query", "weather_query")
    workflow.add_edge("weather_query", "synthesizer")
    
    # 整合完成后结束
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# 创建全局工作流实例
travel_workflow = create_travel_workflow()
