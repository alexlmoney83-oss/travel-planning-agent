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
    r1_strategy_node,  # 新增：R1战略规划节点
    r1_optimization_node,  # 新增：R1优化节点
    synthesizer_node,
)

# ReAct 节点（暂时占位，待实现）
async def thought_node_placeholder(state: TravelPlanState) -> dict:
    """ReAct 思考节点占位符"""
    return {"current_thought": "思考节点待实现"}

async def action_node_placeholder(state: TravelPlanState) -> dict:
    """ReAct 行动节点占位符"""
    return {"current_observation": "行动节点待实现"}

async def observation_node_placeholder(state: TravelPlanState) -> dict:
    """ReAct 观察节点占位符"""
    return {"is_complete": False, "should_continue": True}


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


def react_router(state: TravelPlanState) -> str:
    """ReAct 循环路由函数"""
    iteration_count = state.get("iteration_count", 0) or 0
    max_iterations = state.get("max_iterations", 5) or 5
    is_complete = state.get("is_complete", False)
    should_continue = state.get("should_continue", True)
    
    # 超过最大迭代次数
    if iteration_count >= max_iterations:
        print(f"⚠️ 达到最大迭代次数 {max_iterations}，结束循环")
        return "synthesizer"
    
    # 信息已充分
    if is_complete:
        print("✅ 信息已充分，结束循环")
        return "synthesizer"
    
    # 继续循环
    if should_continue:
        return "thought"
    else:
        return "synthesizer"


def create_react_workflow():
    """
    创建 ReAct Agentic RAG 工作流
    真正的 Agentic RAG：Agent 动态决定使用哪些工具、何时使用
    """
    from travel_agent.graph.nodes import (
        planner_node,
        r1_strategy_node,
        thought_node,
        action_node,
        observation_node,
        r1_optimization_node,
        synthesizer_node,
    )
    
    workflow = StateGraph(TravelPlanState)
    
    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("r1_strategy", r1_strategy_node)  # R1第一次介入：战略规划
    workflow.add_node("thought", thought_node)
    workflow.add_node("action", action_node)
    workflow.add_node("observation", observation_node)
    workflow.add_node("r1_optimization", r1_optimization_node)  # R1第二次介入：优化分析
    workflow.add_node("synthesizer", synthesizer_node)
    
    # 设置入口点
    workflow.set_entry_point("planner")
    
    # Planner 之后的路由（双模型协同）
    workflow.add_conditional_edges(
        "planner",
        route_after_planner_react,
        {
            "end": END,
            "r1_strategy": "r1_strategy",  # 复杂场景：先调用R1
            "react_loop": "thought",  # 简单场景：直接进入ReAct
        }
    )
    
    # R1 Strategy 之后进入 ReAct 循环（需要初始化状态）
    # 使用中间节点来初始化
    def init_react_after_r1(state: TravelPlanState) -> dict:
        """R1 Strategy 后初始化 ReAct 循环状态"""
        update = {}
        if state.get("iteration_count") is None or state.get("iteration_count") == 0:
            update["iteration_count"] = 0
        if state.get("max_iterations") is None:
            # R1模式需要更多迭代（多目的地查询步骤更多）
            query_plan_length = len(state.get('r1_plan', {}).get('query_plan', []))
            # 至少 15 次，或查询计划长度 + 5（留出凗余量）
            max_iter = max(15, query_plan_length + 5)
            update["max_iterations"] = max_iter
            print(f"  🔄 初始化 ReAct 状态: max_iterations={max_iter}")
        return update
    
    workflow.add_node("init_react", init_react_after_r1)
    workflow.add_edge("r1_strategy", "init_react")
    workflow.add_edge("init_react", "thought")
    
    # ReAct 循环：thought → action → observation → (thought or r1_optimization or synthesizer)
    workflow.add_edge("thought", "action")
    workflow.add_edge("action", "observation")
    
    # Observation 之后的路由（双模型协同）
    def route_after_observation(state: TravelPlanState) -> str:
        """
        Observation后的路由决策
        
        - 如果信息未充分：继续 thought 循环
        - 如果信息充分 + 有R1计划：调用R1优化
        - 其他：直接生成答案
        """
        print(f"\n{'='*60}")
        print("📢 [ROUTE_AFTER_OBSERVATION] 路由决策...")
        print(f"{'='*60}")
        
        is_complete = state.get('is_complete', False)
        should_continue = state.get('should_continue', True)
        has_r1_plan = state.get('r1_plan') is not None
        travel_segments = state.get('travel_segments', [])
        iteration_count = state.get('iteration_count', 0)
        
        # R1模式下，检查query_plan进度
        r1_plan = state.get('r1_plan')
        query_plan_length = len(r1_plan.get('query_plan', [])) if r1_plan else 0
        
        print(f"　📊 状态检查:")
        print(f"　　is_complete = {is_complete}")
        print(f"　　should_continue = {should_continue}")
        print(f"　　iteration_count = {iteration_count}")
        print(f"　　has_r1_plan = {has_r1_plan}")
        print(f"　　query_plan_length = {query_plan_length}")
        print(f"　　travel_segments count = {len(travel_segments) if travel_segments else 0}")
        
        # ==== 关键修复1：R1模式下，强制检查 iteration_count vs query_plan_length ====
        # 这是最可靠的终止条件
        if has_r1_plan and query_plan_length > 0 and iteration_count >= query_plan_length:
            print(f"　✅ R1计划已完成: iteration_count({iteration_count}) >= query_plan_length({query_plan_length})")
            # 递归计算：planner(1) + r1_strategy(1) + init_react(1) + query_plan×3 + r1_opt(1) + synth(1)
            # 单目的地（6步）: 3 + 18 + 2 = 23 ≤ 25 ✅ 可以使用 R1 optimization
            # 多目的地（7步）: 3 + 21 + 2 = 26 > 25 ❌ 跳过 R1 optimization
            is_multi_destination = len(travel_segments) > 1
            if is_multi_destination:
                print(f"　🎯 多目的地场景，跳过 R1 Optimization，直接进入 Synthesizer")
                print(f"{'='*60}\n")
                return "synthesizer"
            else:
                print(f"　🔍 单目的地场景，调用 R1 Optimization 进行优化分析")
                print(f"{'='*60}\n")
                return "r1_optimization"
        
        # ==== 关键修复2：同时检查 is_complete 和 should_continue ====
        # 如果 should_continue=False，不管 is_complete，都应该结束循环
        if not should_continue:
            print(f"　✅ should_continue=False，准备结束循环")
            # 跳过下面的继续循环判断
        elif not is_complete:
            # 信息未充分且应该继续
            print(f"　➡️ 信息未充分，继续 thought 循环")
            print(f"{'='*60}\n")
            return "thought"
        else:
            print(f"　✅ is_complete=True，准备结束循环")
        
        # 信息充分或 should_continue=False，准备结束循环
        # 根据目的地数量决定是否调用 R1 optimization
        is_multi_destination = len(travel_segments) > 1
        if has_r1_plan and not is_multi_destination:
            # 单目的地复杂场景：递归步数有余量，可以调用 R1 optimization
            print(f"　🔍 单目的地复杂场景，调用 R1 Optimization 进行优化分析")
            print(f"{'='*60}\n")
            return "r1_optimization"
        else:
            # 多目的地或简单场景：直接生成
            if is_multi_destination:
                print(f"　🎯 多目的地场景，跳过 R1 Optimization，直接进入 Synthesizer")
            else:
                print(f"　🎯 简单场景，直接进入 Synthesizer")
            print(f"{'='*60}\n")
            return "synthesizer"
    
    workflow.add_conditional_edges(
        "observation",
        route_after_observation,
        {
            "thought": "thought",  # 继续循环
            "r1_optimization": "r1_optimization",  # R1优化
            "synthesizer": "synthesizer",  # 直接生成
        }
    )
    
    # R1优化后进入Synthesizer
    workflow.add_edge("r1_optimization", "synthesizer")
    
    # 生成最终答案
    workflow.add_edge("synthesizer", END)
    
    # 编译工作流
    # 注意：LangGraph API/服务调用时会使用默认的 recursion_limit=25
    # 为了支持复杂的多段行程（9步+ × 3节点/步），需要增加限制
    # 通过环境变量或调用时的config传递
    return workflow.compile()


def route_after_planner_react(state: TravelPlanState) -> str:
    """
规划器之后的路由决策（ReAct 版本）
    
    双模型协同路由：
    - 如果需要clarification：直接结束
    - 如果检测到复杂场景（needs_deep_analysis=True）：先调用R1 Strategy
    - 其他简单场景：直接进入ReAct循环（Qwen3主导）
    """
    # 如果需要用户澄清信息，直接结束并返回问题
    if state.get("needs_clarification", False):
        return "end"
    
    # 检测复杂场景：多目的地、预算优化等
    # 这些场景需要R1的深度推理
    if state.get("needs_deep_analysis", False):
        print("  🧠 检测到复杂场景，调用 R1 Strategy 进行深度规划")
        return "r1_strategy"
    
    # 简单场景：初始化 ReAct 循环状态，Qwen3主导
    print("🤖 简单场景，进入 Qwen3 主导的 ReAct 循环")
    if state.get("iteration_count") is None:
        state["iteration_count"] = 0
    if state.get("max_iterations") is None:
        state["max_iterations"] = 12  # 单目的地通常需要 5-7 次，留余量到 12 次
    
    # 进入 ReAct 循环
    return "react_loop"


# 创建全局工作流实例
# 旧的固定流程工作流（保留作为备用）
# travel_workflow = create_travel_workflow()

# 新的 ReAct Agentic RAG 工作流（默认启用）
travel_workflow = create_react_workflow()

# 如果需要回滚到旧的固定流程，将上面两行注释交换即可
