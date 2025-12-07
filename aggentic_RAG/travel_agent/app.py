"""
旅游规划Agent主应用入口
"""
import asyncio
from langchain_core.messages import HumanMessage
from .graph.workflow import travel_workflow
from .graph.state import TravelPlanState


async def run_travel_agent(user_query: str):
    """运行旅游规划Agent
    
    Args:
        user_query: 用户查询
        
    Returns:
        最终的旅行方案
    """
    # 初始化状态
    initial_state = {
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "destination": None,
        "origin": None,
        "travel_days": None,
        "budget": None,
        "travel_date": None,
        "preferences": None,
        
        # 单次结果（旧模式，向后兼容）
        "rag_results": None,
        "train_info": None,
        "weather_info": None,
        "hotel_info": None,
        "driving_info": None,
        "lucky_day_info": None,
        
        # R1 分析
        "reasoning_chain": None,
        "optimization_suggestions": None,
        "needs_deep_analysis": False,
        "tools_needed": None,
        
        # ReAct 循环状态
        "iteration_count": 0,
        "max_iterations": 8,
        "current_thought": None,
        "current_action": None,
        "current_observation": None,
        "should_continue": True,
        "is_complete": False,
        
        # ReAct 累积历史（列表）
        "thought_history": [],
        "action_history": [],
        "observation_history": [],
        "rag_results_history": [],
        "tool_results_history": [],
        
        # ReAct 工具管理
        "available_tools": None,
        "tool_call_count": {},
        "information_gaps": [],
        
        # 最终输出
        "travel_plan": None,
    }
    
    # 运行工作流
    # 设置 recursion_limit=100 以支持复杂的多目的地场景
    # 8步计划 × 3节点/步 = 24, 再加上其他节点，需要更大的限制
    result = await travel_workflow.ainvoke(
        initial_state,
        config={"recursion_limit": 100}
    )
    
    return result


async def main():
    """测试主函数"""
    test_query = "帮我规划从北京到上海的3天旅游，预算3000元，12月10日出发"
    
    print(f"🎯 查询: {test_query}\n")
    
    result = await run_travel_agent(test_query)
    
    print("\n✅ 规划完成！")
    print("\n📋 最终方案:")
    print(result.get("travel_plan", "未生成方案"))


if __name__ == "__main__":
    asyncio.run(main())
