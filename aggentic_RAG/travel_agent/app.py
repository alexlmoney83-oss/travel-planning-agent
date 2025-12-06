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
        "rag_results": None,
        "train_info": None,
        "weather_info": None,
        "reasoning_chain": None,
        "optimization_suggestions": None,
        "needs_deep_analysis": False,
        "tools_needed": None,
        "travel_plan": None,
    }
    
    # 运行工作流
    result = await travel_workflow.ainvoke(initial_state)
    
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
