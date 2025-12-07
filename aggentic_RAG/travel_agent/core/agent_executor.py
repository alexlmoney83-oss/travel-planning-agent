"""
核心Agent执行引擎 - 替代LangGraph workflow
无递归限制，支持R1主导和Qwen3主导两种模式
"""
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime


class AgentExecutor:
    """
    旅游规划Agent执行引擎
    
    核心特性：
    - 无递归限制（可设置max_iterations=100+）
    - 支持R1主导模式（按query_plan执行）
    - 支持Qwen3主导模式（自主ReAct）
    - 实时状态回调（用于UI更新）
    """
    
    def __init__(
        self,
        max_iterations: int = 50,
        status_callback: Optional[callable] = None
    ):
        """
        初始化Agent执行引擎
        
        Args:
            max_iterations: 最大迭代次数
            status_callback: 状态更新回调函数，用于UI实时显示
                             signature: callback(step_name: str, status: str, result: Any)
        """
        self.max_iterations = max_iterations
        self.status_callback = status_callback
        
    async def execute(self, user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        执行旅游规划任务
        
        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            
        Returns:
            包含travel_plan和执行日志的字典
        """
        # 初始化状态
        state = self._initialize_state(user_query, conversation_history)
        
        try:
            # 阶段1: Planner - 理解用户需求
            await self._update_status("planner", "正在理解您的需求...", None)
            state = await self._planner_phase(state)
            
            # 检查是否需要用户澄清
            if state.get("needs_clarification"):
                return {
                    "travel_plan": state.get("clarification_questions", "请提供更多信息"),
                    "execution_log": state["execution_log"],
                    "state": state
                }
            
            # 阶段2: 策略规划（可选）
            if state.get("needs_deep_analysis"):
                await self._update_status("r1_strategy", "正在进行深度分析和战略规划...", None)
                state = await self._r1_strategy_phase(state)
            
            # 阶段3: ReAct循环 - 收集信息
            state = await self._react_phase(state)
            
            # 阶段4: R1优化（可选，仅单目的地复杂场景）
            if self._should_optimize(state):
                await self._update_status("r1_optimization", "正在优化旅行方案...", None)
                state = await self._r1_optimization_phase(state)
            
            # 阶段5: Synthesizer - 生成最终方案
            await self._update_status("synthesizer", "正在生成旅行方案...", None)
            state = await self._synthesizer_phase(state)
            
            return {
                "travel_plan": state.get("travel_plan"),
                "execution_log": state["execution_log"],
                "state": state
            }
            
        except Exception as e:
            await self._update_status("error", f"执行出错: {str(e)}", None)
            return {
                "travel_plan": f"抱歉，处理您的请求时出现错误：{str(e)}",
                "execution_log": state.get("execution_log", []),
                "state": state,
                "error": str(e)
            }
    
    def _initialize_state(self, user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """初始化状态"""
        return {
            "user_query": user_query,
            "messages": conversation_history or [],
            "destination": "",
            "origin": "",
            "travel_days": 0,
            "budget": 0,
            "travel_date": "",
            "preferences": [],
            "rag_results": "",
            "train_info": "",
            "hotel_info": "",
            "driving_info": "",
            "flight_info": "",
            "weather_info": "",
            "lucky_day_info": "",
            "r1_plan": None,
            "travel_segments": [],
            "iteration_count": 0,
            "execution_log": [],  # 记录每个步骤的执行情况
            "needs_clarification": False,
            "needs_deep_analysis": False,
            "scenario_type": "simple_query"
        }
    
    async def _planner_phase(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Planner阶段 - 提取用户需求"""
        from travel_agent.graph.nodes import planner_node
        
        result = await planner_node(state)
        state.update(result)
        
        self._log_execution(state, "planner", "完成", {
            "scenario_type": state.get("scenario_type"),
            "destination": state.get("destination"),
            "needs_deep_analysis": state.get("needs_deep_analysis")
        })
        
        return state
    
    async def _r1_strategy_phase(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """R1 Strategy阶段 - 深度分析和战略规划"""
        from travel_agent.graph.nodes import r1_strategy_node
        
        result = await r1_strategy_node(state)
        state.update(result)
        
        self._log_execution(state, "r1_strategy", "完成", {
            "query_plan_steps": len(state.get("r1_plan", {}).get("query_plan", [])),
            "travel_segments": len(state.get("travel_segments", []))
        })
        
        return state
    
    async def _react_phase(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        ReAct循环阶段 - 执行工具调用收集信息
        
        支持两种模式：
        1. R1主导模式：按照query_plan顺序执行
        2. Qwen3主导模式：自主决策下一步
        """
        from travel_agent.graph.nodes import thought_node, action_node, observation_node
        
        iteration = 0
        r1_plan = state.get("r1_plan")
        query_plan = r1_plan.get("query_plan", []) if r1_plan else []
        
        # R1主导模式
        if query_plan:
            print(f"🤖 R1主导模式: 执行 {len(query_plan)} 步query_plan")
            
            for step_idx, step in enumerate(query_plan):
                iteration += 1
                state["iteration_count"] = iteration
                
                tool_name = step.get("tool", "")
                params = step.get("params", {})
                description = step.get("description", "")
                
                await self._update_status(
                    f"react_step_{iteration}",
                    f"[{iteration}/{len(query_plan)}] {description}",
                    None
                )
                
                print(f"  步骤 {iteration}: {tool_name} - {description}")
                
                # 直接执行action（跳过thought，因为已经有计划了）
                action_state = {
                    **state,
                    "current_action": {
                        "tool": tool_name,
                        "params": params,
                        "segment": step.get("segment", 0)
                    }
                }
                
                # Action
                action_result = await action_node(action_state)
                state.update(action_result)
                
                # Observation（更新结果到state）
                obs_result = await observation_node(state)
                state.update(obs_result)
                
                self._log_execution(state, f"react_step_{iteration}", tool_name, {
                    "params": params,
                    "description": description
                })
        
        # Qwen3主导模式
        else:
            print(f"🤖 Qwen3主导模式: 自主ReAct循环")
            
            while iteration < self.max_iterations:
                iteration += 1
                state["iteration_count"] = iteration
                
                # Thought: LLM决定下一步
                thought_result = await thought_node(state)
                state.update(thought_result)
                
                # 检查是否完成
                if not state.get("should_continue", True) or state.get("is_complete", False):
                    print(f"  Qwen3决定结束（迭代 {iteration}）")
                    break
                
                current_action = state.get("current_action", {})
                tool_name = current_action.get("tool", "")
                
                if tool_name == "final_answer":
                    print(f"  Qwen3选择final_answer（迭代 {iteration}）")
                    break
                
                await self._update_status(
                    f"react_step_{iteration}",
                    f"[{iteration}] 正在{current_action.get('description', tool_name)}...",
                    None
                )
                
                print(f"  步骤 {iteration}: {tool_name}")
                
                # Action
                action_result = await action_node(state)
                state.update(action_result)
                
                # Observation
                obs_result = await observation_node(state)
                state.update(obs_result)
                
                self._log_execution(state, f"react_step_{iteration}", tool_name, {
                    "params": current_action.get("params"),
                    "description": current_action.get("description", "")
                })
        
        return state
    
    async def _r1_optimization_phase(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """R1 Optimization阶段 - 优化分析（仅单目的地复杂场景）"""
        from travel_agent.graph.nodes import r1_optimization_node
        
        result = await r1_optimization_node(state)
        state.update(result)
        
        self._log_execution(state, "r1_optimization", "完成", {})
        
        return state
    
    async def _synthesizer_phase(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesizer阶段 - 生成最终方案"""
        from travel_agent.graph.nodes import synthesizer_node
        
        result = await synthesizer_node(state)
        state.update(result)
        
        self._log_execution(state, "synthesizer", "完成", {})
        
        return state
    
    def _should_optimize(self, state: Dict[str, Any]) -> bool:
        """判断是否需要R1优化"""
        has_r1_plan = state.get("r1_plan") is not None
        travel_segments = state.get("travel_segments", [])
        is_single_destination = len(travel_segments) <= 1
        
        # 只有单目的地复杂场景才调用R1 optimization
        return has_r1_plan and is_single_destination
    
    def _log_execution(self, state: Dict[str, Any], step_name: str, status: str, details: Dict):
        """记录执行日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "status": status,
            "details": details
        }
        state["execution_log"].append(log_entry)
    
    async def _update_status(self, step_name: str, status: str, result: Any):
        """更新状态（用于UI实时显示）"""
        if self.status_callback:
            await self.status_callback(step_name, status, result)
        
        print(f"[{step_name}] {status}")
