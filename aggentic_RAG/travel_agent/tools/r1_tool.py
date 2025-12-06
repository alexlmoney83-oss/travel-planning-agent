"""
DeepSeek R1深度分析工具
"""
from openai import AsyncOpenAI
import json
from typing import Dict, Any, Optional

from ..config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    R1_MODEL,
    R1_TEMPERATURE,
)
from ..config.prompts import R1_ANALYSIS_PROMPT_TEMPLATE


class DeepSeekR1Analyzer:
    """DeepSeek R1深度分析器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.client = AsyncOpenAI(
            base_url=base_url or DEEPSEEK_BASE_URL,
            api_key=api_key or DEEPSEEK_API_KEY
        )
        self.model = R1_MODEL
        self.temperature = R1_TEMPERATURE
    
    async def analyze(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """深度分析复杂问题
        
        Args:
            problem: 需要分析的问题
            context: 上下文信息
            
        Returns:
            JSON格式的分析结果
        """
        prompt = R1_ANALYSIS_PROMPT_TEMPLATE.format(
            problem=problem,
            context=json.dumps(context or {}, ensure_ascii=False, indent=2)
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_result = {
                "analysis": f"分析失败: {str(e)}",
                "constraints": [],
                "suggestions": [],
                "reasoning": "无法完成深度分析"
            }
            return json.dumps(error_result, ensure_ascii=False)
    
    async def optimize_route(
        self,
        destinations: list,
        budget: float,
        days: int
    ) -> Dict[str, Any]:
        """优化旅行路线"""
        problem = f"""
        请优化以下旅行路线：
        - 目的地清单: {', '.join(destinations)}
        - 预算限制: {budget}元
        - 时间限制: {days}天
        
        要求：在预算和时间限制内，给出最优的游览顺序和每个地点的停留时间。
        """
        
        context = {
            "destinations": destinations,
            "budget": budget,
            "days": days
        }
        
        result = await self.analyze(problem, context)
        
        try:
            return json.loads(result)
        except:
            return {"error": "优化失败", "raw_response": result}


# 全局R1实例
_r1_instance = None


def get_r1_instance() -> DeepSeekR1Analyzer:
    """获取全局R1实例"""
    global _r1_instance
    if _r1_instance is None:
        _r1_instance = DeepSeekR1Analyzer()
    return _r1_instance
