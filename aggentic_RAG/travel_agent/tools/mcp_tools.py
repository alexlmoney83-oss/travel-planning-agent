"""
MCP工具封装 - 基于client.py实现
"""
from langchain.tools import Tool
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
import json
import os
import ssl
import httpx
from pathlib import Path
import sys
import warnings

# suppress async generator warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*async_generator.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*generator didn\'t stop.*')

# 尝试导入agents.mcp，如果失败提供详细错误
try:
    from agents.mcp import MCPServerSse
except ImportError as e:
    print(f"\n❌ 导入agents.mcp失败: {e}")
    print(f"🔍 Python解释器: {sys.executable}")
    print(f"🔍 sys.path前5项:")
    for i, p in enumerate(sys.path[:5]):
        print(f"  {i+1}. {p}")
    
    # 尝试查找agents包是否存在
    try:
        import agents
        print(f"✅ agents包找到: {agents.__file__}")
        print(f"❌ 但agents.mcp模块不存在")
    except ImportError:
        print(f"❌ agents包未安装")
    
    raise ImportError(
        f"\n\nopenai-agents包未正确安装或agents.mcp模块不可用\n"
        f"Python: {sys.executable}\n"
        f"请运行: pip install openai-agents"
    ) from e

from ..config.settings import PROJECT_ROOT, MCP_CONFIG_PATH

# 绕过代理直连ModelScope（避免VPN代理导致SSL问题）
os.environ['NO_PROXY'] = os.environ.get('NO_PROXY', '') + ',modelscope.net,api-inference.modelscope.net'

# 创建不验证SSL的httpx客户端（仅用于开发/测试）
def create_insecure_httpx_client():
    """创建禁用SSL验证的httpx客户端"""
    return httpx.AsyncClient(
        verify=False, 
        timeout=60.0,  # 增加超时时间到60秒
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)  # 增加连接池
    )


class MCPToolManager:
    """MCP工具管理器 - 管理所有MCP服务器连接"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or MCP_CONFIG_PATH
        self.mcp_servers = {}
        self.exit_stack = None
        
    async def initialize(self):
        """初始化所有MCP服务器连接"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"MCP配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.exit_stack = AsyncExitStack()
        
        for server_conf in config.get("mcp_servers", []):
            name = server_conf.get("name")
            url = server_conf.get("url")
            
            if not url:
                print(f"⚠️ 警告: 服务器 {name} 缺少URL，跳过")
                continue
            
            try:
                server = await self.exit_stack.enter_async_context(
                    MCPServerSse(name=name, params={"url": url})
                )
                self.mcp_servers[name] = server
                # 静默模式，不输出连接日志
            except Exception as e:
                # 静默失败，不输出错误
                pass
    
    async def call_tool(self, server_name: str, tool_name: str, max_retries: int = 2, **kwargs) -> str:
        """
        调用MCP工具，带重试机制
        
        Args:
            server_name: MCP服务器名称
            tool_name: 工具名称
            max_retries: 最大重试次数（默认2次）
            **kwargs: 工具参数
        """
        if server_name not in self.mcp_servers:
            return json.dumps({
                "error": f"MCP服务器 {server_name} 未连接",
                "available_servers": list(self.mcp_servers.keys())
            }, ensure_ascii=False)
        
        import asyncio
        last_error = None
        
        # 重试逻辑
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"  🔄 第{attempt}次重试 {server_name}.{tool_name}...")
                    await asyncio.sleep(1 * attempt)  # 指数退避: 1s, 2s
                
                result = await self.mcp_servers[server_name].call_tool(
                    tool_name, 
                    arguments=kwargs
                )
                
                # 处理MCP返回的CallToolResult对象
                if hasattr(result, 'content'):
                    # 提取content字段
                    content = result.content
                    if isinstance(content, list) and len(content) > 0:
                        # 如果content是列表，提取第一个元素的text
                        if hasattr(content[0], 'text'):
                            return content[0].text
                        else:
                            return str(content[0])
                    elif isinstance(content, str):
                        return content
                    else:
                        return json.dumps(content, ensure_ascii=False, indent=2)
                else:
                    # 如果没有content属性，尝试直接序列化
                    return json.dumps(result, ensure_ascii=False, indent=2, default=str)
                    
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # 判断是否是SSE连接中断错误，可重试
                is_retryable = any([
                    "peer closed connection" in error_str,
                    "incomplete chunked read" in error_str,
                    "remoteprotocolerror" in error_str,
                    "timeout" in error_str,
                    "connection reset" in error_str
                ])
                
                if is_retryable and attempt < max_retries:
                    print(f"  ⚠️ [MCP错误] {server_name}.{tool_name} - {type(e).__name__}")
                    print(f"     原因: SSE连接中断，将重试...")
                    continue  # 重试
                else:
                    # 不可重试或已达最大重试次数
                    break
        
        # 所有重试均失败，记录错误
        import traceback
        error_msg = f"工具调用失败: {str(last_error)}"
        print(f"\n❌ [MCP错误] {server_name}.{tool_name} (重试{max_retries}次后仍失败)")
        print(f"   错误: {str(last_error)}")
        print(f"   类型: {type(last_error).__name__}")
        if "peer closed" in str(last_error).lower():
            print(f"   原因: SSE连接中断，可能是MCP Server负载过高或返回数据过大")
        elif "timeout" in str(last_error).lower():
            print(f"   原因: 网络超时，请检查MCP服务器是否正常运行")
        
        return json.dumps({
            "error": error_msg,
            "server": server_name,
            "tool": tool_name,
            "error_type": type(last_error).__name__,
            "retries": max_retries
        }, ensure_ascii=False)
    
    async def list_tools(self, server_name: str) -> List[str]:
        """列出指定服务器的可用工具"""
        if server_name not in self.mcp_servers:
            return []
        
        try:
            tools = await self.mcp_servers[server_name].list_tools()
            tool_names = []
            for tool in tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                elif hasattr(tool, 'function') and hasattr(tool.function, 'name'):
                    tool_names.append(tool.function.name)
                elif isinstance(tool, dict) and 'name' in tool:
                    tool_names.append(tool['name'])
                else:
                    tool_names.append(str(tool))
            return tool_names
        except Exception as e:
            print(f"获取工具列表失败: {e}")
            return []
    
    async def cleanup(self):
        """清理资源"""
        if self.exit_stack:
            try:
                await self.exit_stack.aclose()
            except Exception:
                # 静默忽略清理错误
                pass


# 全局MCP管理器实例
_mcp_manager = None


async def get_mcp_manager() -> MCPToolManager:
    """获取全局MCP管理器实例"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPToolManager()
        await _mcp_manager.initialize()
    return _mcp_manager


def create_12306_tool(manager: MCPToolManager) -> Tool:
    """创建12306工具 - 获取当前日期"""
    
    async def get_current_date() -> str:
        """获取当前日期
        
        Returns:
            当前日期 (YYYY-MM-DD格式)
        """
        return await manager.call_tool(
            "12306 Server",
            "get-current-date"
        )
    
    return Tool(
        name="12306获取日期",
        description="获取当前日期，用于查询火车票等操作",
        func=lambda *args, **kwargs: "请使用异步调用",
        coroutine=get_current_date
    )


def create_gaode_tool(manager: MCPToolManager) -> Tool:
    """创建高德地图地理编码工具"""
    
    async def geocode_address(address: str) -> str:
        """地理编码 - 将地址转换为经纬度
        
        Args:
            address: 地址名称（如"北京天安门"）
            
        Returns:
            地理编码结果（包含经纬度、行政区划等）
        """
        return await manager.call_tool(
            "Gaode Server",
            "maps_geo",
            address=address
        )
    
    return Tool(
        name="高德地图",
        description="将地址转换为经纬度坐标。输入: address(地址名称)",
        func=lambda *args, **kwargs: "请使用异步调用",
        coroutine=geocode_address
    )


async def get_all_mcp_tools() -> List[Tool]:
    """获取所有MCP工具"""
    manager = await get_mcp_manager()
    
    return [
        create_12306_tool(manager),
        create_gaode_tool(manager),
    ]
