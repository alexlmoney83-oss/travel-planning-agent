"""
MCP工具封装 - 基于client.py实现
"""
from langchain.tools import Tool
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from agents.mcp import MCPServerSse
import json
import os
import ssl
import httpx
from pathlib import Path

from ..config.settings import PROJECT_ROOT, MCP_CONFIG_PATH

# 绕过代理直连ModelScope（避免VPN代理导致SSL问题）
os.environ['NO_PROXY'] = os.environ.get('NO_PROXY', '') + ',modelscope.net,api-inference.modelscope.net'

# 创建不验证SSL的httpx客户端（仅用于开发/测试）
def create_insecure_httpx_client():
    """创建禁用SSL验证的httpx客户端"""
    return httpx.AsyncClient(verify=False, timeout=30.0)


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
                print(f"✅ 已连接MCP服务器: {name}")
                
                # 列出可用工具
                try:
                    tools = await self.list_tools(name)
                    print(f"   可用工具: {', '.join(tools) if tools else '无'}")
                except Exception as e:
                    print(f"   ⚠️ 无法获取工具列表: {e}")
            except Exception as e:
                print(f"❌ 连接失败 {name}: {e}")
    
    async def call_tool(self, server_name: str, tool_name: str, **kwargs) -> str:
        """调用MCP工具"""
        if server_name not in self.mcp_servers:
            return json.dumps({
                "error": f"MCP服务器 {server_name} 未连接",
                "available_servers": list(self.mcp_servers.keys())
            }, ensure_ascii=False)
        
        try:
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
            return json.dumps({
                "error": f"工具调用失败: {str(e)}",
                "server": server_name,
                "tool": tool_name
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
                print("✅ MCP连接已关闭")
            except Exception as e:
                print(f"⚠️ MCP清理警告: {e}")


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
