#!/usr/bin/env python3
"""
MCP 服务器健康检查工具
快速诊断各个 MCP Server 的连接状态和可用性
"""
import asyncio
import json
from aggentic_RAG.travel_agent.tools.mcp_tools import get_mcp_manager


async def check_server_health(server_name: str) -> dict:
    """检查单个 MCP Server 的健康状态"""
    try:
        manager = await get_mcp_manager()
        
        # 检查服务器是否注册
        if server_name not in manager.mcp_servers:
            return {
                "server": server_name,
                "status": "❌ 未注册",
                "error": "服务器未在 MCPManager 中注册"
            }
        
        # 尝试列出工具
        tools = await manager.list_tools(server_name)
        
        if not tools:
            return {
                "server": server_name,
                "status": "⚠️ 连接但无工具",
                "tools_count": 0,
                "error": "无法获取工具列表"
            }
        
        # 解析工具详情
        tool_names = []
        if isinstance(tools, list):
            for tool in tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                elif isinstance(tool, dict):
                    tool_names.append(tool.get('name', '未知'))
        
        return {
            "server": server_name,
            "status": "✅ 正常",
            "tools_count": len(tool_names),
            "tools": tool_names[:5]  # 只显示前5个工具
        }
    
    except Exception as e:
        return {
            "server": server_name,
            "status": "❌ 连接失败",
            "error": str(e)
        }


async def main():
    """执行健康检查"""
    print("=" * 60)
    print("🔍 MCP 服务器健康检查")
    print("=" * 60)
    print()
    
    # 需要检查的服务器列表
    servers = [
        "12306 Server",
        "Gaode Server",
        "bazi Server",
        "flight Server",
    ]
    
    results = []
    for server in servers:
        print(f"检查 {server}...")
        result = await check_server_health(server)
        results.append(result)
        print()
    
    # 打印汇总报告
    print("=" * 60)
    print("📊 健康检查报告")
    print("=" * 60)
    print()
    
    for result in results:
        print(f"🔧 {result['server']}")
        print(f"   状态: {result['status']}")
        
        if result['status'] == "✅ 正常":
            print(f"   工具数: {result['tools_count']}")
            print(f"   示例工具: {', '.join(result.get('tools', []))}")
        elif 'error' in result:
            print(f"   错误: {result['error']}")
        
        print()
    
    # 统计
    normal_count = sum(1 for r in results if r['status'] == "✅ 正常")
    warning_count = sum(1 for r in results if r['status'].startswith("⚠️"))
    error_count = sum(1 for r in results if r['status'].startswith("❌"))
    
    print("=" * 60)
    print(f"✅ 正常: {normal_count}  |  ⚠️ 警告: {warning_count}  |  ❌ 错误: {error_count}")
    print("=" * 60)
    print()
    
    # 给出建议
    if error_count > 0:
        print("⚠️ 建议:")
        print("1. 检查 servers_config.json 配置是否正确")
        print("2. 确认对应的 MCP Server 进程是否正在运行")
        print("3. 查看后端日志查找具体错误信息")
        print("4. 尝试重启后端服务")
    elif warning_count > 0:
        print("⚠️ 部分服务器可用但工具列表为空，可能需要检查配置")
    else:
        print("✅ 所有 MCP 服务器状态正常！")


if __name__ == "__main__":
    asyncio.run(main())
