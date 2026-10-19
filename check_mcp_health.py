#!/usr/bin/env python3
"""
MCP 服务器健康检查工具
快速诊断各个 MCP Server 的连接状态和可用性

修复说明（fork）：
- 原实现把 `manager.list_tools()` 的返回值当成对象/字典解析，但该方法实际返回
  的是工具名字符串列表（List[str]），导致**连接正常的服务器也会被判为"无工具"**；
- 服务器列表原先硬编码且漏掉 biying Server，改为从 servers_config.json 读取；
- 补充失败原因（来自 MCPToolManager.get_health()）、--json 输出与非零退出码，
  便于 CI / 脚本判断。
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from aggentic_RAG.travel_agent.tools.mcp_tools import get_mcp_manager


def load_server_names() -> list:
    """从配置文件读取服务器名（含可选服务器），配置缺失时回退到默认列表"""
    try:
        from aggentic_RAG.travel_agent.config.settings import MCP_CONFIG_PATH

        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        names = [s.get("name") for s in config.get("mcp_servers", []) if s.get("name")]
        if names:
            return names
    except Exception:
        pass
    return ["12306 Server", "Gaode Server", "biying Server", "bazi Server", "flight Server"]


async def check_one(manager, server_name: str) -> dict:
    """检查单个服务器的状态；失败原因优先取自 get_health()"""
    health = manager.get_health()

    if server_name not in manager.mcp_servers:
        return {
            "server": server_name,
            "status": "failed",
            "reason": health.get("failed", {}).get(server_name, "未注册或未配置 URL"),
            "tools": [],
        }

    try:
        tools = await manager.list_tools(server_name)
    except Exception as e:
        return {
            "server": server_name,
            "status": "failed",
            "reason": f"{type(e).__name__}: {e}",
            "tools": [],
        }

    # list_tools() 返回 List[str]（工具名）；兼容未来可能改成对象的情况
    names = []
    for tool in tools or []:
        if isinstance(tool, str):
            names.append(tool)
        elif hasattr(tool, "name"):
            names.append(str(tool.name))
        elif isinstance(tool, dict) and "name" in tool:
            names.append(str(tool["name"]))
        else:
            names.append(str(tool))

    if not names:
        return {
            "server": server_name,
            "status": "degraded",
            "reason": "已连接但未返回工具列表",
            "tools": [],
        }

    return {"server": server_name, "status": "ok", "reason": "", "tools": names}


async def main() -> int:
    parser = argparse.ArgumentParser(description="MCP 服务器健康检查")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args()

    manager = await get_mcp_manager()
    servers = load_server_names()

    results = []
    for name in servers:
        results.append(await check_one(manager, name))

    ok = [r for r in results if r["status"] == "ok"]
    degraded = [r for r in results if r["status"] == "degraded"]
    failed = [r for r in results if r["status"] == "failed"]

    if args.json:
        print(
            json.dumps(
                {
                    "total": len(results),
                    "ok": len(ok),
                    "degraded": len(degraded),
                    "failed": len(failed),
                    "servers": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if not failed else 1

    print("=" * 60)
    print("🔍 MCP 服务器健康检查")
    print("=" * 60)
    print(f"配置服务器: {len(servers)} 个")
    print()

    for r in results:
        icon = {"ok": "✅", "degraded": "⚠️", "failed": "❌"}[r["status"]]
        print(f"{icon} {r['server']}")
        if r["status"] == "ok":
            preview = ", ".join(r["tools"][:5])
            more = "" if len(r["tools"]) <= 5 else f" …共 {len(r['tools'])} 个"
            print(f"   工具: {preview}{more}")
        else:
            print(f"   原因: {r['reason']}")
        print()

    print("=" * 60)
    print(f"✅ 正常: {len(ok)}  |  ⚠️ 降级: {len(degraded)}  |  ❌ 失败: {len(failed)}")
    print("=" * 60)

    if failed:
        print("\n排查建议：")
        print("1. 检查 aggentic_RAG/travel_agent/config/servers_config.json 的 URL 是否已填写")
        print("2. 确认对应 MCP Server 已启动、网络可达")
        print("3. 重新运行本脚本查看失败原因（异常类型 + 消息）")

    # 有失败则以非零退出码返回，便于 CI / 部署脚本判断
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
