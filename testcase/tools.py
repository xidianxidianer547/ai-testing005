import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

def get_weather(city: str):
    """我能获取天气预报"""
    print("今天是个好天气")
    return f"{city},晴天"

# MCP 客户端 - 延迟加载
_client = None
_docling_tools = []

async def _init_mcp_client():
    global _client, _docling_tools
    if _client is None:
        _client = MultiServerMCPClient(
            {
                "docling-server": {
                    "url": "http://localhost:8000/sse",
                    "transport": "sse"
                }
            }
        )
        _docling_tools = await _client.get_tools()
    return _docling_tools

def get_docling_tools():
    """获取 docling 工具列表"""
    try:
        return asyncio.run(_init_mcp_client())
    except Exception as e:
        print(f"[警告] MCP 客户端连接失败: {e}")
        return []

# 导出所有工具
__all__ = ["get_weather", "get_docling_tools"]