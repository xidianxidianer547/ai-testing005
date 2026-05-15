"""
测试用例智能体的工具模块

本模块负责初始化和管理测试用例生成智能体所需的各类工具，
主要通过 MCP (Model Context Protocol) 协议连接外部服务，
为智能体提供文档解析等扩展能力。

MCP 协议说明：
    MCP (Model Context Protocol) 是一种开放协议，用于标准化 AI 模型与
    外部工具/服务之间的通信。它允许 AI 助手通过统一的接口访问各种能力，
    如文件系统、数据库、API 等。

依赖服务：
    - docling-mcp: 文档解析服务，支持 PDF、Word、Markdown 等格式的文档解析
      启动命令: uvx --from docling-mcp docling-mcp-server --transport sse --port 8003
"""
"""
版权所有 (c) 2023-2026 北京慧测信息技术有限公司(但问智能) 保留所有权利。

本代码版权归北京慧测信息技术有限公司(但问智能)所有，仅用于学习交流目的，未经公司商业授权，
不得用于任何商业用途，包括但不限于商业环境部署、售卖或以任何形式进行商业获利。违者必究。

授权商业应用请联系微信：huice666
"""

# pylint: disable  MC8zOmFIVnBZMlhwb2I3bW1JN25rb2M2Um5BMmRBPT06MGJlYjZmODQ=

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


def create_mcp_client() -> MultiServerMCPClient:
    """
    创建并配置 MCP 客户端
    
    创建一个 MultiServerMCPClient 实例，配置连接到多个 MCP 服务器。
    当前配置连接 docling-server 用于文档解析。
    
    Returns:
        MultiServerMCPClient: 配置好的 MCP 客户端实例
        
    Note:
        使用 SSE (Server-Sent Events) 传输协议与 MCP 服务器通信，
        确保 docling-mcp 服务器已在指定端口启动。
    """
    # 配置 MCP 服务器连接信息
    # 键名为服务器标识，在代码中可通过该名称引用特定服务器
    server_configs = {
        # Docling 文档解析服务配置
        # 该服务提供文档解析能力，可将 PDF、Word 等文档转换为结构化文本
        "docling-server": {
            # SSE 服务端点 URL
            "url": "http://127.0.0.1:8000/sse",
            # 传输协议：sse (Server-Sent Events)
            "transport": "sse",
        }
    }
# pragma: no cover  MS8zOmFIVnBZMlhwb2I3bW1JN25rb2M2Um5BMmRBPT06MGJlYjZmODQ=
    
    return MultiServerMCPClient(server_configs)


# ============================================================================
# 全局工具初始化
# ============================================================================

# 创建 MCP 客户端实例
# 该实例用于管理与所有 MCP 服务器的连接
client = create_mcp_client()
# fmt: off  Mi8zOmFIVnBZMlhwb2I3bW1JN25rb2M2Um5BMmRBPT06MGJlYjZmODQ=

# 异步获取所有可用的 MCP 工具
# asyncio.run() 在同步上下文中运行异步代码
# client.get_tools() 返回所有已连接服务器提供的工具列表
tools = asyncio.run(client.get_tools())

# 打印工具加载信息，便于调试和确认服务连接状态
print(tools)
print(f"已加载 {len(tools)} 个工具: {[t.name for t in tools]}")
