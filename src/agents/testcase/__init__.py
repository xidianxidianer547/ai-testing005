"""
Testcase Agent Package

本包是一个基于 LangChain 和 LangGraph 构建的测试用例生成智能体，
用于自动分析需求文档并生成专业、全面的测试用例。

主要功能：
    - 通过 MCP (Model Context Protocol) 连接 Docling 文档解析服务
    - 解析各类需求文档（PDF、Word、Markdown 等）
    - 基于 LLM 生成结构化的测试用例

依赖说明：
    - langchain: LLM 应用开发框架
    - langchain-mcp-adapters: MCP 协议适配器
    - langchain-deepseek: DeepSeek 模型集成
    - docling-mcp: 文档解析 MCP 服务

使用示例：
    >>> from agents.testcase.agent import agent
    >>> result = agent.invoke({
    ...     "messages": [{"role": "user", "content": "请分析这份需求文档并生成测试用例"}]
    ... })

Author: AI Assistant
Date: 2026-03-14
"""
"""
版权所有 (c) 2023-2026 北京慧测信息技术有限公司(但问智能) 保留所有权利。

本代码版权归北京慧测信息技术有限公司(但问智能)所有，仅用于学习交流目的，未经公司商业授权，
不得用于任何商业用途，包括但不限于商业环境部署、售卖或以任何形式进行商业获利。违者必究。

授权商业应用请联系微信：huice666
"""

# fmt: off  MC8yOmFIVnBZMlhwb2I3bW1JN25rb2M2YTJKNGFBPT06ZmRmZDBiOGE=

# 包版本号
__version__ = "0.1.0"
# pragma: no cover  MS8yOmFIVnBZMlhwb2I3bW1JN25rb2M2YTJKNGFBPT06ZmRmZDBiOGE=

# 包级别的导出接口
# from .agent import agent
# from .tools import tools
