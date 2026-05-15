from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime

from core.middleware import PDFAttachmentMiddleware
from pdf_haha_processor import PDFProcessor

load_dotenv()
# llm = ChatDeepSeek(model=”deepseek-chat”)
llm = init_chat_model("deepseek:deepseek-chat")


def get_weather1(city: str) -> str:
    '''
    获取指定城市的天气信息

    这是一个示例工具函数，演示如何为智能体定义可调用的工具。
    在实际应用中，该函数可以调用真实的天气API。

    Args:
        city(str):城市名称，如”北京”、”shanghai”

    returns:
        str:该城市的天气情况描述

    :param city:
    :return:
    '''
    return f"{city},晴天"

# 智能体里面，在向大模型发送信息之前执行这个方法。agent里面加上middleware
@before_model()
def check_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(state)
    return None

# 创建 PDF 处理器与中间件实例
pdf_processor = PDFProcessor()
pdf_middleware = PDFAttachmentMiddleware(pdf_processor=pdf_processor)

agent = create_agent(
    model=llm, #使用deepseek模型
    tools=[get_weather1], #注册天气查询工具
    middleware=[check_message, pdf_middleware], # 大模型发送消息之前，会执行这些中间件
    system_prompt="You are a helpful assistant"
)
