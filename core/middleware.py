"""PDF 附件提取中间件。

从 HumanMessage 的 attachments 中提取 base64 编码的 PDF 数据，
调用 pdf_haha_processor 解析后，将 PDF 内容注入到 SystemMessage 中。
"""

import asyncio
import logging
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, SystemMessage

from pdf_haha_processor import PDFProcessor

logger = logging.getLogger(__name__)


class PDFAttachmentMiddleware(AgentMiddleware):
    """PDF 附件中间件：在模型调用前提取并解析 PDF，内容追加到 SystemMessage。"""

    def __init__(self, pdf_processor: PDFProcessor | None = None):
        super().__init__()
        self.pdf_processor = pdf_processor or PDFProcessor()

    def _extract_pdf_contents(self, messages: list) -> list[str]:
        """从消息列表中提取并解析所有 PDF 附件（同步）。"""
        pdf_contents: list[str] = []

        for msg in messages:
            if not isinstance(msg, HumanMessage):
                continue

            attachments = msg.additional_kwargs.get("attachments", [])
            for attachment in attachments:
                mime_type = attachment.get("mimeType", "")
                if mime_type != "application/pdf":
                    continue

                b64_data = attachment.get("data", "")
                if not b64_data:
                    continue

                filename = attachment.get("metadata", {}).get("filename", "unknown.pdf")
                logger.info(f"发现 PDF 附件: {filename}，开始解析")

                try:
                    pdf_text = self.pdf_processor.process(b64_data)
                    if pdf_text:
                        pdf_contents.append(f"--- PDF 文档: {filename} ---\n{pdf_text}")
                        logger.info(f"PDF 解析完成: {filename}，内容长度 {len(pdf_text)} 字符")
                    else:
                        logger.warning(f"PDF 解析结果为空: {filename}")
                except Exception as e:
                    logger.error(f"PDF 解析失败 ({filename}): {e}")

        return pdf_contents

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        pdf_contents = self._extract_pdf_contents(request.messages)

        if not pdf_contents:
            return handler(request)

        all_pdf_text = "\n\n".join(pdf_contents)
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": f"\n\n以下是用户上传的 PDF 文档内容，请基于此内容回答用户的问题：\n\n{all_pdf_text}"}
        ]
        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """异步版本：在线程池中执行 PDF 解析，避免阻塞事件循环。"""
        pdf_contents = await asyncio.to_thread(self._extract_pdf_contents, request.messages)

        if not pdf_contents:
            return await handler(request)

        all_pdf_text = "\n\n".join(pdf_contents)
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": f"\n\n以下是用户上传的 PDF 文档内容，请基于此内容回答用户的问题：\n\n{all_pdf_text}"}
        ]
        new_system_message = SystemMessage(content=new_content)
        return await handler(request.override(system_message=new_system_message))
