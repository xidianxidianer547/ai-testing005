from __future__ import annotations

"""
PDF 上下文注入中间件 (PDFContextMiddleware)

从 messages 中最后一个用户消息的 additional_kwargs.attachments 中提取 PDF。

消息格式示例：
    HumanMessage(
        content="解读一下",
        additional_kwargs={
            "attachments": [{
                "type": "file",
                "mimeType": "application/pdf",
                "data": "JVBERi0xLjc...",  # base64 编码
                "metadata": {"filename": "doc.pdf"}
            }]
        }
    )

设计说明（v3）：
    - thread_id 通过 langgraph.config.get_config() 从当前异步上下文获取，
      Runtime 对象本身不含 thread_id（官方文档明确说明）
    - 原始 SYSTEM_PROMPT 由构造函数直接传入（original_system_prompt 参数），
      完全不依赖运行时的 request.system_message（避免读到已污染的内容）
    - 中间件内部维护 thread_id → doc_text 的 per-session 状态字典
    - 每次请求扫描「全部历史消息」，找到本会话最后一次上传的 PDF 作为当前有效文档
    - 注入时通过 request.override(system_message=...) 创建新请求实例，
      原始全局 agent 对象完全不受影响（ModelRequest 官方不可变替换模式，
      底层用 dataclasses.replace，不涉及 pickle / deepcopy，彻底规避 _thread.RLock 问题）
    - 新会话无历史 PDF 时，system_message 自动还原为纯净的原始提示词
"""
"""
版权所有 (c) 2023-2026 北京慧测信息技术有限公司(但问智能) 保留所有权利。

本代码版权归北京慧测信息技术有限公司(但问智能)所有，仅用于学习交流目的，未经公司商业授权，
不得用于任何商业用途，包括但不限于商业环境部署、售卖或以任何形式进行商业获利。违者必究。

授权商业应用请联系微信：huice666
"""


import base64
import hashlib
import logging
from typing import Any, Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ResponseT
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.typing import ContextT
# pragma: no cover  MC80OmFIVnBZMlhwb2I3bW1JN25rb2M2UVdwd1ZRPT06NmIwNmVlOTQ=

from processors.pdf import PDFProcessor

logger = logging.getLogger(__name__)

_DOCUMENT_TEMPLATE = """\
以下是用户上传的参考文档，请在回答时充分参考其内容：

<document>
{content}
</document>
"""


def _decode_base64(data: str) -> bytes:
    """将 base64 字符串解码为 bytes。"""
    if "," in data:
        data = data.split(",", 1)[1]
    return base64.b64decode(data)

# pylint: disable  MS80OmFIVnBZMlhwb2I3bW1JN25rb2M2UVdwd1ZRPT06NmIwNmVlOTQ=

class PDFContextMiddleware(AgentMiddleware):
    """PDF 文档上下文注入中间件（v3：构造函数传入原始提示词 + get_config 获取 thread_id）。

    核心改进：
    1. original_system_prompt 由构造函数直接传入，完全不依赖运行时 request.system_message，
       彻底规避因服务器未重启导致快照读到污染内容的问题。
    2. thread_id 通过官方 langgraph.config.get_config() 从当前异步上下文获取，
       Runtime 对象本身不含 thread_id。
    3. _session_docs 以 thread_id 为键，维护 per-session 文档状态，新会话天然隔离。
    4. 扫描全部历史消息找最新 PDF，同会话换文件时直接替换（不追加）。
    5. 使用 request.override() 官方不可变替换，不涉及 pickle/deepcopy，
       彻底规避 _thread.RLock 不可序列化问题。
    """

    def __init__(
        self,
        original_system_prompt: str | list | None = None,
        enable_cache: bool = True,
        max_content_length: int = 80_000,
    ):
        """
        Args:
            original_system_prompt: 智能体的原始系统提示词，直接传入 SYSTEM_PROMPT 常量。
                注入时始终以此为基底，确保内容干净，不受运行时污染影响。
                若为 None，则首次请求时从 request.system_message 读取（兼容旧用法）。
            enable_cache: 是否启用 PDF 解析缓存（相同文件不重复解析）。
            max_content_length: PDF 文档内容的最大字符数，超出时截断。
        """
        self._processor = PDFProcessor(enable_cache=enable_cache)
        self._max_content_length = max_content_length
        # 原始系统提示词（只读，永不被修改）
        self._original_system_content: str | list | None = original_system_prompt
        # per-session 文档状态：thread_id -> doc_text（替换语义）
        self._session_docs: dict[str, str] = {}
        # per-session 已解析 PDF 的 hash：thread_id -> pdf_md5
        # 用于判断"当前消息是否携带了和上次不同的新 PDF"
        # 相同 hash → 直接复用 _session_docs，跳过解析
        # 不同 hash → 新文件，重新解析并覆盖
        # 不存在   → 从未上传过，不触发解析
        self._session_pdf_hash: dict[str, str] = {}

    @staticmethod
    def _clean_file_blocks_from_messages(messages: list) -> list:
        """清理 messages 中 content 包含的 type='file' block。

        前端上传 PDF 时，除了 additional_kwargs.attachments 外，
        有时还会在 content 列表里插入 ``{"type": "file", ...}`` block。
        DeepSeek / OpenAI API 只支持 ``text`` 和 ``image_url`` 两种 type，
        遇到 ``file`` 会直接报 400。因此必须在发给 LLM 前过滤掉。
        """
        cleaned = []
        changed = False
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                new_blocks = [
                    block
                    for block in content
                    if not (isinstance(block, dict) and block.get("type") == "file")
                ]
                if len(new_blocks) != len(content):
                    changed = True
                    # 尽量保留消息的附加属性（id、additional_kwargs 等）
                    if hasattr(msg, "model_copy"):
                        cleaned.append(msg.model_copy(update={"content": new_blocks}))
                    elif hasattr(msg, "copy"):
                        cleaned.append(msg.copy(update={"content": new_blocks}))
                    else:
                        cleaned.append(msg.__class__(content=new_blocks))
                    continue
            cleaned.append(msg)
        return cleaned if changed else messages

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> Any:
        """拦截 LLM 调用，按会话注入 PDF 文档上下文后再转发。"""

        # ── 第一步：兜底快照（original_system_prompt 未传入时的兼容逻辑）──
        # 注意：此路径仅在未传 original_system_prompt 时触发，且只读取一次。
        # 如已传入，_original_system_content 在 __init__ 时即已确定，此处直接跳过。
        if self._original_system_content is None and request.system_message is not None:
            self._original_system_content = request.system_message.content
            logger.warning(
                "[PDFContextMiddleware] original_system_prompt 未传入，"
                "已从首次 request.system_message 快照，建议通过构造函数显式传入以确保安全。"
            )

        # ── 第二步：从 LangGraph 异步上下文获取 thread_id ──
        thread_id = self._get_thread_id()

        # ── 第三步：只处理「最后一条」用户消息中的 PDF 附件 ──
        # 关键逻辑：_extract_latest_pdf_from_history 扫描全部历史消息，
        # 但我们只在「当前请求的最新消息」中有新 PDF 时才触发解析。
        # 通过 MD5 hash 比对来判断是否是新文件：
        #   - 从未上传  → _session_pdf_hash 无此 key，跳过解析，直接用已有 doc（若有）
        #   - 相同 hash → 同一份文件，跳过解析，复用 _session_docs 中内容
        #   - 不同 hash → 新文件，重新解析并覆盖
        pdf_info = self._extract_pdf_from_last_message(request)
        if pdf_info is not None:
            pdf_data, pdf_name = pdf_info
            pdf_hash = hashlib.md5(pdf_data).hexdigest()
            if self._session_pdf_hash.get(thread_id) == pdf_hash:
                # 同一份文件，已经解析过，直接跳过
                logger.debug(
                    "[PDFContextMiddleware] 会话 %s PDF 未变化（hash=%s），跳过重复解析",
                    thread_id, pdf_hash,
                )
            else:
                # 新文件（或首次上传），触发解析
                logger.info("[PDFContextMiddleware] 检测到新 PDF: %s，开始提取文本…", pdf_name)
                text = self._processor.extract_text(pdf_data, pdf_name)
                if text:
                    self._session_docs[thread_id] = text        # 替换，不追加
                    self._session_pdf_hash[thread_id] = pdf_hash  # 记录 hash
                    logger.info(
                        "[PDFContextMiddleware] 会话 %s 文档已更新: %s，长度: %d 字符",
                        thread_id, pdf_name, len(text),
                    )

        # ── 第四步：根据会话文档状态决定是否注入 ──
        # 决策矩阵：
        #   有文档(_session_docs 命中)            → override，注入文档到 system_message
        #   无文档 + 从未上传(_session_docs 无key) → 直接透传，request 本身已是干净状态
        current_doc = self._session_docs.get(thread_id)
        if current_doc:
            request = request.override(system_message=self._build_system_message(current_doc))
            logger.info("[PDFContextMiddleware] 会话 %s system_message 已用最新文档重建", thread_id)
        else:
            # 本会话从未上传过文档，直接透传，无需任何 override 操作
            logger.debug("[PDFContextMiddleware] 会话 %s 无文档记录，透传原始 request", thread_id)

        # ── 第五步：清理 messages 中 LLM API 不支持的 file block ──
        cleaned_messages = self._clean_file_blocks_from_messages(request.messages)
        if cleaned_messages is not request.messages:
            request = request.override(messages=cleaned_messages)
            logger.debug("[PDFContextMiddleware] 已清理 messages 中的 file block")

        return await handler(request)

    # ──────────────────────────────────────────────
    # 内部辅助方法
    # ──────────────────────────────────────────────

    def _get_thread_id(self) -> str:
        """从 LangGraph 当前异步上下文获取 thread_id，用于区分不同会话。

        官方说明：Runtime 对象不含 config，thread_id 须通过
        langgraph.config.get_config() 从 contextvars 中读取。
        路径: get_config()["configurable"]["thread_id"]
        若取不到则回退到 "__default__"（单用户本地调试场景）。
        """
        try:
            from langgraph.config import get_config
            config = get_config()
            tid = (
                config.get("metadata").get("thread_id")
                or config.get("configurable", {}).get("thread_id")
            )
            if tid:
                return str(tid)
        except Exception:
            pass
        return "__default__"

    def _extract_pdf_from_last_message(self, request: ModelRequest) -> tuple[bytes, str] | None:
        """只从「最后一条用户消息」中提取 PDF 附件。

        与旧的 _extract_latest_pdf_from_history 不同：
        - 旧方法：扫描全部历史消息 → 每轮对话都会找到历史中的 PDF → 每次都触发解析
        - 新方法：只看最后一条消息 → 只有用户本次上传了 PDF 才返回数据
        结合外层的 hash 比对，实现"首次上传才解析，后续对话完全跳过"的正确语义。
        """
        if not request.messages:
            return None

        # 只取最后一条消息
        last_msg = request.messages[-1]
        if not isinstance(last_msg, HumanMessage):
            return None

        attachments = last_msg.additional_kwargs.get("attachments", [])
        if not isinstance(attachments, list):
            return None
# fmt: off  Mi80OmFIVnBZMlhwb2I3bW1JN25rb2M2UVdwd1ZRPT06NmIwNmVlOTQ=

        for att in attachments:
            if not isinstance(att, dict):
                continue
            if att.get("mimeType", "").lower() != "application/pdf":
                continue

            data = att.get("data")
            if not data or not isinstance(data, str):
                continue

            try:
                pdf_bytes = _decode_base64(data)
                filename = att.get("metadata", {}).get("filename", "document.pdf")
                return pdf_bytes, filename
            except Exception as e:
                logger.warning("[PDFContextMiddleware] PDF 解码失败: %s", e)
                continue

        return None
# noqa  My80OmFIVnBZMlhwb2I3bW1JN25rb2M2UVdwd1ZRPT06NmIwNmVlOTQ=

    def _build_system_message(self, doc_text: str) -> SystemMessage:
        """以原始 system_message.content 为基底，拼接文档块，返回全新的 SystemMessage。

        始终从 _original_system_content 出发（替换语义，非追加），
        彻底避免多次上传时内容叠加，且不触碰任何现有 request 对象。
        """
        if len(doc_text) > self._max_content_length:
            doc_text = doc_text[: self._max_content_length] + "\n\n[文档内容已截断...]"

        doc_block_text = _DOCUMENT_TEMPLATE.format(content=doc_text)
        base_content = self._original_system_content

        if isinstance(base_content, str):
            new_content: str | list = base_content + "\n\n" + doc_block_text
        elif isinstance(base_content, list):
            new_content = list(base_content) + [{"type": "text", "text": doc_block_text}]
        else:
            new_content = doc_block_text

        return SystemMessage(content=new_content)

    def clear_session(self, thread_id: str) -> None:
        """主动清除指定会话的文档状态（可供外部调用，例如用户点击「清除上下文」）。"""
        removed = self._session_docs.pop(thread_id, None)
        self._session_pdf_hash.pop(thread_id, None)
        if removed is not None:
            logger.info("[PDFContextMiddleware] 会话 %s 的文档状态已清除", thread_id)

    def get_session_stats(self) -> dict:
        """获取当前所有活跃会话的文档状态统计（调试用）。"""
        return {
            "active_sessions": len(self._session_docs),
            "session_ids": list(self._session_docs.keys()),
            "doc_lengths": {tid: len(text) for tid, text in self._session_docs.items()},
        }
