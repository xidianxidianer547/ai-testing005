"""PDF 文档处理器核心模块。

支持接收 base64 编码的 PDF 数据，借助 PyMuPDF4LLM + LLMImageBlobParser
解析文本与图片，最终返回 PDF 的完整内容。
"""

import base64
import os
import re
import tempfile
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

load_dotenv()


def _clean_base64(data: str) -> str:
    """去除 base64 字符串中常见的 data URI 前缀。

    例如: data:application/pdf;base64,JVBERi0... -> JVBERi0...
    """
    if not data:
        return data
    cleaned = re.sub(r"^data:[^;]+;base64,", "", data.strip())
    return cleaned


class PDFProcessor:
    """PDF 处理器，支持 base64 输入与图文解析。

    使用 PyMuPDF4LLMLoader 解析 PDF 文本内容，
    使用 LLMImageBlobParser + ChatOpenAI 解析 PDF 中的图片内容。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 1024,
        mode: str = "page",
        extract_images: bool = True,
    ):
        """初始化 PDF 处理器。

        Args:
            model_name: OpenAI 模型名称，默认从环境变量 PDF_PROCESSOR_MODEL 读取，
                        否则使用 "gpt-5.4-mini"。
            max_tokens: LLM 最大 token 数，默认 1024。
            mode: PyMuPDF4LLMLoader 的解析模式，默认 "page"（按页解析）。
            extract_images: 是否提取并解析图片，默认 True。
        """
        self.model_name = model_name or os.getenv(
            "PDF_PROCESSOR_MODEL", "gpt-5.4-mini"
        )
        self.max_tokens = max_tokens
        self.mode = mode
        self.extract_images = extract_images

    def _build_loader(self, pdf_path: str) -> PyMuPDF4LLMLoader:
        """根据配置构建 PyMuPDF4LLMLoader 实例。"""
        return PyMuPDF4LLMLoader(
            pdf_path,
            mode=self.mode,
            extract_images=self.extract_images,
            images_parser=LLMImageBlobParser(
                model=ChatOpenAI(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                )
            ),
        )

    def process(self, base64_pdf: str) -> str:
        """处理 base64 编码的 PDF 数据，返回完整文本内容。

        Args:
            base64_pdf: base64 编码的 PDF 字符串（可带 data URI 前缀）。

        Returns:
            PDF 的完整文本内容（含图片 LLM 解析结果），各页之间以双换行分隔。

        Raises:
            ValueError: 当 base64 解码失败或 PDF 解析失败时抛出。
        """
        # 1. 清理 base64 数据
        raw_data = _clean_base64(base64_pdf)

        # 2. base64 解码为二进制
        try:
            pdf_bytes = base64.b64decode(raw_data)
        except Exception as e:
            raise ValueError(f"Base64 解码失败: {e}") from e

        # 3. 写入临时文件 -> 用 PyMuPDF4LLMLoader 解析 -> 拼接结果
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            loader = self._build_loader(tmp_path)
            docs = loader.load()

            if not docs:
                return ""

            # 按页拼接所有内容
            full_text = "\n\n".join(doc.page_content for doc in docs)
            return full_text

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"PDF 解析失败: {e}") from e
        finally:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def process_pdf_base64(
    base64_pdf: str,
    model_name: Optional[str] = None,
    max_tokens: int = 1024,
    mode: str = "page",
    extract_images: bool = True,
) -> str:
    """快捷函数：处理 base64 编码的 PDF 数据，返回完整文本内容。

    Args:
        base64_pdf: base64 编码的 PDF 字符串（可带 data URI 前缀）。
        model_name: OpenAI 模型名称，默认使用 "gpt-5.4-mini"。
        max_tokens: LLM 最大 token 数，默认 1024。
        mode: PyMuPDF4LLMLoader 的解析模式，默认 "page"。
        extract_images: 是否提取并解析图片，默认 True。

    Returns:
        PDF 的完整文本内容。
    """
    processor = PDFProcessor(
        model_name=model_name,
        max_tokens=max_tokens,
        mode=mode,
        extract_images=extract_images,
    )
    return processor.process(base64_pdf)
