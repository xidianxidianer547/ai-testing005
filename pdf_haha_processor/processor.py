"""PDF 文档处理器核心模块。

支持接收 base64 编码的 PDF 数据，借助 PyMuPDF4LLM + LLMImageBlobParser
解析文本与图片，最终返回 PDF 的完整内容。

特性：
- 基于内容 SHA256 哈希的缓存机制，避免同一文件重复解析。
- 支持内存缓存（默认）与本地文件缓存（持久化）。
- mode="single" 单文档模式，支持表格提取与多模态图片解析。
"""

import base64
import hashlib
import json
import logging
import os
import pickle
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 缓存抽象与实现
# ---------------------------------------------------------------------------

class PDFCache(ABC):
    """PDF 解析结果缓存接口。"""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """根据 key 获取缓存内容，不存在时返回 None。"""
        raise NotImplementedError

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """将解析结果写入缓存。"""
        raise NotImplementedError


class MemoryPDFCache(PDFCache):
    """内存缓存，仅在当前进程内有效，重启后失效。"""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        self._cache[key] = value


class FilePDFCache(PDFCache):
    """本地文件缓存，解析结果持久化到磁盘，重启后仍然有效。"""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._index_file = self.cache_dir / "index.json"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        if self._index_file.exists():
            try:
                return json.loads(self._index_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        self._index_file.write_text(
            json.dumps(self._index, ensure_ascii=False),
            encoding="utf-8",
        )

    def get(self, key: str) -> Optional[str]:
        if key not in self._index:
            return None
        cache_file = Path(self._index[key])
        if not cache_file.exists():
            del self._index[key]
            self._save_index()
            return None
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception:
            return None

    def set(self, key: str, value: str) -> None:
        cache_file = self.cache_dir / f"{key}.pkl"
        cache_file.write_bytes(pickle.dumps(value))
        self._index[key] = str(cache_file)
        self._save_index()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _clean_base64(data: str) -> str:
    """去除 base64 字符串中常见的 data URI 前缀。"""
    if not data:
        return data
    cleaned = re.sub(r"^data:[^;]+;base64,", "", data.strip())
    return cleaned


# ---------------------------------------------------------------------------
# PDF 处理器
# ---------------------------------------------------------------------------

class PDFProcessor:
    """PDF 处理器，支持 base64 输入、图文解析与结果缓存。"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 1024,
        mode: str = "single",
        extract_images: bool = True,
        table_strategy: str = "lines",
        enable_multimodal: bool = True,
        cache: Optional[PDFCache] = None,
    ):
        """
        初始化 PDF 处理器。

        Args:
            model_name: OpenAI 模型名称，默认从环境变量 PDF_PROCESSOR_MODEL 读取，
                        否则使用 "gpt-5.4-mini"。
            max_tokens: LLM 最大 token 数，默认 1024。
            mode: PyMuPDF4LLMLoader 的解析模式，默认 "single"。
            extract_images: 是否提取图片，默认 True。
            table_strategy: 表格提取策略，默认 "lines"。
            enable_multimodal: 是否启用多模态图片解析，默认 True。
            cache: 缓存实例，默认使用 MemoryPDFCache（内存缓存）。
                   可传入 FilePDFCache 实现持久化缓存。
        """
        self.model_name = model_name or os.getenv("PDF_PROCESSOR_MODEL", "gpt-5.4-mini")
        self.max_tokens = max_tokens
        self.mode = mode
        self.extract_images = extract_images
        self.table_strategy = table_strategy
        self.enable_multimodal = enable_multimodal
        self.cache = cache or MemoryPDFCache()

    def process(self, base64_pdf: str) -> str:
        """
        处理 base64 编码的 PDF 数据，返回完整文本内容。

        首次解析后会将结果缓存；后续传入相同内容会直接返回缓存结果，
        不再重复调用 LLM 解析。

        Args:
            base64_pdf: base64 编码的 PDF 字符串（可带 data URI 前缀）。

        Returns:
            PDF 的完整文本内容（含图片解析结果）。

        Raises:
            ValueError: 当 base64 解码失败或 PDF 解析失败时抛出。
        """
        raw_data = _clean_base64(base64_pdf)
        content_hash = hashlib.sha256(raw_data.encode("utf-8")).hexdigest()

        # 1. 检查缓存
        cached = self.cache.get(content_hash)
        if cached is not None:
            logger.info("PDF 解析结果命中缓存，直接返回")
            return cached

        # 2. 解码 base64
        try:
            pdf_bytes = base64.b64decode(raw_data)
        except Exception as e:
            logger.error(f"Base64 解码失败: {e}")
            raise ValueError(f"Base64 解码失败: {e}") from e

        logger.info(f"开始解析 PDF，大小: {len(pdf_bytes)} bytes，多模态: {self.enable_multimodal}")

        # 3. 写入临时 PDF 文件并解析
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            loader_kwargs = {
                "file_path": tmp_path,
                "mode": self.mode,
                "table_strategy": self.table_strategy,
            }

            if self.enable_multimodal and self.extract_images:
                image_parser = LLMImageBlobParser(
                    model=ChatOpenAI(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                    )
                )
                loader_kwargs["extract_images"] = True
                loader_kwargs["images_parser"] = image_parser
            else:
                loader_kwargs["extract_images"] = False

            loader = PyMuPDF4LLMLoader(**loader_kwargs)
            docs = loader.load()

            if docs:
                full_text = "\n\n".join(doc.page_content for doc in docs)
                logger.info(f"PDF 解析成功，内容长度: {len(full_text)} 字符")
            else:
                full_text = ""
                logger.warning("PDF 解析后内容为空")
        except Exception as e:
            logger.error(f"PDF 解析失败: {e}")
            raise ValueError(f"PDF 解析失败: {e}") from e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.debug(f"临时文件已删除: {tmp_path}")

        # 4. 写入缓存
        self.cache.set(content_hash, full_text)
        logger.info("PDF 解析结果已缓存")
        return full_text

    def clear_cache(self) -> None:
        """清空缓存（仅对 MemoryPDFCache 有效，FilePDFCache 需单独处理）。"""
        if isinstance(self.cache, MemoryPDFCache):
            self.cache._cache.clear()
            logger.info("内存缓存已清空")

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息。"""
        if isinstance(self.cache, MemoryPDFCache):
            return {
                "cache_type": "memory",
                "cached_count": len(self.cache._cache),
            }
        elif isinstance(self.cache, FilePDFCache):
            return {
                "cache_type": "file",
                "cache_dir": str(self.cache.cache_dir),
                "cached_count": len(self.cache._index),
            }
        return {"cache_type": "unknown", "cached_count": 0}


def process_pdf_base64(
    base64_pdf: str,
    model_name: Optional[str] = None,
    max_tokens: int = 1024,
    mode: str = "single",
    extract_images: bool = True,
    table_strategy: str = "lines",
    enable_multimodal: bool = True,
    cache: Optional[PDFCache] = None,
) -> str:
    """
    快捷函数：处理 base64 编码的 PDF 数据，返回完整文本内容。

    Args:
        base64_pdf: base64 编码的 PDF 字符串（可带 data URI 前缀）。
        model_name: OpenAI 模型名称，默认使用 "gpt-5.4-mini"。
        max_tokens: LLM 最大 token 数，默认 1024。
        mode: PyMuPDF4LLMLoader 的解析模式，默认 "single"。
        extract_images: 是否提取图片，默认 True。
        table_strategy: 表格提取策略，默认 "lines"。
        enable_multimodal: 是否启用多模态图片解析，默认 True。
        cache: 缓存实例，默认使用 MemoryPDFCache。

    Returns:
        PDF 的完整文本内容。
    """
    processor = PDFProcessor(
        model_name=model_name,
        max_tokens=max_tokens,
        mode=mode,
        extract_images=extract_images,
        table_strategy=table_strategy,
        enable_multimodal=enable_multimodal,
        cache=cache,
    )
    return processor.process(base64_pdf)
