"""PDF 文档处理器：支持 base64 PDF 输入，返回完整解析内容。"""

from .processor import PDFProcessor, process_pdf_base64

__all__ = ["PDFProcessor", "process_pdf_base64"]
