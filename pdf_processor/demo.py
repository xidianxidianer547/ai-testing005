"""PDF 处理器使用示例。

运行前请确保：
1. 环境变量 OPENAI_API_KEY / OPENAI_BASE_URL 已设置（或在 .env 文件中配置）。
2. 有可用的 PDF 文件或 base64 编码 PDF 数据。

用法：
    python -m pdf_processor.demo
    python -m pdf_processor.demo path/to/your.pdf
"""

import base64
import sys
from pathlib import Path

from pdf_processor import process_pdf_base64


def main():
    """主入口：读取 PDF 文件 -> base64 编码 -> 调用处理器 -> 输出结果。"""

    # 支持通过命令行参数传入 PDF 文件路径
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = Path("example_data/layout-parser-paper.pdf")

    if not pdf_path.exists():
        print(f"[ERROR] PDF 文件不存在: {pdf_path}")
        print("用法: python -m pdf_processor.demo <pdf文件路径>")
        print("或者直接在代码中调用 process_pdf_base64(base64_str)")
        return

    # 读取 PDF 并转为 base64
    print(f"正在读取文件: {pdf_path}")
    base64_pdf = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    print(f"文件大小: {pdf_path.stat().st_size / 1024:.1f} KB")
    print("开始解析 PDF ...\n")

    try:
        result = process_pdf_base64(base64_pdf)
        print("=" * 60)
        print("  PDF 解析结果")
        print("=" * 60)
        print(result)
        print("=" * 60)
        print(f"解析完成，共 {len(result)} 个字符。")
    except Exception as e:
        print(f"[ERROR] 解析失败: {e}")


if __name__ == "__main__":
    main()
