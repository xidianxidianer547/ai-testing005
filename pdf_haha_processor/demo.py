"""PDF 处理器使用示例。

运行前请确保：
1. 环境变量 OPENAI_API_KEY 已设置（或在 .env 文件中配置）。
2. 有可用的 base64 编码 PDF 数据。

用法示例：
    python -m pdf_haha_processor.demo
"""

import base64
import sys
from pathlib import Path

# 将项目根目录加入路径，确保可以导入 pdf_haha_processor
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pdf_haha_processor import process_pdf_base64


def main():
    # 示例：读取本地 PDF 并转为 base64，然后调用处理器
    # 你可以把这里换成前端传入的 base64 字符串
    sample_pdf_path = Path("example_data/layout-parser-paper.pdf")

    if not sample_pdf_path.exists():
        print(f"示例文件不存在: {sample_pdf_path}")
        print("请准备一个 PDF 文件，或直接将 base64 字符串传入 process_pdf_base64()")
        return

    base64_pdf = base64.b64encode(sample_pdf_path.read_bytes()).decode("utf-8")
    print("开始解析 PDF ...")

    try:
        result = process_pdf_base64(base64_pdf)
        print("\n=== PDF 解析结果 ===\n")
        print(result)
    except Exception as e:
        print(f"解析失败: {e}")


if __name__ == "__main__":
    main()
