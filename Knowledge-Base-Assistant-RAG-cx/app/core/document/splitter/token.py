"""
Token 分块器 —— 按 token 数分块（预留，依赖 tiktoken）。

与字符级分块的区别：
    - FixedLengthSplitter / RecursiveSplitter: 按字符数切分，简单快速
    - TokenSplitter: 按 token 数切分，更精确地匹配 LLM 的 token 限制

使用场景：
    需要严格控制每个 chunk 的 token 数（如嵌入模型有 token 上限）。

依赖：
    pip install tiktoken
"""
import logging
from langchain_core.documents import Document

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)


class TokenSplitter(BaseDocumentSplitter):
    """按 token 数分块（预留，依赖 tiktoken）。"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, encoding_name: str = "cl100k_base"):
        """初始化 Token 分块器。

        Args:
            chunk_size: 每个 chunk 的最大 token 数。
            chunk_overlap: 相邻 chunk 的重叠 token 数。
            encoding_name: tiktoken 编码名称，默认 "cl100k_base"（GPT-4 使用）。
        """
        super().__init__(chunk_size, chunk_overlap)
        self.encoding_name = encoding_name

    def split(self, documents: list[Document]) -> list[Document]:
        """按 token 数切分文档（待实现）。"""
        raise NotImplementedError("TokenSplitter 尚未实现，请安装 tiktoken 后使用")
