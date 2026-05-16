import logging

from langchain_core.documents import Document

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)


class TokenSplitter(BaseDocumentSplitter):
    """按 token 数分块（预留，依赖 tiktoken）"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, encoding_name: str = "cl100k_base"):
        super().__init__(chunk_size, chunk_overlap)
        self.encoding_name = encoding_name

    def split(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError("TokenSplitter 尚未实现，请安装 tiktoken 后使用")
