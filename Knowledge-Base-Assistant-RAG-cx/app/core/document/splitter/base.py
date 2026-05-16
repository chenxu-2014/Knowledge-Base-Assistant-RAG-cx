import logging
from abc import ABC, abstractmethod

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseDocumentSplitter(ABC):
    """文档分块器抽象基类"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self._validate_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须大于 0，当前值: {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap 不能为负，当前值: {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )

    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        """
        将文档切分为块。
        metadata 自动从原文档继承到每个 chunk。
        """
        ...
