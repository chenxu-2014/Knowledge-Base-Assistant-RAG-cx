"""
文档分块器抽象基类 —— RAG 离线阶段第二步。

职责：定义文本拆分的统一接口 (split) 和公共参数校验逻辑。

设计模式：策略模式 —— 不同的分块策略（递归、智能、固定长度、Token）
         都实现 BaseDocumentSplitter.split()，通过 DocumentSplitterFactory 切换。

参数说明：
    chunk_size     — 每个 chunk 的最大字符数。太大会导致 embedding 语义模糊，
                     太小会丢失上下文。推荐 300~800。
    chunk_overlap  — 相邻 chunk 的重叠字符数。保证跨 chunk 的上下文连续性，
                     推荐 chunk_size 的 10%~20%。
"""
import logging
from abc import ABC, abstractmethod

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseDocumentSplitter(ABC):
    """文档分块器抽象基类。

    所有分块策略（RecursiveSplitter、SmartSplitter 等）继承此类，
    复用参数校验逻辑，子类只需实现 split() 方法。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """初始化分块器。

        Args:
            chunk_size: 每个 chunk 的最大字符数。
            chunk_overlap: 相邻 chunk 之间的重叠字符数。
        """
        self._validate_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _validate_params(chunk_size: int, chunk_overlap: int) -> None:
        """校验分块参数的合法性。

        校验规则：
            - chunk_size 必须 > 0
            - chunk_overlap 必须 >= 0
            - chunk_overlap 必须 < chunk_size（否则切不出有意义的块）
        """
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
        """将文档切分为块。

        每个 chunk 是一个 Document，metadata 自动从原文档继承。
        子类实现具体的切分逻辑。

        Args:
            documents: 待切分的文档列表（通常是 DocumentLoader 的输出）。

        Returns:
            list[Document]: 切分后的 chunk 列表。
        """
        ...
