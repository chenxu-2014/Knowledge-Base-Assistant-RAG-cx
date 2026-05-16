import logging
from abc import ABC, abstractmethod

from app.core.vectorstore.models import SearchResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """重排序器抽象基类"""

    @abstractmethod
    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """
        对检索结果重排序，返回 top_k 个最相关结果。
        返回的 SearchResult.score 已被 reranker 的分数替换。
        """
        ...
