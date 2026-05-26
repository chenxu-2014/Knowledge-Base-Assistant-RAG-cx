"""
重排序器抽象基类 —— RAG 检索阶段的可选增强组件。

职责：定义重排序的统一接口。重排序器对粗召回的检索结果进行精排，
      提升最相关结果的排名，过滤噪声。

在 RAG 流程中的位置（_retrieve 方法）：
    1. 粗召回: vectorstore.similarity_search() → retrieve_k 个候选
    2. 阈值过滤: score_threshold 过滤
    3. 精排: reranker.rerank() → top_k 个最终结果  ← 此步骤
    4. 拼接上下文 → 调用 LLM

调用关系：
    RAGChain._retrieve()
      └─> self._reranker.rerank(question, candidates, top_k)
           └─> CrossEncoderReranker.rerank()
"""
import logging
from abc import ABC, abstractmethod

from app.core.vectorstore.models import SearchResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """重排序器抽象基类。

    子类实现 rerank() 方法，对检索结果进行精排。
    当前实现：CrossEncoderReranker（基于 sentence-transformers CrossEncoder）。
    """

    @abstractmethod
    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """对检索结果重排序，返回 top_k 个最相关结果。

        Args:
            query: 用户查询文本。
            results: 粗召回的检索结果列表。
            top_k: 返回的最相关结果数量。

        Returns:
            list[SearchResult]: 重排序后的结果列表（最多 top_k 个），
                               score 已被 reranker 的分数替换。
        """
        ...
