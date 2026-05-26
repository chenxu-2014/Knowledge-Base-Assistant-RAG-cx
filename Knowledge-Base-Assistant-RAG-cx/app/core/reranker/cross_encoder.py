"""
Cross-Encoder 重排序器 —— 基于 BAAI/bge-reranker-v2-m3 的精排。

与 Embedding 检索（Bi-Encoder）的区别：
    - Bi-Encoder: query 和 doc 分别编码，计算余弦相似度（快，但精度略低）
    - Cross-Encoder: query 和 doc 拼接后一起编码，输出相关性分数（慢，但精度更高）

使用方式：
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    top_results = reranker.rerank(query, candidates, top_k=4)

依赖：
    pip install sentence-transformers

在 RAG 流程中的位置：
    RAGChain._retrieve()
      └─> self._reranker.rerank(question, candidates, top_k)
           └─> CrossEncoderReranker.rerank()
"""
import logging

from app.core.reranker.base import BaseReranker
from app.core.vectorstore.models import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 的重排序器。

    使用 sentence-transformers 的 CrossEncoder 类加载预训练模型，
    对 (query, document) 对进行相关性打分。

    Attributes:
        _model: sentence_transformers.CrossEncoder 实例。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """初始化 CrossEncoder 模型。

        Args:
            model_name: HuggingFace 模型名称。默认 "BAAI/bge-reranker-v2-m3"，
                       针对中文优化的重排序模型。

        注意：首次使用需下载模型（约 2GB），可能较慢。
        """
        logger.info("加载 CrossEncoder 模型: %s", model_name)
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """对检索结果重排序。

        流程：
            1. 构建 (query, document) 对
            2. CrossEncoder 预测相关性分数
            3. 按分数降序排列，取 top_k 个
            4. 用 reranker 的分数替换原始 score

        Args:
            query: 用户查询文本。
            results: 粗召回的检索结果列表。
            top_k: 返回的最相关结果数量。

        Returns:
            list[SearchResult]: 重排序后的结果列表。
        """
        if not results:
            return []

        # 构建 (query, document) 对，供 CrossEncoder 打分
        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)

        # 按分数降序排列，取 top_k
        reranked = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        # 用 reranker 的分数替换原始相似度分数
        output = []
        for result, score in reranked:
            new_result = SearchResult(
                content=result.content,
                metadata=result.metadata,
                score=float(score),
            )
            output.append(new_result)

        logger.info(
            "CrossEncoder rerank: %d → %d, top_score=%.4f",
            len(results),
            len(output),
            output[0].score if output else 0,
        )
        return output
