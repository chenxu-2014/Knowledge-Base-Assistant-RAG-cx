import logging

from app.core.reranker.base import BaseReranker
from app.core.vectorstore.models import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 的重排序器（依赖 sentence-transformers）"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        logger.info("加载 CrossEncoder 模型: %s", model_name)
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)

        reranked = sorted(
            zip(results, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

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
