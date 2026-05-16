import logging

from app.core.reranker.base import BaseReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """创建重排序器实例"""

    @staticmethod
    def create(backend: str = "none", **kwargs) -> BaseReranker | None:
        if backend == "none":
            logger.info("未启用 reranker")
            return None

        if backend == "cross_encoder":
            from app.core.reranker.cross_encoder import CrossEncoderReranker

            return CrossEncoderReranker(**kwargs)

        raise ValueError(f"不支持的 reranker: {backend}，可选: none, cross_encoder")


__all__ = ["BaseReranker", "RerankerFactory"]
