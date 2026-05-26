"""
重排序器模块 —— RAG 检索阶段的可选增强组件。

职责：提供重排序器的工厂和统一接口。
      重排序器对粗召回的检索结果进行精排，提升最相关结果的排名。

支持的重排序器：
    "none"          — 不使用重排序（默认）
    "cross_encoder" — CrossEncoderReranker（基于 BAAI/bge-reranker-v2-m3）

调用关系：
    RAGChainFactory.create(reranker=RerankerFactory.create("cross_encoder"))
      └─> RerankerFactory.create("cross_encoder")
           └─> CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
"""
import logging

from app.core.reranker.base import BaseReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    """重排序器工厂 —— 创建重排序器实例。

    使用方式：
        reranker = RerankerFactory.create("cross_encoder")
        chain = RAGChainFactory.create(..., reranker=reranker)
    """

    @staticmethod
    def create(backend: str = "none", **kwargs) -> BaseReranker | None:
        """根据后端名称创建重排序器实例。

        Args:
            backend: 后端名称，可选 "none" / "cross_encoder"。
            **kwargs: 传递给 reranker 构造函数的参数（如 model_name）。

        Returns:
            BaseReranker | None: 重排序器实例，"none" 时返回 None。
        """
        if backend == "none":
            logger.info("未启用 reranker")
            return None

        if backend == "cross_encoder":
            from app.core.reranker.cross_encoder import CrossEncoderReranker

            return CrossEncoderReranker(**kwargs)

        raise ValueError(f"不支持的 reranker: {backend}，可选: none, cross_encoder")


__all__ = ["BaseReranker", "RerankerFactory"]
