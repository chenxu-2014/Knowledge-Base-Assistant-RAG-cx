import logging

from langchain_core.language_models import BaseChatModel

from app.core.chain.rag_chain import RAGChain, RAGResult, SourceDocument
from app.core.chain.prompts import DEFAULT_SYSTEM_PROMPT
from app.core.reranker.base import BaseReranker
from app.core.vectorstore.manager import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGChainFactory:
    """从配置创建 RAGChain"""

    @staticmethod
    def create(
        llm: BaseChatModel,
        vectorstore: VectorStoreManager,
        top_k: int = 4,
        retrieve_k: int | None = None,
        score_threshold: float | None = None,
        reranker: BaseReranker | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> RAGChain:
        logger.info(
            "创建 RAGChain: top_k=%d, retrieve_k=%s, reranker=%s",
            top_k,
            retrieve_k,
            type(reranker).__name__ if reranker else "None",
        )
        return RAGChain(
            llm=llm,
            vectorstore=vectorstore,
            top_k=top_k,
            retrieve_k=retrieve_k,
            score_threshold=score_threshold,
            reranker=reranker,
            system_prompt=system_prompt,
        )


__all__ = [
    "RAGChain",
    "RAGChainFactory",
    "RAGResult",
    "SourceDocument",
]
