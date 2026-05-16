import logging

from langchain_core.embeddings import Embeddings

from app.core.vectorstore.manager import VectorStoreManager
from app.core.vectorstore.models import SearchResult, SourceInfo, VersionInfo

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """从配置创建 VectorStoreManager"""

    @staticmethod
    def from_settings(settings, embedding: Embeddings) -> VectorStoreManager:
        logger.info(
            "创建 VectorStoreManager: collection=%s, persist=%s",
            settings.chroma_collection_name,
            settings.chroma_persist_dir,
        )
        return VectorStoreManager(
            embedding=embedding,
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )


__all__ = [
    "VectorStoreManager",
    "VectorStoreFactory",
    "SearchResult",
    "SourceInfo",
    "VersionInfo",
]
