"""
向量数据库模块 —— RAG 离线阶段第四步：存入向量数据库。

职责：将嵌入后的 Document 写入 Chroma 向量库，并提供语义检索接口。

核心组件：
    VectorStoreFactory  — 工厂，从配置创建 VectorStoreManager
    VectorStoreManager  — Chroma 向量库管理器（增删查、版本管理、回滚）
    SearchResult        — 语义检索结果数据模型
    SourceInfo          — 来源文件信息
    VersionInfo         — 版本记录信息

调用关系：
    离线写入:
      pipeline.process(file_path) → chunks
      └─> vectorstore.upsert_documents(source, chunks)
           └─> Chroma.add_documents()

    在线检索:
      chain.invoke(question)
      └─> vectorstore.similarity_search(query, top_k)
           └─> Chroma.similarity_search_with_relevance_scores()
"""
from app.core.vectorstore.manager import VectorStoreManager
from app.core.vectorstore.models import SearchResult, SourceInfo, VersionInfo


class VectorStoreFactory:
    """从配置创建 VectorStoreManager。

    使用方式：
        vectorstore = VectorStoreFactory.from_settings(settings, embedding)
    """

    @staticmethod
    def from_settings(settings, embedding) -> VectorStoreManager:
        """根据配置和 Embedding 实例创建 VectorStoreManager。

        Args:
            settings: Settings 配置对象。
            embedding: Embedding 实例（EmbeddingFactory 的输出）。

        Returns:
            VectorStoreManager: 向量库管理器实例。
        """
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
