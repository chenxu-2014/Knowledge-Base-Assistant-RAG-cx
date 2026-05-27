"""
向量数据库模块 —— RAG 离线阶段第四步：存入向量数据库。

职责：将嵌入后的 Document 写入向量库（Chroma 或 Milvus），并提供语义检索接口。

核心组件：
    VectorStoreFactory       — 工厂，从配置创建 Chroma 或 Milvus 管理器
    VectorStoreManager       — Chroma 向量库管理器（增删查、版本管理、回滚）
    MilvusVectorStoreManager — Milvus 向量库管理器（支持 Dense + Sparse 混合检索）
    SearchResult             — 语义检索结果数据模型
    SourceInfo               — 来源文件信息
    VersionInfo              — 版本记录信息

调用关系：
    离线写入:
      pipeline.process(file_path) → chunks
      └─> vectorstore.upsert_documents(source, chunks)
           └─> Chroma.add_documents() / Milvus.add_documents()

    在线检索:
      chain.invoke(question)
      └─> vectorstore.similarity_search(query, top_k)
           └─> Chroma / Milvus hybrid search
"""
from app.core.vectorstore.manager import VectorStoreManager
from app.core.vectorstore.models import SearchResult, SourceInfo, VersionInfo


class VectorStoreFactory:
    """从配置创建向量库管理器（Chroma 或 Milvus）。

    使用方式：
        vectorstore = VectorStoreFactory.from_settings(settings, embedding)
    """

    @staticmethod
    def from_settings(settings, embedding):
        """根据配置和 Embedding 实例创建向量库管理器。

        通过 settings.vectorstore_provider 决定使用 Chroma 还是 Milvus：
            - "chroma"（默认）: 使用 VectorStoreManager（Chroma 本地持久化）
            - "milvus": 使用 MilvusVectorStoreManager（Milvus 混合检索）

        Args:
            settings: Settings 配置对象。
            embedding: Embedding 实例（EmbeddingFactory 的输出）。

        Returns:
            VectorStoreManager | MilvusVectorStoreManager: 向量库管理器实例。
        """
        provider = getattr(settings, "vectorstore_provider", "chroma")

        if provider == "milvus":
            from app.core.vectorstore.milvus_manager import MilvusVectorStoreManager

            return MilvusVectorStoreManager(
                embedding=embedding,
                uri=settings.milvus_uri,
                token=settings.milvus_token,
                collection_name=settings.milvus_collection_name,
                dense_weight=settings.milvus_dense_weight,
                sparse_weight=settings.milvus_sparse_weight,
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
