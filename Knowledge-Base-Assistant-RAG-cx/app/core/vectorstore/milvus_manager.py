"""
Milvus 向量库管理器 —— 支持 Dense + Sparse BM25 混合检索。

职责：
    1. 文档写入（自动生成 dense 向量 + BM25 稀疏向量）
    2. 混合检索（WeightedRanker 融合 dense + sparse）
    3. 版本管理（同 Chroma 版本机制）

与 VectorStoreManager（Chroma 版）的区别：
    - 底层使用 Milvus 替代 Chroma
    - similarity_search 内置 BM25 关键词检索 + 向量语义检索的混合搜索
    - 其余公共接口保持一致，上层代码无需修改
"""
import logging
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus, BM25BuiltInFunction

from app.core.vectorstore.models import SearchResult, SourceInfo, VersionInfo

logger = logging.getLogger(__name__)


class MilvusVectorStoreManager:
    """Milvus 向量库管理器，支持混合检索。

    使用 langchain_milvus.Milvus 作为底层存储，
    配合 BM25BuiltInFunction 实现 dense + sparse 混合搜索。

    Attributes:
        _embedding: Embedding 实例（生成 dense 向量）。
        _uri: Milvus 服务地址。
        _token: 认证 token。
        _collection_name: 集合名称。
        _dense_weight: 混合检索中 dense 权重。
        _sparse_weight: 混合检索中 sparse 权重。
        _store: Milvus 实例（懒加载）。
    """

    def __init__(
        self,
        embedding: Embeddings,
        uri: str = "http://localhost:19530",
        token: str = "root:Milvus",
        collection_name: str = "knowledge_base",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ):
        self._embedding = embedding
        self._uri = uri
        self._token = token
        self._collection_name = collection_name
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._store: Milvus | None = None

    def _get_store(self) -> Milvus:
        """懒加载 Milvus 实例。"""
        if self._store is None:
            logger.info(
                "初始化 Milvus: collection=%s, uri=%s",
                self._collection_name,
                self._uri,
            )
            self._store = Milvus(
                embedding_function=self._embedding,
                builtin_function=BM25BuiltInFunction(),
                # dense 由 embedding_function 生成，sparse 由 BM25BuiltInFunction 生成
                vector_field=["dense", "sparse"],
                connection_args={"uri": self._uri, "token": self._token},
                collection_name=self._collection_name,
                auto_id=False,
                primary_field="id",
                text_field="text",
                enable_dynamic_field=True,
                drop_old=False,
            )
        return self._store

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int, version: int) -> str:
        """生成确定性 ID: {source}::{version}::{chunk_index}。"""
        return f"{source}::{version}::{chunk_index}"

    # ================================================================
    # 基础操作
    # ================================================================

    def add_documents(
        self, documents: list[Document], source: str, version: int
    ) -> list[str]:
        """写入文档块到 Milvus。

        dense 向量由 embedding 函数自动生成，sparse 向量（BM25）由 Milvus 服务端自动生成。

        Args:
            documents: 待写入的 chunk 列表。
            source: 来源文件名。
            version: 版本号。

        Returns:
            list[str]: 生成的 chunk ID 列表。
        """
        if not documents:
            return []

        store = self._get_store()
        texts = []
        ids = []
        metadatas = []

        for i, doc in enumerate(documents):
            doc.metadata["_chunk_index"] = i
            doc.metadata["_version"] = version
            doc.metadata["_is_current"] = True
            doc.metadata.setdefault("source", source)

            chunk_id = self._make_chunk_id(source, i, version)
            ids.append(chunk_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

        # langchain_milvus.Milvus 通过 add_texts 写入
        # dense 向量自动由 embedding_function 生成
        # sparse 向量自动由 BM25BuiltInFunction 生成
        store.add_texts(texts=texts, ids=ids, metadatas=metadatas)

        logger.info(
            "Milvus 写入 %d 个 chunk: source=%s, version=%d",
            len(ids), source, version,
        )
        return ids

    def similarity_search(
        self, query: str, top_k: int = 4
    ) -> list[SearchResult]:
        """混合检索：dense 语义匹配 + sparse BM25 关键词匹配。

        使用 WeightedRanker 融合两路结果。

        Args:
            query: 用户查询文本。
            top_k: 返回最相关的 top_k 个结果。

        Returns:
            list[SearchResult]: 检索结果列表。
        """
        store = self._get_store()

        # 混合检索：通过 ranker_type 和 ranker_params 传入 WeightedRanker
        results = store.similarity_search_with_score(
            query,
            k=top_k,
            expr='_is_current == true',
            ranker_type="weighted",
            ranker_params={"weights": [self._dense_weight, self._sparse_weight]},
        )

        return [
            SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
            )
            for doc, score in results
        ]

    def delete_by_source(self, source: str) -> int:
        """删除某文件所有版本的所有 chunk。

        Args:
            source: 文件名。

        Returns:
            int: 删除的 chunk 数量。
        """
        store = self._get_store()
        try:
            # 查询该 source 的所有 chunk
            results = store.similarity_search(
                "", k=10000,
                expr=f'source == "{source}"',
            )
            count = len(results)
            if count > 0:
                store.delete(expr=f'source == "{source}"')
                logger.info("Milvus 删除 %d 个 chunk: source=%s", count, source)
            return count
        except Exception as e:
            logger.error("Milvus 删除失败: source=%s, error=%s", source, e)
            return 0

    def get_sources(self) -> list[SourceInfo]:
        """列出所有当前版本的来源文件。

        Returns:
            list[SourceInfo]: 每个文件的基本信息。
        """
        store = self._get_store()
        try:
            results = store.similarity_search(
                "", k=10000,
                expr='_is_current == true',
            )
            counter: dict[str, int] = defaultdict(int)
            version_map: dict[str, int] = {}
            for doc in results:
                source = doc.metadata.get("source", "unknown")
                counter[source] += 1
                version_map[source] = doc.metadata.get("_version", 1)

            return [
                SourceInfo(
                    source=s, chunk_count=counter[s], version=version_map.get(s, 1)
                )
                for s in sorted(counter)
            ]
        except Exception:
            return []

    def count(self) -> int:
        """当前版本的 chunk 总数。"""
        try:
            results = self._get_store().similarity_search(
                "", k=10000, expr='_is_current == true',
            )
            return len(results)
        except Exception:
            return 0

    def count_all(self) -> int:
        """所有 chunk 总数（含历史版本）。"""
        try:
            results = self._get_store().similarity_search("", k=50000)
            return len(results)
        except Exception:
            return 0

    # ================================================================
    # 版本管理
    # ================================================================

    def upsert_documents(self, source: str, documents: list[Document]) -> int:
        """增量更新：删除旧版本 → 写入新版本。

        Args:
            source: 文件名。
            documents: 分块后的 Document 列表。

        Returns:
            int: 新写入的 chunk 数量。
        """
        if not documents:
            return 0

        current_version = self._get_current_version(source)
        new_version = current_version + 1

        if current_version > 0:
            # 归档旧版本：删掉当前版本
            try:
                store = self._get_store()
                store.delete(expr=f'source == "{source}" and _is_current == true')
                logger.info(
                    "Milvus 归档旧版本: source=%s, version=%d",
                    source, current_version,
                )
            except Exception as e:
                logger.warning("归档旧版本时出错（可能无旧数据）: %s", e)

        # 写入新版本
        ids = self.add_documents(documents, source=source, version=new_version)
        logger.info("Milvus upsert 完成: source=%s, version=%d", source, new_version)
        return len(ids)

    def clear(self) -> int:
        """清空集合。"""
        try:
            store = self._get_store()
            results = store.similarity_search("", k=50000)
            count = len(results)
            if count > 0:
                store.delete(expr="id != ''")
                logger.info("Milvus 清空集合: 删除 %d 个 chunk", count)
            return count
        except Exception:
            return 0

    def _get_current_version(self, source: str) -> int:
        """获取某文件当前最高版本号。"""
        try:
            store = self._get_store()
            results = store.similarity_search(
                "", k=10000,
                expr=f'source == "{source}" and _is_current == true',
            )
            if not results:
                return 0
            versions = set()
            for doc in results:
                v = doc.metadata.get("_version", 1)
                versions.add(v)
            return max(versions) if versions else 0
        except Exception:
            return 0

    # 版本管理（简化版，与 Chroma 版接口对齐）

    def rollback(self, source: str, target_version: int) -> bool:
        """回滚到指定版本（简化实现：删除当前版本，重新写入目标版本）。

        Milvus 不支持 metadata 更新，回滚需要重新插入。
        此方法为占位实现，实际需要从备份恢复。
        """
        logger.warning("Milvus 回滚功能暂未实现: source=%s, version=%d", source, target_version)
        return False

    def get_versions(self, source: str) -> list[VersionInfo]:
        """获取某文件的所有版本记录。"""
        try:
            store = self._get_store()
            results = store.similarity_search(
                "", k=10000,
                expr=f'source == "{source}"',
            )
            version_chunks: dict[int, dict] = {}
            for doc in results:
                ver = doc.metadata.get("_version", 1)
                is_cur = doc.metadata.get("_is_current", False)
                if ver not in version_chunks:
                    version_chunks[ver] = {"count": 0, "is_current": False}
                version_chunks[ver]["count"] += 1
                if is_cur:
                    version_chunks[ver]["is_current"] = True

            return sorted(
                [
                    VersionInfo(
                        source=source, version=v,
                        chunk_count=info["count"],
                        is_current=info["is_current"],
                    )
                    for v, info in version_chunks.items()
                ],
                key=lambda x: x.version,
            )
        except Exception:
            return []

    def delete_version(self, source: str, version: int) -> int:
        """删除指定版本的历史记录。"""
        try:
            store = self._get_store()
            store.delete(
                expr=f'source == "{source}" and _version == {version} and _is_current == false'
            )
            logger.info("Milvus 删除历史版本: source=%s, version=%d", source, version)
            return 1
        except Exception:
            return 0

    def rebuild(self, documents_by_source: dict[str, list[Document]]) -> int:
        """全量重建。"""
        self.clear()
        total = 0
        for source, docs in documents_by_source.items():
            ids = self.add_documents(docs, source=source, version=1)
            total += len(ids)
        logger.info("Milvus 全量重建完成: %d 个文件, 共 %d 个 chunk", len(documents_by_source), total)
        return total
