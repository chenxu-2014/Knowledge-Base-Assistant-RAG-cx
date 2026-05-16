import hashlib
import logging
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.vectorstore.models import SearchResult, SourceInfo, VersionInfo

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Chroma 向量库管理器，支持版本历史与回滚"""

    def __init__(self, embedding: Embeddings, persist_dir: str, collection_name: str):
        self._embedding = embedding
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._store: Chroma | None = None

    # ================================================================
    # 内部方法
    # ================================================================

    def _get_store(self) -> Chroma:
        """懒加载 Chroma 实例"""
        if self._store is None:
            logger.info(
                "初始化 Chroma: collection=%s, persist=%s",
                self._collection_name,
                self._persist_dir,
            )
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding,
                persist_directory=self._persist_dir,
            )
        return self._store

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int, version: int) -> str:
        """生成确定性 ID: source::version::index"""
        return f"{source}::{version}::{chunk_index}"

    def _get_current_version(self, source: str) -> int:
        """获取某文件当前最高版本号"""
        store = self._get_store()
        data = store.get(
            where={"$and": [{"source": source}, {"_is_current": True}]},
            include=[],
        )
        if not data["ids"]:
            return 0
        # 从 ID 中提取版本号
        versions = set()
        for id_ in data["ids"]:
            parts = id_.split("::")
            if len(parts) >= 2:
                try:
                    versions.add(int(parts[1]))
                except ValueError:
                    pass
        return max(versions) if versions else 0

    def _get_current_chunks(self, source: str) -> list[Document]:
        """获取某文件当前版本的所有 chunk"""
        store = self._get_store()
        data = store.get(
            where={"$and": [{"source": source}, {"_is_current": True}]},
            include=["documents", "metadatas"],
        )
        chunks = []
        for text, meta in zip(data["documents"], data["metadatas"]):
            chunks.append(Document(page_content=text, metadata=meta))
        return chunks

    # ================================================================
    # 基础操作
    # ================================================================

    def add_documents(
        self, documents: list[Document], source: str, version: int
    ) -> list[str]:
        """写入文档，返回生成的 ID 列表"""
        if not documents:
            return []

        store = self._get_store()
        ids = []
        for i, doc in enumerate(documents):
            doc.metadata["_chunk_index"] = i
            doc.metadata["_version"] = version
            doc.metadata["_is_current"] = True
            doc.metadata.setdefault("source", source)
            ids.append(self._make_chunk_id(source, i, version))

        store.add_documents(documents, ids=ids)
        logger.info(
            "写入 %d 个 chunk: source=%s, version=%d", len(ids), source, version
        )
        return ids

    def similarity_search(
        self, query: str, top_k: int = 4
    ) -> list[SearchResult]:
        """语义检索，只返回当前版本"""
        store = self._get_store()
        results = store.similarity_search_with_relevance_scores(
            query, k=top_k, filter={"_is_current": True}
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
        """删除某文件所有版本的所有 chunk"""
        store = self._get_store()
        data = store.get(where={"source": source}, include=[])
        count = len(data["ids"])
        if count > 0:
            store.delete(where={"source": source})
            logger.info("删除 %d 个 chunk: source=%s", count, source)
        return count

    def get_sources(self) -> list[SourceInfo]:
        """列出所有当前版本的来源文件"""
        store = self._get_store()
        data = store.get(
            where={"_is_current": True},
            include=["metadatas"],
        )
        counter: dict[str, int] = defaultdict(int)
        version_map: dict[str, int] = {}
        for meta in data["metadatas"]:
            source = meta.get("source", "unknown")
            counter[source] += 1
            version_map[source] = meta.get("_version", 1)

        return [
            SourceInfo(
                source=s, chunk_count=counter[s], version=version_map.get(s, 1)
            )
            for s in sorted(counter)
        ]

    def count(self) -> int:
        """集合中当前版本的 chunk 总数"""
        store = self._get_store()
        data = store.get(where={"_is_current": True}, include=[])
        return len(data["ids"])

    def count_all(self) -> int:
        """集合中所有 chunk 总数（含历史版本）"""
        store = self._get_store()
        return store._collection.count()

    # ================================================================
    # 版本管理
    # ================================================================

    def upsert_documents(self, source: str, documents: list[Document]) -> int:
        """
        增量更新：
        - 首次写入：version=1，直接写入
        - 非首次：将旧版本标记为非当前，version+1 写入新 chunk
        """
        if not documents:
            return 0

        current_version = self._get_current_version(source)
        new_version = current_version + 1

        if current_version > 0:
            # 将旧版本标记为非当前（保留历史）
            store = self._get_store()
            old_data = store.get(
                where={"$and": [{"source": source}, {"_is_current": True}]},
                include=[],
            )
            if old_data["ids"]:
                # Chroma 不支持直接更新 metadata，需要删后重写
                old_docs = self._get_current_chunks(source)
                store.delete(ids=old_data["ids"])
                for doc in old_docs:
                    doc.metadata["_is_current"] = False
                store.add_documents(old_docs, ids=old_data["ids"])
                logger.info(
                    "归档旧版本: source=%s, version=%d, %d 个 chunk",
                    source,
                    current_version,
                    len(old_data["ids"]),
                )

        ids = self.add_documents(documents, source=source, version=new_version)
        logger.info("upsert 完成: source=%s, version=%d", source, new_version)
        return len(ids)

    def rollback(self, source: str, target_version: int) -> bool:
        """
        回滚到指定版本：
        - 当前版本标记为非当前
        - 目标版本标记为当前
        """
        store = self._get_store()

        # 检查目标版本是否存在
        target_data = store.get(
            where={
                "$and": [
                    {"source": source},
                    {"_version": target_version},
                    {"_is_current": False},
                ]
            },
            include=["documents", "metadatas"],
        )
        if not target_data["ids"]:
            logger.error(
                "回滚失败: source=%s, version=%d 不存在", source, target_version
            )
            return False

        current_version = self._get_current_version(source)
        if current_version == 0:
            logger.error("回滚失败: source=%s 无当前版本", source)
            return False

        # 当前版本 → 非当前
        current_data = store.get(
            where={"$and": [{"source": source}, {"_is_current": True}]},
            include=["documents", "metadatas"],
        )
        if current_data["ids"]:
            store.delete(ids=current_data["ids"])
            for meta in current_data["metadatas"]:
                meta["_is_current"] = False
            restored_current = [
                Document(page_content=t, metadata=m)
                for t, m in zip(current_data["documents"], current_data["metadatas"])
            ]
            store.add_documents(restored_current, ids=current_data["ids"])

        # 目标版本 → 当前
        store.delete(ids=target_data["ids"])
        for meta in target_data["metadatas"]:
            meta["_is_current"] = True
        restored_target = [
            Document(page_content=t, metadata=m)
            for t, m in zip(target_data["documents"], target_data["metadatas"])
        ]
        store.add_documents(restored_target, ids=target_data["ids"])

        logger.info(
            "回滚成功: source=%s, %d → %d", source, current_version, target_version
        )
        return True

    def get_versions(self, source: str) -> list[VersionInfo]:
        """获取某文件的所有版本记录"""
        store = self._get_store()
        data = store.get(where={"source": source}, include=["metadatas"])

        version_chunks: dict[int, dict] = {}
        for meta in data["metadatas"]:
            ver = meta.get("_version", 1)
            is_cur = meta.get("_is_current", False)
            if ver not in version_chunks:
                version_chunks[ver] = {"count": 0, "is_current": False}
            version_chunks[ver]["count"] += 1
            if is_cur:
                version_chunks[ver]["is_current"] = True

        return sorted(
            [
                VersionInfo(
                    source=source,
                    version=v,
                    chunk_count=info["count"],
                    is_current=info["is_current"],
                )
                for v, info in version_chunks.items()
            ],
            key=lambda x: x.version,
        )

    def delete_version(self, source: str, version: int) -> int:
        """删除指定版本的历史记录（不能删除当前版本）"""
        store = self._get_store()
        data = store.get(
            where={
                "$and": [
                    {"source": source},
                    {"_version": version},
                    {"_is_current": False},
                ]
            },
            include=[],
        )
        count = len(data["ids"])
        if count == 0:
            logger.warning("版本不存在或是当前版本，跳过删除: source=%s, v=%d", source, version)
            return 0
        store.delete(ids=data["ids"])
        logger.info("删除历史版本: source=%s, version=%d, %d 个 chunk", source, version, count)
        return count

    # ================================================================
    # 全量重建
    # ================================================================

    def clear(self) -> int:
        """清空整个集合"""
        store = self._get_store()
        data = store.get(include=[])
        count = len(data["ids"])
        if count > 0:
            store.delete(ids=data["ids"])
            logger.info("清空集合: 删除 %d 个 chunk", count)
        return count

    def rebuild(self, documents_by_source: dict[str, list[Document]]) -> int:
        """全量重建：清空集合 + 逐文件写入（version=1）"""
        self.clear()
        total = 0
        for source, docs in documents_by_source.items():
            ids = self.add_documents(docs, source=source, version=1)
            total += len(ids)
        logger.info("全量重建完成: %d 个文件, 共 %d 个 chunk", len(documents_by_source), total)
        return total
