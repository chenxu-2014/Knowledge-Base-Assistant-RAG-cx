"""
Chroma 向量库管理器 —— RAG 离线阶段第四步的核心实现。

职责：
    1. 将嵌入后的 Document 写入 Chroma（add_documents / upsert_documents）
    2. 在线阶段提供语义检索接口（similarity_search）
    3. 版本管理 —— 同一文件多次上传时，旧版本归档，新版本生效
    4. 支持版本回滚（rollback）

版本管理机制：
    每个 chunk 的 metadata 中维护 _version（版本号）和 _is_current（是否当前版本）。
    首次上传：version=1, _is_current=True
    再次上传：旧版本 chunk 的 _is_current 标记为 False（归档），新 chunk 以 version+1 写入
    检索时：只查 _is_current=True 的 chunk，历史版本不参与检索

ID 生成规则：
    {source}::{version}::{chunk_index}
    例如: "report.pdf::1::0" 表示 report.pdf 的第 1 版的第 0 个 chunk

调用关系：
    离线写入: document.py 中 upload_document()
      └─> vectorstore.upsert_documents(filename, chunks)

    在线检索: rag_chain.py 中 _retrieve()
      └─> vectorstore.similarity_search(question, top_k)
"""
import hashlib
import logging
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.vectorstore.models import SearchResult, SourceInfo, VersionInfo

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Chroma 向量库管理器，支持版本历史与回滚。

    懒加载设计：_store 在首次访问时才初始化 Chroma 实例，避免启动时阻塞。

    Attributes:
        _embedding: Embedding 实例，传给 Chroma 用于向量化查询和文档。
        _persist_dir: Chroma 持久化目录路径。
        _collection_name: Chroma 集合名称（类似数据库表名）。
        _store: Chroma 实例（懒加载）。
    """

    def __init__(self, embedding: Embeddings, persist_dir: str, collection_name: str):
        self._embedding = embedding
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._store: Chroma | None = None  # 懒加载，首次使用时初始化

    # ================================================================
    # 内部方法
    # ================================================================

    def _get_store(self) -> Chroma:
        """懒加载 Chroma 实例。

        首次调用时初始化 Chroma，后续复用同一实例。
        persist_directory 指定磁盘持久化路径，重启后数据不丢失。
        """
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
        """生成确定性 ID: {source}::{version}::{chunk_index}。

        确定性 ID 的好处：
            - 同一文件、同一版本的同一 chunk，ID 始终相同
            - 避免重复写入时产生重复数据
            - 方便调试和版本追踪
        """
        return f"{source}::{version}::{chunk_index}"

    def _get_current_version(self, source: str) -> int:
        """获取某文件当前最高版本号。

        通过查询 _is_current=True 的 chunk，从其 ID 中提取版本号。
        返回 0 表示该文件从未上传过。
        """
        store = self._get_store()
        data = store.get(
            where={"$and": [{"source": source}, {"_is_current": True}]},
            include=[],  # 只需要 ID，不需要文档内容
        )
        if not data["ids"]:
            return 0
        # 从 ID（如 "report.pdf::2::5"）中提取版本号
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
        """获取某文件当前版本的所有 chunk。

        用于版本切换时：先取出旧 chunk，修改 metadata 后重新写入。
        """
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
        """写入文档块到 Chroma，返回生成的 ID 列表。

        为每个 chunk 注入 _chunk_index、_version、_is_current 等元信息，
        然后调用 Chroma.add_documents() 写入。

        Args:
            documents: 待写入的 chunk 列表（DocumentPipeline 的输出）。
            source: 来源文件名（用于 ID 生成和 metadata 标记）。
            version: 版本号。

        Returns:
            list[str]: 生成的 chunk ID 列表。
        """
        if not documents:
            return []

        store = self._get_store()
        ids = []
        for i, doc in enumerate(documents):
            doc.metadata["_chunk_index"] = i
            doc.metadata["_version"] = version
            doc.metadata["_is_current"] = True
            doc.metadata.setdefault("source", source)  # 若无 source 字段则注入
            ids.append(self._make_chunk_id(source, i, version))

        store.add_documents(documents, ids=ids)
        logger.info(
            "写入 %d 个 chunk: source=%s, version=%d", len(ids), source, version
        )
        return ids

    def similarity_search(
        self, query: str, top_k: int = 4
    ) -> list[SearchResult]:
        """语义检索 —— RAG 在线阶段的核心接口。

        使用余弦相似度从向量库中检索与 query 最相关的 chunk。
        只返回 _is_current=True 的当前版本，历史版本不参与检索。

        Args:
            query: 用户查询文本。
            top_k: 返回最相关的 top_k 个结果，默认 4。

        Returns:
            list[SearchResult]: 检索结果列表，按相似度降序排列。
        """
        store = self._get_store()
        # similarity_search_with_relevance_scores 返回 (Document, score) 对
        # score 是余弦相似度，范围 0~1，越大越相关
        results = store.similarity_search_with_relevance_scores(
            query, k=top_k, filter={"_is_current": True}  # 只查当前版本
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
        data = store.get(where={"source": source}, include=[])
        count = len(data["ids"])
        if count > 0:
            store.delete(where={"source": source})
            logger.info("删除 %d 个 chunk: source=%s", count, source)
        return count

    def get_sources(self) -> list[SourceInfo]:
        """列出所有当前版本的来源文件。

        Returns:
            list[SourceInfo]: 每个文件的基本信息（名称、chunk 数、版本号）。
        """
        store = self._get_store()
        data = store.get(
            where={"_is_current": True},
            include=["metadatas"],
        )
        # 统计每个 source 的 chunk 数和版本号
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
        """集合中当前版本的 chunk 总数。"""
        store = self._get_store()
        data = store.get(where={"_is_current": True}, include=[])
        return len(data["ids"])

    def count_all(self) -> int:
        """集合中所有 chunk 总数（含历史版本）。"""
        store = self._get_store()
        return store._collection.count()

    # ================================================================
    # 版本管理
    # ================================================================

    def upsert_documents(self, source: str, documents: list[Document]) -> int:
        """增量更新 —— RAG 离线阶段写入向量库的核心接口。

        版本管理逻辑：
            - 首次写入：version=1，直接写入
            - 非首次写入：
                1. 查询当前版本号 current_version
                2. 将当前版本的所有 chunk 的 _is_current 标记为 False（归档保留）
                3. 以 current_version+1 的版本号写入新 chunk

        调用关系：
            document.py 中 upload_document()
              └─> vectorstore.upsert_documents(filename, chunks)

        Args:
            source: 文件名，用作版本管理的唯一标识。
            documents: 分块后的 Document 列表（DocumentPipeline.process() 的输出）。

        Returns:
            int: 新写入的 chunk 数量。
        """
        if not documents:
            return 0

        current_version = self._get_current_version(source)
        new_version = current_version + 1

        if current_version > 0:
            # 非首次上传：将旧版本标记为非当前（归档）
            store = self._get_store()
            old_data = store.get(
                where={"$and": [{"source": source}, {"_is_current": True}]},
                include=[],
            )
            if old_data["ids"]:
                # Chroma 不支持直接更新 metadata，采用"删后重写"策略
                old_docs = self._get_current_chunks(source)
                store.delete(ids=old_data["ids"])
                for doc in old_docs:
                    doc.metadata["_is_current"] = False  # 标记为历史版本
                store.add_documents(old_docs, ids=old_data["ids"])
                logger.info(
                    "归档旧版本: source=%s, version=%d, %d 个 chunk",
                    source,
                    current_version,
                    len(old_data["ids"]),
                )

        # 写入新版本
        ids = self.add_documents(documents, source=source, version=new_version)
        logger.info("upsert 完成: source=%s, version=%d", source, new_version)
        return len(ids)

    def rollback(self, source: str, target_version: int) -> bool:
        """回滚到指定版本。

        逻辑：
            1. 检查目标版本是否存在（_is_current=False）
            2. 当前版本标记为 _is_current=False
            3. 目标版本标记为 _is_current=True

        Args:
            source: 文件名。
            target_version: 目标版本号。

        Returns:
            bool: 回滚是否成功。
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

        # 当前版本 → 非当前（删后重写 metadata）
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
        """获取某文件的所有版本记录。

        Args:
            source: 文件名。

        Returns:
            list[VersionInfo]: 版本列表，按版本号升序排列。
        """
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
        """删除指定版本的历史记录（不能删除当前版本）。

        Args:
            source: 文件名。
            version: 要删除的版本号。

        Returns:
            int: 删除的 chunk 数量，0 表示版本不存在或是当前版本。
        """
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
        """清空整个集合（删除所有 chunk）。"""
        store = self._get_store()
        data = store.get(include=[])
        count = len(data["ids"])
        if count > 0:
            store.delete(ids=data["ids"])
            logger.info("清空集合: 删除 %d 个 chunk", count)
        return count

    def rebuild(self, documents_by_source: dict[str, list[Document]]) -> int:
        """全量重建：清空集合 + 逐文件写入（version=1）。

        Args:
            documents_by_source: {文件名: chunk列表} 的字典。

        Returns:
            int: 总共写入的 chunk 数量。
        """
        self.clear()
        total = 0
        for source, docs in documents_by_source.items():
            ids = self.add_documents(docs, source=source, version=1)
            total += len(ids)
        logger.info("全量重建完成: %d 个文件, 共 %d 个 chunk", len(documents_by_source), total)
        return total
