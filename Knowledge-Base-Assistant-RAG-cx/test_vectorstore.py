"""
VectorStore 测试脚本。
使用 mock embedding，不依赖外部 API。
"""

import hashlib
import logging
import shutil
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test")

TEST_DIR = Path(__file__).parent / "test_chroma_db"


class MockEmbedding(Embeddings):
    """确定性 mock embedding，同一文本始终产生相同向量"""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _text_to_vector(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(self.dim):
            byte_val = h[i % len(h)]
            vec.append(byte_val / 255.0)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)


def make_docs(texts: list[str], source: str) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": source, "file_type": ".txt"})
        for t in texts
    ]


def setup():
    from app.core.vectorstore import VectorStoreManager

    return VectorStoreManager(
        embedding=MockEmbedding(),
        persist_dir=str(TEST_DIR),
        collection_name="test_collection",
    )


# ================================================================
# 基础操作
# ================================================================

def test_add_and_search(store):
    logger.info("=" * 50)
    logger.info("测试: 写入和检索")

    docs = make_docs(
        ["公司成立于2020年", "专注于人工智能技术", "员工手册第一章"],
        source="公司简介.txt",
    )
    ids = store.add_documents(docs, source="公司简介.txt", version=1)
    assert len(ids) == 3

    results = store.similarity_search("公司成立", top_k=2)
    assert len(results) >= 1
    assert results[0].score is not None
    logger.info("检索结果: %s (score=%.4f)", results[0].content[:20], results[0].score)


def test_count_and_sources(store):
    logger.info("=" * 50)
    logger.info("测试: 计数和来源列表")

    total = store.count()
    assert total >= 3

    sources = store.get_sources()
    assert any(s.source == "公司简介.txt" for s in sources)
    logger.info("总数: %d, 来源: %s", total, [(s.source, s.chunk_count) for s in sources])


def test_delete_by_source(store):
    logger.info("=" * 50)
    logger.info("测试: 按来源删除")

    docs = make_docs(["临时数据1", "临时数据2"], source="临时文件.txt")
    store.add_documents(docs, source="临时文件.txt", version=1)

    before = store.count()
    deleted = store.delete_by_source("临时文件.txt")
    after = store.count()

    assert deleted == 2
    assert after == before - 2
    logger.info("删除前: %d, 删除: %d, 删除后: %d", before, deleted, after)


# ================================================================
# 版本管理
# ================================================================

def test_upsert_first_time(store):
    logger.info("=" * 50)
    logger.info("测试: 首次 upsert")

    docs = make_docs(["第一章 总则", "第二章 考勤", "第三章 奖惩"], source="制度文档.txt")
    count = store.upsert_documents("制度文档.txt", docs)
    assert count == 3

    versions = store.get_versions("制度文档.txt")
    assert len(versions) == 1
    assert versions[0].version == 1
    assert versions[0].is_current is True
    logger.info("首次 upsert: %d chunk, version=1", count)


def test_upsert_update(store):
    logger.info("=" * 50)
    logger.info("测试: 二次 upsert（增量更新）")

    docs_v2 = make_docs(
        ["第一章 总则（修订版）", "第二章 考勤（修订版）", "第三章 奖惩（修订版）", "第四章 离职"],
        source="制度文档.txt",
    )
    count = store.upsert_documents("制度文档.txt", docs_v2)
    assert count == 4

    sources = store.get_sources()
    src = next(s for s in sources if s.source == "制度文档.txt")
    assert src.version == 2
    assert src.chunk_count == 4

    versions = store.get_versions("制度文档.txt")
    assert len(versions) == 2
    v1 = next(v for v in versions if v.version == 1)
    v2 = next(v for v in versions if v.version == 2)
    assert v1.is_current is False
    assert v2.is_current is True
    assert v1.chunk_count == 3
    assert v2.chunk_count == 4

    logger.info("版本: %s", [(v.version, v.chunk_count, v.is_current) for v in versions])


def test_rollback(store):
    logger.info("=" * 50)
    logger.info("测试: 版本回滚")

    ok = store.rollback("制度文档.txt", target_version=1)
    assert ok is True

    versions = store.get_versions("制度文档.txt")
    v1 = next(v for v in versions if v.version == 1)
    v2 = next(v for v in versions if v.version == 2)
    assert v1.is_current is True
    assert v2.is_current is False

    sources = store.get_sources()
    src = next(s for s in sources if s.source == "制度文档.txt")
    assert src.version == 1
    assert src.chunk_count == 3
    logger.info("回滚成功: version=%d, chunk=%d", src.version, src.chunk_count)


def test_rollback_nonexistent(store):
    logger.info("=" * 50)
    logger.info("测试: 回滚到不存在版本")

    ok = store.rollback("制度文档.txt", target_version=99)
    assert ok is False
    logger.info("回滚失败处理正确")


def test_delete_version(store):
    logger.info("=" * 50)
    logger.info("测试: 删除历史版本")

    deleted = store.delete_version("制度文档.txt", version=2)
    assert deleted == 4

    versions = store.get_versions("制度文档.txt")
    assert all(v.version != 2 for v in versions)
    logger.info("删除历史版本成功: 剩余版本 %s", [v.version for v in versions])

    deleted = store.delete_version("制度文档.txt", version=1)
    assert deleted == 0
    logger.info("当前版本不可删除: 正确")


# ================================================================
# 全量重建
# ================================================================

def test_rebuild(store):
    logger.info("=" * 50)
    logger.info("测试: 全量重建")

    all_docs = {
        "文档A.txt": make_docs(["A1", "A2", "A3"], source="文档A.txt"),
        "文档B.txt": make_docs(["B1", "B2"], source="文档B.txt"),
    }
    total = store.rebuild(all_docs)
    assert total == 5

    for src in ["文档A.txt", "文档B.txt"]:
        versions = store.get_versions(src)
        assert len(versions) == 1
        assert versions[0].version == 1

    sources = store.get_sources()
    assert len(sources) == 2
    logger.info("全量重建: %d 文件, %d chunk", len(all_docs), total)


def test_clear(store):
    logger.info("=" * 50)
    logger.info("测试: 清空集合")

    before = store.count()
    deleted = store.clear()
    after = store.count()

    assert before > 0
    assert deleted == before
    assert after == 0
    logger.info("清空: 删除 %d, 剩余 %d", deleted, after)


# ================================================================
# 主流程
# ================================================================

def cleanup():
    if TEST_DIR.exists():
        try:
            shutil.rmtree(TEST_DIR)
            logger.info("清理测试目录: %s", TEST_DIR)
        except PermissionError:
            logger.warning("清理失败（文件被占用），跳过")


def main():
    store = setup()

    try:
        test_add_and_search(store)
        test_count_and_sources(store)
        test_delete_by_source(store)

        test_upsert_first_time(store)
        test_upsert_update(store)
        test_rollback(store)
        test_rollback_nonexistent(store)
        test_delete_version(store)

        test_rebuild(store)
        test_clear(store)

        logger.info("=" * 50)
        logger.info("全部测试通过！")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
