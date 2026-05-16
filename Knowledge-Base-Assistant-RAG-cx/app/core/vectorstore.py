from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """Chroma 向量库管理器"""

    def __init__(self, persist_dir: str, collection_name: str, embedding: Embeddings):
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding = embedding
        self._store: Chroma | None = None

    def initialize(self) -> Chroma:
        """初始化或加载已有的 Chroma 集合"""
        raise NotImplementedError

    def add_documents(self, documents: list[Document]) -> list[str]:
        """将文档写入向量库，返回文档 ID 列表"""
        raise NotImplementedError

    def similarity_search(self, query: str, top_k: int = 4) -> list[Document]:
        """相似度检索"""
        raise NotImplementedError

    def delete_document(self, doc_id: str) -> None:
        """根据文档 ID 删除"""
        raise NotImplementedError

    def list_documents(self) -> list[dict]:
        """列出集合中的文档元信息"""
        raise NotImplementedError
