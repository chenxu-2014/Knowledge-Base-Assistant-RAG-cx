import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)


class RecursiveSplitter(BaseDocumentSplitter):
    """递归分块器，按分隔符层级切分，保留语义边界"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        super().__init__(chunk_size, chunk_overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            logger.warning("输入文档列表为空，返回空结果")
            return []

        logger.info(
            "递归分块: %d 篇文档, chunk_size=%d, overlap=%d",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )

        try:
            chunks = self._splitter.split_documents(documents)
        except Exception as e:
            logger.error("递归分块失败: %s", e)
            raise

        logger.info("递归分块完成: %d 篇文档 → %d 个块", len(documents), len(chunks))
        return chunks
