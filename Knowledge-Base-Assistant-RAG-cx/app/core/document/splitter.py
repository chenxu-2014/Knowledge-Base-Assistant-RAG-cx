import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """文档分块器"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须大于 0，当前值: {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap 不能为负，当前值: {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """将文档切分为固定大小的块，元数据自动继承"""
        if not documents:
            logger.warning("输入文档列表为空，返回空结果")
            return []

        logger.info(
            "开始分块: %d 篇文档, chunk_size=%d, overlap=%d",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )

        try:
            chunks = self._splitter.split_documents(documents)
        except Exception as e:
            logger.error("文档分块失败: %s", e)
            raise

        logger.info("分块完成: %d 篇文档 → %d 个块", len(documents), len(chunks))
        return chunks
