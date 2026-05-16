import logging

from langchain_core.documents import Document

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)


class FixedLengthSplitter(BaseDocumentSplitter):
    """固定长度分块器，按字符位置硬切，不考虑语义边界"""

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            logger.warning("输入文档列表为空，返回空结果")
            return []

        logger.info(
            "固定长度分块: %d 篇文档, chunk_size=%d, overlap=%d",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )

        chunks: list[Document] = []
        step = self.chunk_size - self.chunk_overlap

        for doc in documents:
            text = doc.page_content
            if not text:
                continue

            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append(
                    Document(page_content=chunk_text, metadata=doc.metadata.copy())
                )
                start += step

        logger.info("固定长度分块完成: %d 篇文档 → %d 个块", len(documents), len(chunks))
        return chunks
