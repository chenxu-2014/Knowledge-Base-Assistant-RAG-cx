"""
固定长度分块器 —— 按字符位置硬切，不考虑语义边界。

最简单的分块策略：以 chunk_size 为窗口、chunk_overlap 为步长滑动切分。
不关心标题、段落、句子等语义结构，适合无明显结构的纯文本。

使用场景：
    文档没有明确的段落/标题结构，或需要最简单快速的分块。

切分示意（chunk_size=10, overlap=3）：
    文本: "ABCDEFGHIJKLMNO"
    chunk1: "ABCDEFGHIJ"  (0:10)
    chunk2: "HIJKLMNOP"   (7:17, 步长=10-3=7)
    chunk3: "OPQRST"      (14:...)
"""
import logging

from langchain_core.documents import Document

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)


class FixedLengthSplitter(BaseDocumentSplitter):
    """固定长度分块器，按字符位置硬切，不考虑语义边界。"""

    def split(self, documents: list[Document]) -> list[Document]:
        """对每个 Document 按固定字符长度切分。

        切分逻辑：
            1. 遍历每个 Document 的 page_content
            2. 以 chunk_size 为窗口大小
            3. 以 (chunk_size - chunk_overlap) 为步长滑动
            4. 每个窗口生成一个 chunk Document，metadata 继承自原文档

        Args:
            documents: 待切分的文档列表。

        Returns:
            list[Document]: 固定长度切分后的 chunk 列表。
        """
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
        # 步长 = 窗口大小 - 重叠量，控制相邻 chunk 的滑动距离
        step = self.chunk_size - self.chunk_overlap

        for doc in documents:
            text = doc.page_content
            if not text:
                continue

            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                # 每个 chunk 复制原文档的 metadata（source、file_type 等）
                chunks.append(
                    Document(page_content=chunk_text, metadata=doc.metadata.copy())
                )
                start += step  # 滑动到下一个窗口

        logger.info("固定长度分块完成: %d 篇文档 → %d 个块", len(documents), len(chunks))
        return chunks
