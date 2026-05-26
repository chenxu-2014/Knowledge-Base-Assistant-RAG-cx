"""
递归分块器 —— 按分隔符层级切分，保留语义边界（默认推荐策略）。

底层使用 langchain 的 RecursiveCharacterTextSplitter，
按优先级尝试多种分隔符（\n\n → \n → 空格 → 字符），在保留语义的前提下
将文本切分为不超过 chunk_size 的块。

与 FixedLengthSplitter 的区别：
    - FixedLength: 按字符位置硬切，可能在句子中间断开
    - Recursive:   按语义边界切分，优先在段落/句子边界处断开

使用场景：
    通用场景的首选策略，适合大多数结构化文档。
"""
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)


class RecursiveSplitter(BaseDocumentSplitter):
    """递归分块器，按分隔符层级切分，保留语义边界。

    langchain RecursiveCharacterTextSplitter 的分隔符优先级：
        ["\n\n", "\n", " ", ""] —— 先按双换行（段落）切，再按单换行切，
        再按空格切，最后按字符切。这样尽量在语义边界处断开。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """初始化递归分块器。

        Args:
            chunk_size: 每个 chunk 的最大字符数。
            chunk_overlap: 相邻 chunk 之间的重叠字符数。
        """
        super().__init__(chunk_size, chunk_overlap)
        # 内部委托给 langchain 的 RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # 以字符数衡量长度
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """使用递归字符分割器切分文档。

        流程：
            1. 尝试按 \n\n（段落）分割
            2. 若某块仍超限，按 \n（换行）分割
            3. 若仍超限，按空格分割
            4. 最终按单个字符分割

        Args:
            documents: 待切分的文档列表。

        Returns:
            list[Document]: 切分后的 chunk 列表，metadata 自动继承。
        """
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
            # split_documents 会自动处理多篇文档，metadata 从原文档继承到每个 chunk
            chunks = self._splitter.split_documents(documents)
        except Exception as e:
            logger.error("递归分块失败: %s", e)
            raise

        logger.info("递归分块完成: %d 篇文档 → %d 个块", len(documents), len(chunks))
        return chunks
