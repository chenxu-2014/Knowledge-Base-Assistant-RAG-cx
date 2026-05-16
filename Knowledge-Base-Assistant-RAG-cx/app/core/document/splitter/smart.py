import logging
import re

from langchain_core.documents import Document

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)

# 分隔符优先级：标题 → 段落 → 句子 → 字符
_SEPARATORS = [
    (r"\n(?=#{1,6}\s)", "heading"),           # Markdown 标题前换行
    (r"(?m)^(?=#{1,6}\s)", "heading"),         # 行首 Markdown 标题
    (r"\n(?=[^\S\n]*第[一二三四五六七八九十百千\d]+[章节篇部分])", "heading"),  # 中文编号标题
    (r"(?m)^[^\S\n]*第[一二三四五六七八九十百千\d]+[章节篇部分]", "heading"),
    (r"\n\n", "paragraph"),                     # 空行分段
    (r"(?<=[。！？.!?；;])\s*", "sentence"),    # 句末标点
    (r"(?<=\n)", "line"),                        # 单换行
    ("", "character"),                           # 字符级 fallback
]


class SmartSplitter(BaseDocumentSplitter):
    """
    智能分块器。
    优先级: 标题 → 段落 → 句子 → 字符，逐级 fallback。
    """

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            logger.warning("输入文档列表为空，返回空结果")
            return []

        logger.info(
            "智能分块: %d 篇文档, chunk_size=%d, overlap=%d",
            len(documents),
            self.chunk_size,
            self.chunk_overlap,
        )

        chunks: list[Document] = []
        for doc in documents:
            text = doc.page_content
            if not text:
                continue
            pieces = self._recursive_split(text)
            for piece in pieces:
                chunks.append(
                    Document(page_content=piece, metadata=doc.metadata.copy())
                )

        logger.info(
            "智能分块完成: %d 篇文档 → %d 个块", len(documents), len(chunks)
        )
        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        """按分隔符层级递归切分，直到所有块不超过 chunk_size"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for pattern, level in _SEPARATORS:
            if pattern == "":
                # 最终 fallback：字符级硬切
                return self._split_by_character(text)

            parts = re.split(pattern, text)
            parts = [p for p in parts if p.strip()]
            if len(parts) <= 1:
                continue

            # 合并小块，超大块递归
            merged = self._merge_parts(parts)
            if len(merged) > 1 or (merged and len(merged[0]) <= self.chunk_size):
                return merged

        return self._split_by_character(text)

    def _merge_parts(self, parts: list[str]) -> list[str]:
        """
        将切分后的片段合并，使每个块尽量接近 chunk_size。
        超过 chunk_size 的块递归进一步切分。
        """
        chunks: list[str] = []
        current = ""

        for part in parts:
            # 当前块 + 新片段不超过限制，直接拼接
            if current and len(current) + len(part) + 1 <= self.chunk_size:
                current += part
            else:
                # 存入已累积的块
                if current:
                    chunks.append(current)

                # 单个片段已超限，递归切分
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    def _split_by_character(self, text: str) -> list[str]:
        """最终 fallback：按字符数硬切"""
        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += step
        return chunks
