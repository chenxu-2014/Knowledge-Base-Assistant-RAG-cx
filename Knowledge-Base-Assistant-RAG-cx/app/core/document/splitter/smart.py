"""
智能分块器 —— 优先级: 标题 → 段落 → 句子 → 字符，逐级 fallback。

与 RecursiveSplitter 的区别：
    - RecursiveSplitter: 使用 langchain 内置分隔符（\n\n → \n → 空格 → 字符）
    - SmartSplitter:    自定义更精细的分隔符优先级，支持中文标题识别（"第X章"）

分隔符优先级（_SEPARATORS）：
    1. Markdown 标题（# 开头）      — 最强语义边界
    2. 中文编号标题（第X章/节/篇）  — 中文文档常见结构
    3. 空行（\n\n）                 — 段落分隔
    4. 句末标点（。！？等）          — 句子分隔
    5. 单换行（\n）                 — 行分隔
    6. 字符级                       — 最终 fallback

使用场景：
    Markdown 文档或有明确章节结构的中文文档。
"""
import logging
import re

from langchain_core.documents import Document

from app.core.document.splitter.base import BaseDocumentSplitter

logger = logging.getLogger(__name__)

# 分隔符优先级：标题 → 段落 → 句子 → 字符
# 每个元素是 (正则模式, 语义级别名称)
_SEPARATORS = [
    (r"\n(?=#{1,6}\s)", "heading"),           # Markdown 标题前换行
    (r"(?m)^(?=#{1,6}\s)", "heading"),         # 行首 Markdown 标题
    (r"\n(?=[^\S\n]*第[一二三四五六七八九十百千\d]+[章节篇部分])", "heading"),  # 中文编号标题前换行
    (r"(?m)^[^\S\n]*第[一二三四五六七八九十百千\d]+[章节篇部分]", "heading"),   # 行首中文编号标题
    (r"\n\n", "paragraph"),                     # 空行分段
    (r"(?<=[。！？.!?；;])\s*", "sentence"),    # 句末标点
    (r"(?<=\n)", "line"),                        # 单换行
    ("", "character"),                           # 字符级 fallback（兜底）
]


class SmartSplitter(BaseDocumentSplitter):
    """智能分块器。

    按分隔符优先级逐级尝试切分，直到所有块不超过 chunk_size。
    对超大块递归进一步细分，对小块尽量合并以接近 chunk_size。
    """

    def split(self, documents: list[Document]) -> list[Document]:
        """智能分块。

        对每个 Document 调用 _recursive_split() 进行层级化切分。

        Args:
            documents: 待切分的文档列表。

        Returns:
            list[Document]: 切分后的 chunk 列表。
        """
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
            # 递归切分，优先在语义边界处断开
            pieces = self._recursive_split(text)
            for piece in pieces:
                # 每个 chunk 复制原文档的 metadata
                chunks.append(
                    Document(page_content=piece, metadata=doc.metadata.copy())
                )

        logger.info(
            "智能分块完成: %d 篇文档 → %d 个块", len(documents), len(chunks)
        )
        return chunks

    def _recursive_split(self, text: str) -> list[str]:
        """按分隔符层级递归切分，直到所有块不超过 chunk_size。

        流程：
            1. 如果文本本身 ≤ chunk_size，直接返回
            2. 按优先级尝试每个分隔符：
               a. 若能切出多个片段 → 合并小块、递归处理超大块
               b. 若切不出（只有 1 片段）→ 尝试下一个分隔符
            3. 所有分隔符都失败 → 字符级硬切（_split_by_character）

        Args:
            text: 待切分的文本。

        Returns:
            list[str]: 切分后的文本片段列表。
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for pattern, level in _SEPARATORS:
            if pattern == "":
                # 最终 fallback：字符级硬切
                return self._split_by_character(text)

            parts = re.split(pattern, text)
            parts = [p for p in parts if p.strip()]
            if len(parts) <= 1:
                # 该分隔符切不出多个片段，尝试下一个
                continue

            # 合并小块到接近 chunk_size，超大块递归进一步切分
            merged = self._merge_parts(parts)
            if len(merged) > 1 or (merged and len(merged[0]) <= self.chunk_size):
                return merged

        # 所有分隔符都无效，字符级兜底
        return self._split_by_character(text)

    def _merge_parts(self, parts: list[str]) -> list[str]:
        """将切分后的片段合并，使每个块尽量接近 chunk_size。

        逻辑：
            - 遍历每个片段
            - 若当前块 + 新片段 ≤ chunk_size → 拼接
            - 否则存入当前块，新片段作为下一块的开头
            - 若单个片段已超 chunk_size → 递归进一步切分

        Args:
            parts: 待合并的文本片段列表。

        Returns:
            list[str]: 合并后的文本块列表。
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

                # 单个片段已超限，递归进一步切分
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
        """最终 fallback：按字符数硬切，逻辑同 FixedLengthSplitter。

        当所有语义分隔符都无效时使用（例如一整段无标点的长文本）。
        """
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
