"""
向量库数据模型 —— 定义检索结果和元信息的 Pydantic 模型。

SearchResult 是核心数据结构，贯穿检索 → rerank → 拼接上下文 → 返回给用户的全链路。
"""
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """语义检索结果 —— 单条从向量库中检索到的文档片段。

    Attributes:
        content: 文档片段的文本内容。
        metadata: 元信息字典，包含 source、file_type、_chunk_index 等。
        score: 与查询的相似度分数（0~1，越大越相关），来自 Chroma 的余弦相似度。
    """

    content: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None


class SourceInfo(BaseModel):
    """来源文件信息 —— 用于列出知识库中已有的文档。

    Attributes:
        source: 文件名。
        chunk_count: 当前版本的 chunk 数量。
        version: 版本号。
    """

    source: str
    chunk_count: int
    version: int


class VersionInfo(BaseModel):
    """某文件的版本记录 —— 用于版本管理和回滚。

    Attributes:
        source: 文件名。
        version: 版本号。
        chunk_count: 该版本的 chunk 数量。
        is_current: 是否为当前生效版本。
    """

    source: str
    version: int
    chunk_count: int
    is_current: bool
