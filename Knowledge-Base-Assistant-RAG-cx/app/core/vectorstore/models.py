from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """语义检索结果"""

    content: str
    metadata: dict = Field(default_factory=dict)
    score: float | None = None


class SourceInfo(BaseModel):
    """来源文件信息"""

    source: str
    chunk_count: int
    version: int


class VersionInfo(BaseModel):
    """某文件的版本记录"""

    source: str
    version: int
    chunk_count: int
    is_current: bool
