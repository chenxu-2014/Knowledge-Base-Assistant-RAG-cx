"""
请求/响应数据模型 —— API 层的 Pydantic 模型。

职责：定义 API 接口的请求体和响应体结构，FastAPI 自动进行参数校验和文档生成。

模型对应关系：
    ChatRequest        → POST /api/chat 的请求体
    ChatResponse       → POST /api/chat 的响应体
    SourceDocument     → ChatResponse 中的引用来源
    DocumentUploadResponse → POST /api/documents/upload 的响应体
    DocumentInfo       → GET /api/documents/ 的单个文档信息
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。

    Attributes:
        question: 用户问题，不能为空。
        chat_history: 可选的对话历史，格式 [{"role": "user/assistant", "content": "..."}]。
    """
    question: str = Field(..., min_length=1, description="用户问题")
    chat_history: list[dict] = Field(default_factory=list, description="对话历史")


class SourceDocument(BaseModel):
    """引用来源 —— 回答所依据的文档片段信息。

    Attributes:
        content: 文档片段的文本内容。
        source: 来源文件名。
        score: 相似度分数（0~1）。
        metadata: 完整的元信息字典。
    """
    content: str
    source: str = "未知"
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class ChatResponseChenxu(BaseModel):
    """调试用响应体。"""
    answer: str


class ChatResponse(BaseModel):
    """聊天响应体。

    Attributes:
        answer: LLM 生成的回答文本。
        sources: 回答所依据的引用来源列表。
    """
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """文档上传响应体。

    Attributes:
        doc_id: 文档 ID（当前使用文件名）。
        filename: 文件名。
        chunk_count: 分块后的 chunk 数量。
        message: 操作结果消息。
    """
    doc_id: str
    filename: str
    chunk_count: int
    message: str = "上传成功"


class DocumentInfo(BaseModel):
    """文档列表中的单个文档信息。

    Attributes:
        doc_id: 文档 ID。
        filename: 文件名。
    """
    doc_id: str
    filename: str
