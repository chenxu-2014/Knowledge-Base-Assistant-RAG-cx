from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    chat_history: list[dict] = Field(default_factory=list, description="对话历史")


class SourceDocument(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    message: str = "上传成功"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
