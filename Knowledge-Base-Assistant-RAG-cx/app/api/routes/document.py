from fastapi import APIRouter, UploadFile

from app.schemas.chat import DocumentUploadResponse, DocumentInfo

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile):
    """上传文档到知识库"""
    raise NotImplementedError


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    """列出知识库中的文档"""
    raise NotImplementedError


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """删除知识库中的文档"""
    raise NotImplementedError
