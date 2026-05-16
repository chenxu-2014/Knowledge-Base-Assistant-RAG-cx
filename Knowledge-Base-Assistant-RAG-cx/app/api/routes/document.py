import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, Request, HTTPException

from app.schemas.chat import DocumentUploadResponse, DocumentInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = {".pdf", ".docx", ".doc", ".md", ".txt"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(req: Request, file: UploadFile):
    """上传文档到知识库"""
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"不支持的文件类型: {ext}，支持: {', '.join(SUPPORTED_EXT)}")

    save_path = UPLOAD_DIR / filename
    content = await file.read()
    save_path.write_bytes(content)
    logger.info("文件已保存: %s (%d bytes)", filename, len(content))

    rag = req.app.state.rag
    pipeline = rag["pipeline"]
    vectorstore = rag["vectorstore"]

    try:
        chunks = pipeline.process(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"文档处理失败: {e}")

    if chunks:
        vectorstore.upsert_documents(filename, chunks)

    return DocumentUploadResponse(
        doc_id=filename,
        filename=filename,
        chunk_count=len(chunks),
        message="上传成功",
    )


@router.get("/", response_model=list[DocumentInfo])
async def list_documents(req: Request):
    """列出知识库中的文档"""
    rag = req.app.state.rag
    vectorstore = rag["vectorstore"]
    sources = vectorstore.get_sources()
    return [DocumentInfo(doc_id=s.source, filename=s.source) for s in sources]


@router.delete("/{filename:path}")
async def delete_document(req: Request, filename: str):
    """删除知识库中的文档"""
    rag = req.app.state.rag
    vectorstore = rag["vectorstore"]
    vectorstore.delete_by_source(filename)

    local_file = UPLOAD_DIR / filename
    local_file.unlink(missing_ok=True)

    return {"message": "删除成功", "filename": filename}
