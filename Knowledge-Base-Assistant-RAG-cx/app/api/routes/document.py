"""
文档管理路由 —— RAG 离线阶段的 API 入口。

职责：提供文档上传、列出、删除的 REST 接口。

调用关系（上传流程）：
    POST /api/documents/upload
      └─> upload_document()
           ├─ 保存文件到 data/documents/
           ├─ pipeline.process(file_path)                # 加载 + 分块
           │    └─> DocumentPipeline.process()
           └─> vectorstore.upsert_documents(source, chunks)  # 写入向量库
                └─> VectorStoreManager.upsert_documents()
"""
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, Request, HTTPException

from app.schemas.chat import DocumentUploadResponse, DocumentInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# 上传文件保存目录
UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件格式白名单
SUPPORTED_EXT = {".pdf", ".docx", ".md", ".txt"}


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(req: Request, file: UploadFile):
    """上传文档到知识库 —— 触发完整的离线处理流程。

    流程：
        1. 校验文件格式（SUPPORTED_EXT 白名单）
        2. 保存文件到 data/documents/
        3. DocumentPipeline.process() — 加载 + 分块
        4. VectorStoreManager.upsert_documents() — 写入向量库（自动版本管理）

    Args:
        req: FastAPI Request 对象，通过 req.app.state.rag 访问 RAG 组件。
        file: 上传的文件。

    Returns:
        DocumentUploadResponse: 包含 doc_id、filename、chunk_count 等信息。

    Raises:
        HTTPException 400: 文件格式不支持。
        HTTPException 500: 文档处理失败。
    """
    # 校验文件格式
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise HTTPException(400, f"不支持的文件类型: {ext}，支持: {', '.join(SUPPORTED_EXT)}")

    # 保存文件到本地
    save_path = UPLOAD_DIR / filename
    content = await file.read()
    save_path.write_bytes(content)
    logger.info("文件已保存: %s (%d bytes)", filename, len(content))

    # 获取 RAG 组件
    rag = req.app.state.rag
    pipeline = rag["pipeline"]       # 文档处理管线（加载 + 分块）
    vectorstore = rag["vectorstore"] # 向量库管理器

    # 文档处理：加载 → 分块
    try:
        chunks = pipeline.process(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)  # 处理失败时清理已保存的文件
        raise HTTPException(500, f"文档处理失败: {e}")

    # 写入向量库（自动处理版本管理：旧版本归档，新版本生效）
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
    """列出知识库中的文档。

    通过 VectorStoreManager.get_sources() 获取当前版本的所有来源文件信息。

    Returns:
        list[DocumentInfo]: 文档列表。
    """
    rag = req.app.state.rag
    vectorstore = rag["vectorstore"]
    sources = vectorstore.get_sources()
    return [DocumentInfo(doc_id=s.source, filename=s.source) for s in sources]


@router.delete("/{filename:path}")
async def delete_document(req: Request, filename: str):
    """删除知识库中的文档。

    同时删除向量库中的所有版本 chunk 和本地文件。

    Args:
        filename: 文件名（支持路径格式，如 "subdir/report.pdf"）。

    Returns:
        dict: 删除结果信息。
    """
    rag = req.app.state.rag
    vectorstore = rag["vectorstore"]

    # 从向量库中删除所有版本的 chunk
    vectorstore.delete_by_source(filename)

    # 删除本地文件
    local_file = UPLOAD_DIR / filename
    local_file.unlink(missing_ok=True)

    return {"message": "删除成功", "filename": filename}
