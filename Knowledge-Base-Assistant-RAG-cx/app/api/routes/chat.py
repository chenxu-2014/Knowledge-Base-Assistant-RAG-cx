"""
聊天路由 —— RAG 在线阶段的 API 入口。

职责：接收用户问题，调用 RAGChain 生成回答，返回结果。

调用关系：
    POST /api/chat
      └─> chat_endpoint()
           └─> chain.invoke(question, chat_history)
                └─> RAGChain.invoke() → RAGResult

    POST /api/chat/stream
      └─> chat_stream_endpoint()
           └─> chain.stream_events(question, chat_history) → SSE
"""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, SourceDocument, ChatResponseChenxu

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: Request, body: ChatRequest):
    """知识库问答接口。

    流程：
        1. 从 app.state.rag 获取 RAGChain 实例
        2. 调用 chain.invoke() 执行完整的 RAG 流程
           （向量检索 → 拼接上下文 → 调用 LLM → 生成回答）
        3. 将 RAGResult 转换为 ChatResponse 返回

    Args:
        req: FastAPI Request 对象，通过 req.app.state.rag 访问 RAG 组件。
        body: 请求体，包含 question（问题）和 chat_history（对话历史）。

    Returns:
        ChatResponse: 包含 answer（回答）和 sources（引用来源）。
    """
    # 获取 RAG 链（在 main.py lifespan 中初始化）
    rag = req.app.state.rag
    chain = rag["chain"]

    # 执行 RAG 调用：检索 → 拼接 Prompt → 调用 LLM
    result = chain.invoke(body.question, chat_history=body.chat_history)

    # 将内部 SourceDocument 转换为 API 响应模型
    sources = [
        SourceDocument(
            content=s.content,
            source=s.source,
            score=s.score,
            metadata=s.metadata,
        )
        for s in result.sources
    ]

    return ChatResponse(answer=result.answer, sources=sources)


@router.post("/chat/stream")
async def chat_stream_endpoint(req: Request, body: ChatRequest):
    """知识库问答流式接口（SSE），包含思考过程和引用来源。

    返回 SSE 事件流，每个事件格式：
        data: {"event": "sources", "data": [...]}
        data: {"event": "thinking", "data": "..."}
        data: {"event": "content", "data": "..."}
        data: {"event": "done", "data": ""}
    """
    rag = req.app.state.rag
    chain = rag["chain"]

    def event_generator():
        for event in chain.stream_events(body.question, chat_history=body.chat_history):
            yield f"data: {event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/chatLoc", response_model=ChatResponseChenxu)
async def chat_endpoint(req: Request, body: ChatRequest):
    """知识库问答接口（本地调试用）。"""
    print("==="*20)
    print(Request)
    print(ChatRequest)
    print("==="*20)
    return ChatResponseChenxu(answer="answer:chenxu xiaoshuaige")
