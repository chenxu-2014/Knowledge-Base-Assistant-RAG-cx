import logging

from fastapi import APIRouter, Request

from app.schemas.chat import ChatRequest, ChatResponse, SourceDocument

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: Request, body: ChatRequest):
    """知识库问答接口"""
    rag = req.app.state.rag
    chain = rag["chain"]

    result = chain.invoke(body.question, chat_history=body.chat_history)

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
