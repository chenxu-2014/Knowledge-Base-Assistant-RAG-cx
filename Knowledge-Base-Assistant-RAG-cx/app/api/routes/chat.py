from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """知识库问答接口"""
    print("nihao chat")
    print(ChatRequest.question)
    raise NotImplementedError
