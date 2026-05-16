from fastapi import FastAPI

from app.api.routes import chat, document

app = FastAPI(title="企业知识库助手", version="0.1.0")

app.include_router(chat.router)
app.include_router(document.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
