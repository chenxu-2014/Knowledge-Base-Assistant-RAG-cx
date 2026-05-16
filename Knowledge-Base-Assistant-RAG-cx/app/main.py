import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_openai import ChatOpenAI

from app.config import settings
from app.api.routes import chat, document

logger = logging.getLogger(__name__)


def _init_components():
    """初始化 RAG 组件，供路由层通过 app.state 访问"""
    from app.core.embedding import EmbeddingFactory
    from app.core.vectorstore import VectorStoreFactory
    from app.core.document import DocumentPipeline
    from app.core.chain import RAGChainFactory

    embedding = EmbeddingFactory.from_settings(settings)
    vectorstore = VectorStoreFactory.from_settings(settings, embedding)
    pipeline = DocumentPipeline(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # 直接用 ChatOpenAI（DeepSeek / MiMo 均兼容 OpenAI 协议）
    provider = settings.llm_provider
    api_key = getattr(settings, f"{provider}_api_key", "")
    base_url = getattr(settings, f"{provider}_base_url", "")
    model = getattr(settings, f"{provider}_model", "")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
        max_tokens=2048,
    )

    chain = RAGChainFactory.create(
        llm=llm,
        vectorstore=vectorstore,
        top_k=settings.search_top_k,
    )

    return {
        "embedding": embedding,
        "vectorstore": vectorstore,
        "pipeline": pipeline,
        "chain": chain,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("初始化 RAG 组件...")
    app.state.rag = _init_components()
    logger.info("RAG 组件初始化完成")
    yield
    logger.info("应用关闭")


app = FastAPI(title="企业知识库助手", version="0.1.0", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(document.router)

# 静态文件
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
