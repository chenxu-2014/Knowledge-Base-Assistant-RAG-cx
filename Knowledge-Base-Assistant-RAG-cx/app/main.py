"""
FastAPI 应用入口 —— RAG 知识库问答系统的启动文件。

职责：
    1. 初始化所有 RAG 组件（Embedding、VectorStore、Pipeline、LLM、Chain）
    2. 通过 FastAPI lifespan 管理组件生命周期
    3. 注册 API 路由（chat、document）
    4. 提供静态文件服务

启动流程：
    uvicorn app.main:app --reload
    ↓
    lifespan() → _init_components() → app.state.rag = {...}
    ↓
    等待请求 → POST /api/chat → chat_endpoint() → chain.invoke()
"""
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
    """初始化所有 RAG 组件，供路由层通过 app.state.rag 访问。

    初始化顺序（有依赖关系，不可随意调换）：
        1. Embedding — 将文本转换为向量，无依赖
        2. VectorStore — 依赖 Embedding，用于向量化查询和文档
        3. DocumentPipeline — 文档加载和分块，无依赖
        4. LLM — 大模型实例，无依赖
        5. RAGChain — 依赖 LLM + VectorStore，串联检索和生成

    返回的字典挂载到 app.state.rag，路由通过 req.app.state.rag["chain"] 等访问。
    """
    from app.core.embedding import EmbeddingFactory
    from app.core.vectorstore import VectorStoreFactory
    from app.core.document import DocumentPipeline
    from app.core.chain import RAGChainFactory

    # 第一步：创建 Embedding 实例（根据 settings.embedding_provider 选择供应商）
    embedding = EmbeddingFactory.from_settings(settings)

    # 第二步：创建向量库管理器（传入 embedding，Chroma 内部用它向量化文档和查询）
    vectorstore = VectorStoreFactory.from_settings(settings, embedding)

    # 第三步：创建文档处理管线（加载 + 分块）
    pipeline = DocumentPipeline(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # 第四步：创建 LLM 实例
    # Ollama 使用专门的 ChatOllama 类，其他供应商使用 ChatOpenAI（OpenAI 兼容协议）
    provider = settings.llm_provider
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,     # 低温度，回答更确定性
            num_predict=2048,    # 最大生成 token 数
        )
    else:
        # DeepSeek / Xiaomi MiMo 均兼容 OpenAI 协议
        api_key = getattr(settings, f"{provider}_api_key", "")
        base_url = getattr(settings, f"{provider}_base_url", "")
        model = getattr(settings, f"{provider}_model", "")
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1,   # 低温度，回答更确定性
            max_tokens=2048,   # 最大生成 token 数
        )

    # 第五步：创建 RAG 链（串联检索 + LLM）
    chain = RAGChainFactory.create(
        llm=llm,
        vectorstore=vectorstore,
        top_k=settings.search_top_k,
    )

    # 返回所有组件，挂载到 app.state.rag
    return {
        "embedding": embedding,      # Embedding 实例（用于调试）
        "vectorstore": vectorstore,  # 向量库管理器（文档上传/删除时使用）
        "pipeline": pipeline,        # 文档处理管线（文档上传时使用）
        "chain": chain,              # RAG 链（聊天时使用）
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理器。

    启动时：初始化所有 RAG 组件，挂载到 app.state.rag
    关闭时：清理资源（当前无特殊清理逻辑）
    """
    logger.info("初始化 RAG 组件...")
    app.state.rag = _init_components()
    logger.info("RAG 组件初始化完成")
    yield
    logger.info("应用关闭")


app = FastAPI(title="企业知识库助手", version="0.1.0", lifespan=lifespan)

# 注册 API 路由
app.include_router(chat.router)      # POST /api/chat — 知识库问答
app.include_router(document.router)  # POST/GET/DELETE /api/documents/* — 文档管理

# 静态文件服务（前端页面）
STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/health")
async def health():
    """健康检查接口。"""
    return {"status": "ok"}


@app.get("/")
async def index():
    """返回前端页面。"""
    return FileResponse(STATIC_DIR / "index.html")


# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
