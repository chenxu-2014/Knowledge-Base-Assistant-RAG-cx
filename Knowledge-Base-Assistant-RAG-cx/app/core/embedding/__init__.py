"""
Embedding 工厂模块 —— RAG 离线阶段第三步：文本嵌入。

职责：根据配置选择 Embedding 供应商，创建对应的 Embedding 实例。
      Embedding 将文本转换为高维向量，用于后续的相似度检索。

支持的 Embedding 供应商：
    "deepseek" — DeepSeek embedding API（OpenAI 兼容协议）
    "xiaomi"   — 小米 MiMo text-embedding-001（OpenAI 兼容协议）
    "local"    — 本地 HuggingFace 模型 BAAI/bge-small-zh-v1.5（免费，无需 API）
    "ollama"   — Ollama 本地模型 nomic-embed-text（免费，需运行 Ollama 服务）

调用关系：
    main._init_components()
      └─> EmbeddingFactory.from_settings(settings)   # 根据 embedding_provider 配置选择
           └─> create_deepseek_embedding() / create_xiaomi_embedding() / ...
                └─> 返回 OpenAIEmbeddings / HuggingFaceEmbeddings / OllamaEmbeddings

配置项（.env）：
    embedding_provider = deepseek | xiaomi | local | ollama
"""
import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """根据 provider 名称创建对应 Embedding 实例。

    使用方式：
        # 方式一：从配置自动创建（推荐）
        embedding = EmbeddingFactory.from_settings(settings)

        # 方式二：手动指定供应商
        embedding = EmbeddingFactory.create("deepseek", api_key="xxx", ...)
    """

    # 供应商名 → 创建函数的注册表
    _creators: dict[str, callable] = {}

    @classmethod
    def create(cls, provider: str = "deepseek", **kwargs) -> Embeddings:
        """根据供应商名创建 Embedding 实例。

        Args:
            provider: 供应商名称，如 "deepseek" / "xiaomi" / "local" / "ollama"。
            **kwargs: 传递给创建函数的参数（如 api_key, model 等）。

        Returns:
            Embeddings: LangChain Embedding 实例。

        Raises:
            ValueError: 供应商未注册。
        """
        if provider not in cls._creators:
            raise ValueError(
                f"不支持的 Embedding 供应商: '{provider}'，可选: {list(cls._creators.keys())}"
            )
        return cls._creators[provider](**kwargs)

    @classmethod
    def from_settings(cls, settings) -> Embeddings:
        """从配置对象自动创建 Embedding。

        根据 settings.embedding_provider 字段选择供应商，
        自动从 settings 中读取对应的 api_key、base_url、model 等参数。

        Args:
            settings: Settings 配置对象（app.config.settings）。

        Returns:
            Embeddings: 对应供应商的 Embedding 实例。
        """
        provider = settings.embedding_provider
        logger.info("从配置创建 Embedding, provider=%s", provider)

        if provider == "deepseek":
            return cls.create(
                "deepseek",
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.embedding_deepseek_model,
            )
        elif provider == "xiaomi":
            return cls.create(
                "xiaomi",
                api_key=settings.xiaomi_api_key,
                base_url=settings.xiaomi_base_url,
                model=settings.embedding_xiaomi_model,
            )
        elif provider == "local":
            return cls.create(
                "local",
                model_name=settings.embedding_local_model,
            )
        elif provider == "ollama":
            return cls.create(
                "ollama",
                base_url=settings.ollama_base_url,
                model=settings.embedding_ollama_model,
            )
        else:
            raise ValueError(f"未知的 Embedding 供应商: {provider}")

    @classmethod
    def register(cls, name: str, creator: callable):
        """注册自定义 Embedding 供应商。

        Args:
            name: 供应商名称。
            creator: 创建函数，签名为 (api_key, ...) -> Embeddings。
        """
        cls._creators[name] = creator

    @classmethod
    def available_providers(cls) -> list[str]:
        """返回所有已注册的供应商名称。"""
        return sorted(cls._creators.keys())


# ─── 注册内置供应商 ───

# API 类供应商（OpenAI 兼容协议）
from app.core.embedding.api import create_deepseek_embedding, create_xiaomi_embedding

EmbeddingFactory.register("deepseek", create_deepseek_embedding)
EmbeddingFactory.register("xiaomi", create_xiaomi_embedding)

# Ollama 本地模型
from app.core.embedding.ollama import create_ollama_embedding

EmbeddingFactory.register("ollama", create_ollama_embedding)

# 本地 HuggingFace 模型（可选依赖，未安装 sentence-transformers 时优雅降级）
try:
    from app.core.embedding.local import create_local_embedding

    EmbeddingFactory.register("local", create_local_embedding)
except ImportError:
    logger.warning("本地 Embedding 依赖未安装，local provider 不可用")


__all__ = ["EmbeddingFactory"]
