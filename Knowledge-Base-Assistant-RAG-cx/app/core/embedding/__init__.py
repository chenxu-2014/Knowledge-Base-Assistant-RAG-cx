import logging

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """根据 provider 名称创建对应 Embedding 实例"""

    _creators: dict[str, callable] = {}

    @classmethod
    def create(cls, provider: str = "deepseek", **kwargs) -> Embeddings:
        if provider not in cls._creators:
            raise ValueError(
                f"不支持的 Embedding 供应商: '{provider}'，可选: {list(cls._creators.keys())}"
            )
        return cls._creators[provider](**kwargs)

    @classmethod
    def from_settings(cls, settings) -> Embeddings:
        """从配置对象自动创建，根据 embedding_provider 字段切换"""
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
        """注册自定义供应商"""
        cls._creators[name] = creator

    @classmethod
    def available_providers(cls) -> list[str]:
        return sorted(cls._creators.keys())


# 注册内置供应商
from app.core.embedding.api import create_deepseek_embedding, create_xiaomi_embedding

EmbeddingFactory.register("deepseek", create_deepseek_embedding)
EmbeddingFactory.register("xiaomi", create_xiaomi_embedding)

from app.core.embedding.ollama import create_ollama_embedding

EmbeddingFactory.register("ollama", create_ollama_embedding)

try:
    from app.core.embedding.local import create_local_embedding

    EmbeddingFactory.register("local", create_local_embedding)
except ImportError:
    logger.warning("本地 Embedding 依赖未安装，local provider 不可用")


__all__ = ["EmbeddingFactory"]
