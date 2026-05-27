"""
API 类 Embedding 创建函数 —— 基于 OpenAI 兼容协议的远程 Embedding。

DeepSeek 和小米 MiMo 的 Embedding API 都兼容 OpenAI 协议，
所以统一使用 langchain_openai 的 OpenAIEmbeddings 类。

调用关系：
    EmbeddingFactory.create("deepseek", api_key=..., base_url=..., model=...)
      └─> create_deepseek_embedding(api_key, base_url, model)
           └─> OpenAIEmbeddings(...)   # 返回 LangChain OpenAI Embedding 实例
"""
import logging

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def create_deepseek_embedding(
    api_key: str,
    base_url: str = "https://api.deepseek.com/v1",
    model: str = "deepseek-embedding",
) -> OpenAIEmbeddings:
    """创建 DeepSeek Embedding 实例。

    DeepSeek Embedding API 兼容 OpenAI 协议，直接使用 OpenAIEmbeddings 类。

    Args:
        api_key: DeepSeek API Key。
        base_url: API 地址，默认 "https://api.deepseek.com"。
        model: 模型名称，默认 "deepseek-embedding"。

    Returns:
        OpenAIEmbeddings: 可直接传给 Chroma 的 embedding_function 参数。
    """
    logger.info("创建 DeepSeek Embedding: model=%s, url=%s", model, base_url)
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def create_xiaomi_embedding(
    api_key: str,
    base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
    model: str = "text-embedding-001",
) -> OpenAIEmbeddings:
    """创建小米 MiMo Embedding 实例。

    小米 MiMo 的 text-embedding-001 API 兼容 OpenAI 协议。

    Args:
        api_key: 小米 API Key。
        base_url: API 地址。
        model: 模型名称，默认 "text-embedding-001"。

    Returns:
        OpenAIEmbeddings: 可直接传给 Chroma 的 embedding_function 参数。
    """
    logger.info("创建 Xiaomi Embedding: model=%s, url=%s", model, base_url)
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
