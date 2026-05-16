import logging

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def create_deepseek_embedding(
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-embedding",
) -> OpenAIEmbeddings:
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
    logger.info("创建 Xiaomi Embedding: model=%s, url=%s", model, base_url)
    return OpenAIEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
