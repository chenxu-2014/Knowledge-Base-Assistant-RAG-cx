import logging

from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


def create_ollama_embedding(
    base_url: str = "http://localhost:11434",
    model: str = "nomic-embed-text:v1.5",
) -> OllamaEmbeddings:
    logger.info("创建 Ollama Embedding: model=%s, url=%s", model, base_url)
    return OllamaEmbeddings(
        base_url=base_url,
        model=model,
    )
