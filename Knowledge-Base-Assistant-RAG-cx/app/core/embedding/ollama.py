"""
Ollama Embedding 创建函数 —— 基于 Ollama 的本地模型 Embedding。

Ollama 是本地大模型运行框架，通过 HTTP API 提供推理服务。
需要先启动 Ollama 服务（默认 http://localhost:11434）。

调用关系：
    EmbeddingFactory.create("ollama", base_url="http://localhost:11434", model="nomic-embed-text:v1.5")
      └─> create_ollama_embedding(base_url, model)
           └─> OllamaEmbeddings(...)   # 返回 Ollama Embedding 实例
"""
import logging

from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


def create_ollama_embedding(
    base_url: str = "http://localhost:11434",
    model: str = "nomic-embed-text:v1.5",
) -> OllamaEmbeddings:
    """创建 Ollama Embedding 实例。

    前置条件：Ollama 服务已启动，且已拉取对应模型（ollama pull nomic-embed-text:v1.5）。

    Args:
        base_url: Ollama 服务地址，默认 "http://localhost:11434"。
        model: 模型名称，默认 "nomic-embed-text:v1.5"。

    Returns:
        OllamaEmbeddings: 可直接传给 Chroma 的 embedding_function 参数。
    """
    logger.info("创建 Ollama Embedding: model=%s, url=%s", model, base_url)
    return OllamaEmbeddings(
        base_url=base_url,
        model=model,
    )
