import logging

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def create_local_embedding(
    model_name: str = "BAAI/bge-small-zh-v1.5",
) -> HuggingFaceEmbeddings:
    logger.info("创建本地 Embedding: model=%s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
