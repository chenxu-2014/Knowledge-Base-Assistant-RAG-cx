"""
本地 Embedding 创建函数 —— 基于 HuggingFace 的免费本地 Embedding。

使用 sentence-transformers 库加载本地模型，无需 API Key。
默认模型 BAAI/bge-small-zh-v1.5 是针对中文优化的小型嵌入模型（~90MB）。

优点：免费、无网络依赖、数据不出本地
缺点：首次加载慢（需下载模型）、embedding 质量可能略低于大模型 API

调用关系：
    EmbeddingFactory.create("local", model_name="BAAI/bge-small-zh-v1.5")
      └─> create_local_embedding(model_name)
           └─> HuggingFaceEmbeddings(...)   # 返回本地 Embedding 实例
"""
import logging

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def create_local_embedding(
    model_name: str = "BAAI/bge-small-zh-v1.5",
) -> HuggingFaceEmbeddings:
    """创建本地 HuggingFace Embedding 实例。

    依赖 sentence-transformers 包，未安装时 Import会被 EmbeddingFactory 捕获降级。

    Args:
        model_name: HuggingFace 模型名称或本地路径。
                    默认 "BAAI/bge-small-zh-v1.5"（中文优化的小型模型）。

    Returns:
        HuggingFaceEmbeddings: 本地 Embedding 实例。
    """
    logger.info("创建本地 Embedding: model=%s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},  # 归一化向量，提升余弦相似度精度
    )
