"""
RAG Chain 模块 —— RAG 在线阶段的核心编排。

职责：将向量检索、上下文拼接、LLM 调用串联为完整的 RAG 推理链。

核心组件：
    RAGChainFactory — 工厂，从配置创建 RAGChain
    RAGChain        — RAG 推理链核心类（检索 → rerank → 拼接 Prompt → 调用 LLM）
    RAGResult       — RAG 调用结果（answer + sources）
    SourceDocument  — 引用来源文档

调用关系：
    main._init_components()
      └─> RAGChainFactory.create(llm, vectorstore, top_k)
           └─> RAGChain(llm, vectorstore, ...)

    chat.py 中 chat_endpoint()
      └─> chain.invoke(question, chat_history)
           └─> RAGChain.invoke() → RAGResult
"""
import logging

from langchain_core.language_models import BaseChatModel

from app.core.chain.rag_chain import RAGChain, RAGResult, SourceDocument
from app.core.chain.prompts import DEFAULT_SYSTEM_PROMPT
from app.core.reranker.base import BaseReranker
from app.core.vectorstore.manager import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGChainFactory:
    """RAG Chain 工厂 —— 从配置创建 RAGChain 实例。

    使用方式：
        chain = RAGChainFactory.create(llm=llm, vectorstore=vectorstore, top_k=4)
        result = chain.invoke("什么是 RAG？")
    """

    @staticmethod
    def create(
        llm: BaseChatModel,
        vectorstore: VectorStoreManager,
        top_k: int = 4,
        retrieve_k: int | None = None,
        score_threshold: float | None = None,
        reranker: BaseReranker | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        query_rewrite: bool = True,
        dynamic_threshold: bool = True,
        min_gap: float = 0.05,
    ) -> RAGChain:
        """创建 RAGChain 实例。

        Args:
            llm: 大模型实例（ChatOpenAI / ChatOllama 等）。
            vectorstore: 向量库管理器。
            top_k: 最终返回的检索结果数量。
            retrieve_k: 初始召回数量（默认 top_k * 3，用于 reranker 精排前的粗召回）。
            score_threshold: 静态相似度阈值（dynamic_threshold=False 时生效）。
            reranker: 可选的重排序器，对粗召回结果精排。
            system_prompt: 系统提示词。
            query_rewrite: 是否启用查询改写。
            dynamic_threshold: 是否启用动态阈值（分数断层检测）。
            min_gap: 动态阈值的最小断层距离。

        Returns:
            RAGChain: RAG 推理链实例。
        """
        return RAGChain(
            llm=llm,
            vectorstore=vectorstore,
            top_k=top_k,
            retrieve_k=retrieve_k,
            score_threshold=score_threshold,
            reranker=reranker,
            system_prompt=system_prompt,
            query_rewrite=query_rewrite,
            dynamic_threshold=dynamic_threshold,
            min_gap=min_gap,
        )


__all__ = [
    "RAGChain",
    "RAGChainFactory",
    "RAGResult",
    "SourceDocument",
]
