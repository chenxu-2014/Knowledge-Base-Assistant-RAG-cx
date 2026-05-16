from langchain_core.language_models import BaseChatModel

from app.core.vectorstore import VectorStoreManager


class RAGChain:
    """RAG 链：检索 → 拼接 Prompt → 调用 LLM"""

    def __init__(self, llm: BaseChatModel, vectorstore: VectorStoreManager, top_k: int = 4):
        self._llm = llm
        self._vectorstore = vectorstore
        self._top_k = top_k

    def invoke(self, question: str, chat_history: list[dict] | None = None) -> dict:
        """
        执行一次 RAG 调用。
        返回: {"answer": str, "sources": list[dict]}
        """
        raise NotImplementedError

    def stream(self, question: str, chat_history: list[dict] | None = None):
        """流式返回回答"""
        raise NotImplementedError
