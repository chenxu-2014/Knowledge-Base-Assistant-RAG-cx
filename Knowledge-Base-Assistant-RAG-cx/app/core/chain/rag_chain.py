import logging
from collections.abc import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.chain.prompts import (
    CONTEXT_ITEM,
    DEFAULT_SYSTEM_PROMPT,
    NO_RESULT_MESSAGE,
    USER_TEMPLATE,
)
from app.core.reranker.base import BaseReranker
from app.core.vectorstore.manager import VectorStoreManager
from app.core.vectorstore.models import SearchResult

logger = logging.getLogger(__name__)


class SourceDocument(BaseModel):
    """引用来源"""

    content: str
    source: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RAGResult(BaseModel):
    """RAG 调用结果"""

    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)


class RAGChain:
    """RAG 链：检索 → (可选 rerank) → 拼接 Prompt → 调用 LLM"""

    def __init__(
        self,
        llm: BaseChatModel,
        vectorstore: VectorStoreManager,
        top_k: int = 4,
        retrieve_k: int | None = None,
        score_threshold: float | None = None,
        reranker: BaseReranker | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self._llm = llm
        self._vectorstore = vectorstore
        self._top_k = top_k
        self._retrieve_k = retrieve_k or top_k * 3
        self._score_threshold = score_threshold
        self._reranker = reranker
        self._system_prompt = system_prompt

    # ================================================================
    # 对外接口
    # ================================================================

    def invoke(self, question: str, chat_history: list[dict] | None = None) -> RAGResult:
        """执行一次完整的 RAG 调用，返回回答和引用来源"""
        results = self._retrieve(question)

        if not results:
            logger.info("检索无结果，返回兜底回答: %s", question[:30])
            return RAGResult(answer=NO_RESULT_MESSAGE, sources=[])

        context = self._build_context(results)
        messages = self._build_messages(question, context, chat_history)

        logger.info("调用 LLM: question=%s, context_chunks=%d", question[:30], len(results))
        response = self._llm.invoke(messages)

        return RAGResult(
            answer=response.content,
            sources=self._format_sources(results),
        )

    def stream(self, question: str, chat_history: list[dict] | None = None) -> Iterator[str]:
        """流式返回回答，逐 token yield"""
        results = self._retrieve(question)

        if not results:
            yield NO_RESULT_MESSAGE
            return

        context = self._build_context(results)
        messages = self._build_messages(question, context, chat_history)

        logger.info("流式调用 LLM: question=%s", question[:30])
        for chunk in self._llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def search_only(self, question: str) -> list[SourceDocument]:
        """仅检索，不调 LLM。用于调试或预览"""
        results = self._retrieve(question)
        return self._format_sources(results)

    # ================================================================
    # 内部方法
    # ================================================================

    def _retrieve(self, question: str) -> list[SearchResult]:
        """
        检索流程:
        1. 向量检索 retrieve_k 个候选
        2. score_threshold 过滤
        3. (可选) reranker 重排序
        4. 取 top_k
        """
        # 1. 多召回候选
        candidates = self._vectorstore.similarity_search(
            question, top_k=self._retrieve_k
        )

        # 2. 阈值过滤
        if self._score_threshold is not None and candidates:
            candidates = [
                r for r in candidates
                if r.score is not None and r.score >= self._score_threshold
            ]

        # 3. rerank 或直接截断
        if self._reranker and candidates:
            results = self._reranker.rerank(question, candidates, top_k=self._top_k)
        else:
            results = candidates[: self._top_k]

        logger.info(
            "检索完成: query=%s, candidates=%d, final=%d",
            question[:30],
            len(candidates),
            len(results),
        )
        return results

    @staticmethod
    def _build_context(results: list[SearchResult]) -> str:
        """将检索结果格式化为上下文字符串"""
        parts = []
        for i, r in enumerate(results, start=1):
            source = r.metadata.get("source", "未知")
            parts.append(CONTEXT_ITEM.format(index=i, source=source, content=r.content))
        return "\n\n".join(parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: list[dict] | None,
    ) -> list:
        """组装发送给 LLM 的消息列表"""
        messages = [SystemMessage(content=self._system_prompt)]

        # 多轮对话历史
        if chat_history:
            for turn in chat_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(SystemMessage(content=content))

        user_content = USER_TEMPLATE.format(context=context, question=question)
        messages.append(HumanMessage(content=user_content))

        return messages

    @staticmethod
    def _format_sources(results: list[SearchResult]) -> list[SourceDocument]:
        """将 SearchResult 转换为 SourceDocument"""
        return [
            SourceDocument(
                content=r.content,
                source=r.metadata.get("source", "未知"),
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]
