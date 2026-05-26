"""
RAG 推理链核心实现 —— RAG 在线阶段的核心。

职责：串联向量检索 → (可选 rerank) → 拼接 Prompt → 调用 LLM → 返回结果。

RAG 调用流程（invoke 方法）：
    1. _retrieve(question)     — 向量检索，获取与问题相关的文档片段
    2. _build_context(results) — 将检索结果格式化为上下文文本
    3. _build_messages()       — 组装 system prompt + 对话历史 + 用户问题
    4. llm.invoke(messages)    — 调用大模型生成回答
    5. 返回 RAGResult(answer, sources)

调用关系：
    chat.py 中 chat_endpoint()
      └─> chain.invoke(question, chat_history)
           ├─ _retrieve()       → list[SearchResult]  # 从向量库检索
           ├─ _build_context()  → str                 # 拼接上下文
           ├─ _build_messages() → list[Message]       # 组装 Prompt
           └─ llm.invoke()      → RAGResult           # 调用 LLM
"""
import json
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
    """引用来源 —— 附加在回答后的参考文献信息。

    Attributes:
        content: 文档片段的文本内容。
        source: 来源文件名。
        score: 相似度分数。
        metadata: 完整的元信息字典。
    """

    content: str
    source: str
    score: float | None = None
    metadata: dict = Field(default_factory=dict)


class RAGResult(BaseModel):
    """RAG 调用结果 —— 包含回答文本和引用来源。

    Attributes:
        answer: LLM 生成的回答文本。
        sources: 回答所依据的引用来源列表。
    """

    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)


class RAGChain:
    """RAG 链：检索 → (可选 rerank) → 拼接 Prompt → 调用 LLM。

    三种调用模式：
        invoke()       — 完整调用，返回 RAGResult（answer + sources）
        stream()       — 流式调用，逐 token yield
        search_only()  — 仅检索，不调 LLM（调试用）

    检索流程 (_retrieve)：
        1. 向量检索 retrieve_k 个候选（粗召回）
        2. score_threshold 过滤低质量结果
        3. (可选) reranker 重排序
        4. 取 top_k 个最终结果

    Attributes:
        _llm: 大模型实例。
        _vectorstore: 向量库管理器。
        _top_k: 最终返回的检索结果数量。
        _retrieve_k: 初始召回数量（粗召回，通常 > top_k）。
        _score_threshold: 相似度阈值（可选）。
        _reranker: 重排序器（可选）。
        _system_prompt: 系统提示词。
    """

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
        # retrieve_k 默认为 top_k 的 3 倍，给 reranker 足够的候选空间
        self._retrieve_k = retrieve_k or top_k * 3
        self._score_threshold = score_threshold
        self._reranker = reranker
        self._system_prompt = system_prompt

    # ================================================================
    # 对外接口
    # ================================================================

    def invoke(self, question: str, chat_history: list[dict] | None = None) -> RAGResult:
        """执行一次完整的 RAG 调用，返回回答和引用来源。

        流程：检索 → 拼接上下文 → 组装 Prompt → 调用 LLM → 返回结果。

        Args:
            question: 用户问题。
            chat_history: 可选的对话历史，格式 [{"role": "user/assistant", "content": "..."}]。

        Returns:
            RAGResult: 包含 answer（回答文本）和 sources（引用来源列表）。
        """
        # 第一步：向量检索
        results = self._retrieve(question)

        if not results:
            # 检索无结果时返回兜底回答
            logger.info("检索无结果，返回兜底回答: %s", question[:30])
            return RAGResult(answer=NO_RESULT_MESSAGE, sources=[])

        # 第二步：将检索结果格式化为上下文文本
        context = self._build_context(results)

        # 第三步：组装发送给 LLM 的消息列表
        messages = self._build_messages(question, context, chat_history)

        # 第四步：调用大模型生成回答
        logger.info("调用 LLM: question=%s, context_chunks=%d", question[:30], len(results))
        response = self._llm.invoke(messages)

        return RAGResult(
            answer=response.content,
            sources=self._format_sources(results),
        )

    def stream(self, question: str, chat_history: list[dict] | None = None) -> Iterator[str]:
        """流式返回回答，逐 token yield。

        与 invoke() 的区别：LLM 返回的是流式迭代器，每个 chunk 是一个 token。
        适用于需要实时展示回答的前端场景。

        Args:
            question: 用户问题。
            chat_history: 可选的对话历史。

        Yields:
            str: 回答的逐个 token 片段。
        """
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

    def stream_events(self, question: str, chat_history: list[dict] | None = None) -> Iterator[str]:
        """流式返回结构化事件（SSE 格式），包含检索来源、思考过程和回答内容。

        Yields 的每个字符串是 JSON 格式的 SSE 事件：
            - {"event": "sources", "data": [...]}     — 检索到的引用来源
            - {"event": "thinking", "data": "..."}     — 模型思考过程（若模型支持）
            - {"event": "content", "data": "..."}      — 回答的逐 token 内容
            - {"event": "done", "data": ""}            — 流结束标记

        Args:
            question: 用户问题。
            chat_history: 可选的对话历史。

        Yields:
            str: JSON 格式的事件字符串。
        """
        results = self._retrieve(question)

        if not results:
            yield json.dumps({"event": "content", "data": NO_RESULT_MESSAGE}, ensure_ascii=False)
            yield json.dumps({"event": "done", "data": ""}, ensure_ascii=False)
            return

        # 先推送检索来源
        sources = self._format_sources(results)
        sources_data = [
            {"content": s.content, "source": s.source, "score": s.score}
            for s in sources
        ]
        yield json.dumps({"event": "sources", "data": sources_data}, ensure_ascii=False)

        context = self._build_context(results)
        messages = self._build_messages(question, context, chat_history)

        logger.info("流式事件调用 LLM: question=%s", question[:30])
        for chunk in self._llm.stream(messages):
            # 提取思考过程（DeepSeek / MiMo 等模型支持）
            reasoning = (
                chunk.additional_kwargs.get("reasoning_content")
                or chunk.additional_kwargs.get("thinking")
            )
            if reasoning:
                yield json.dumps({"event": "thinking", "data": reasoning}, ensure_ascii=False)

            # 提取回答内容
            if chunk.content:
                yield json.dumps({"event": "content", "data": chunk.content}, ensure_ascii=False)

        yield json.dumps({"event": "done", "data": ""}, ensure_ascii=False)

    def search_only(self, question: str) -> list[SourceDocument]:
        """仅检索，不调 LLM。用于调试或预览检索效果。

        Args:
            question: 用户问题。

        Returns:
            list[SourceDocument]: 检索到的引用来源列表。
        """
        results = self._retrieve(question)
        return self._format_sources(results)

    # ================================================================
    # 内部方法
    # ================================================================

    def _retrieve(self, question: str) -> list[SearchResult]:
        """向量检索流程。

        步骤：
            1. 多召回候选 —— 调用 vectorstore.similarity_search() 获取 retrieve_k 个候选
            2. 阈值过滤  —— 按 score_threshold 过滤低质量结果
            3. 重排序     —— (可选) 用 reranker 精排，或直接截断 top_k

        Args:
            question: 用户问题。

        Returns:
            list[SearchResult]: 最终的检索结果列表（最多 top_k 个）。
        """
        # 1. 粗召回：获取 retrieve_k 个候选（默认 top_k * 3）
        candidates = self._vectorstore.similarity_search(
            question, top_k=self._retrieve_k
        )

        # 2. 阈值过滤：丢弃相似度低于阈值的候选
        if self._score_threshold is not None and candidates:
            candidates = [
                r for r in candidates
                if r.score is not None and r.score >= self._score_threshold
            ]

        # 3. 精排：有 reranker 时用 reranker 精排，否则直接截取 top_k
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
        """将检索结果格式化为上下文字符串。

        使用 CONTEXT_ITEM 模板，为每个结果标注编号和来源，方便 LLM 引用。
        输出示例：
            [1] 来源：report.pdf
            <chunk1 的文本内容>

            [2] 来源：guide.md
            <chunk2 的文本内容>

        Args:
            results: 检索结果列表。

        Returns:
            str: 格式化后的上下文文本。
        """
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
        """组装发送给 LLM 的消息列表。

        消息顺序：
            1. SystemMessage — 系统提示词（抗幻觉规则）
            2. 历史对话（可选）— HumanMessage / SystemMessage 交替
            3. HumanMessage — 当前问题（USER_TEMPLATE 包含上下文 + 问题）

        Args:
            question: 用户问题。
            context: 格式化后的上下文文本。
            chat_history: 可选的对话历史。

        Returns:
            list: LangChain 消息列表。
        """
        messages = [SystemMessage(content=self._system_prompt)]

        # 追加对话历史（多轮对话支持）
        if chat_history:
            for turn in chat_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(SystemMessage(content=content))

        # 追加当前问题（包含检索到的上下文）
        user_content = USER_TEMPLATE.format(context=context, question=question)
        messages.append(HumanMessage(content=user_content))

        return messages

    @staticmethod
    def _format_sources(results: list[SearchResult]) -> list[SourceDocument]:
        """将 SearchResult 转换为 SourceDocument（用于 API 响应）。

        SearchResult — 内部检索数据模型
        SourceDocument — 对外 API 响应模型
        """
        return [
            SourceDocument(
                content=r.content,
                source=r.metadata.get("source", "未知"),
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]
