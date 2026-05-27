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
    DIRECT_SYSTEM_PROMPT,
    DIRECT_USER_TEMPLATE,
    NO_RESULT_MESSAGE,
    QUERY_REWRITE_PROMPT,
    QUERY_REWRITE_TEMPLATE,
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
        query_rewrite: bool = True,
        dynamic_threshold: bool = True,
        min_gap: float = 0.05,
    ):
        self._llm = llm
        self._vectorstore = vectorstore
        self._top_k = top_k
        self._retrieve_k = retrieve_k or top_k * 3
        self._score_threshold = score_threshold
        self._reranker = reranker
        self._system_prompt = system_prompt
        self._query_rewrite = query_rewrite
        self._dynamic_threshold = dynamic_threshold
        self._min_gap = min_gap

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
        # 第一步：向量检索（自动改写查询）
        results = self._retrieve(question, chat_history)

        if not results:
            # 检索无结果时，直接调用大模型回答
            logger.info("检索无结果，直接调用大模型: %s", question[:30])
            messages = self._build_direct_messages(question, chat_history)
            response = self._llm.invoke(messages)
            return RAGResult(
                answer="当前知识库中未找到相关信息，以下为大模型回答：\n\n" + response.content,
                sources=[],
            )

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
        results = self._retrieve(question, chat_history)

        if not results:
            # 检索无结果时，直接流式调用大模型
            logger.info("检索无结果，直接流式调用大模型: %s", question[:30])
            yield "当前知识库中未找到相关信息，以下为大模型回答：\n\n"
            messages = self._build_direct_messages(question, chat_history)
            for chunk in self._llm.stream(messages):
                if chunk.content:
                    yield chunk.content
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
        results = self._retrieve(question, chat_history)

        if not results:
            # 检索无结果时，直接流式调用大模型
            logger.info("检索无结果，直接流式事件调用大模型: %s", question[:30])
            yield json.dumps(
                {"event": "content", "data": "当前知识库中未找到相关信息，以下为大模型回答：\n\n"},
                ensure_ascii=False,
            )
            messages = self._build_direct_messages(question, chat_history)
            for chunk in self._llm.stream(messages):
                reasoning = (
                    chunk.additional_kwargs.get("reasoning_content")
                    or chunk.additional_kwargs.get("thinking")
                )
                if reasoning:
                    yield json.dumps({"event": "thinking", "data": reasoning}, ensure_ascii=False)
                if chunk.content:
                    yield json.dumps({"event": "content", "data": chunk.content}, ensure_ascii=False)
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
        results = self._retrieve(question, chat_history=None)
        return self._format_sources(results)

    # ================================================================
    # 内部方法
    # ================================================================

    def _retrieve(
        self, question: str, chat_history: list[dict] | None = None
    ) -> list[SearchResult]:
        """向量检索流程。

        步骤：
            0. (可选) 查询改写 —— 用 LLM 将问题优化为更适合检索的查询
            1. 多召回候选 —— 调用 vectorstore.similarity_search() 获取 retrieve_k 个候选
            2. 阈值过滤  —— 动态断层检测 或 静态阈值过滤，至少保留 top_k 个
            3. 重排序     —— (可选) 用 reranker 精排，或直接截断 top_k

        Args:
            question: 用户问题。
            chat_history: 可选的对话历史（用于查询改写）。

        Returns:
            list[SearchResult]: 最终的检索结果列表（最多 top_k 个）。
        """
        # 0. 查询改写：有对话历史时，用 LLM 将指代词还原为具体名词
        search_query = question
        if self._query_rewrite and chat_history:
            search_query = self._rewrite_query(question, chat_history)
            logger.info("查询改写: %s → %s", question[:30], search_query[:50])

        # 1. 粗召回：获取 retrieve_k 个候选（默认 top_k * 3）
        candidates = self._vectorstore.similarity_search(
            search_query, top_k=self._retrieve_k
        )

        # 2. 阈值过滤
        if candidates:
            if self._dynamic_threshold:
                candidates = self._filter_by_score_gap(candidates)
            elif self._score_threshold is not None:
                candidates = [
                    r for r in candidates
                    if r.score is not None and r.score >= self._score_threshold
                ]

        # 3. 精排：有 reranker 时用 reranker 精排，否则直接截取 top_k
        if self._reranker and candidates:
            results = self._reranker.rerank(search_query, candidates, top_k=self._top_k)
        else:
            results = candidates[: self._top_k]

        logger.info(
            "检索完成: query=%s, candidates=%d, final=%d, scores=%s",
            search_query[:30],
            len(candidates),
            len(results),
            [round(r.score, 4) for r in results] if results else [],
        )
        return results

    def _filter_by_score_gap(self, results: list[SearchResult]) -> list[SearchResult]:
        """动态阈值：分数断层检测 + top_k 保底。

        算法：
            1. 按分数降序排列
            2. 找相邻分数的最大断层位置
            3. 如果断层 > min_gap 且断层前的结果数 >= top_k，在断层处截断
            4. 否则保留 top_k 个

        Args:
            results: 已排序（或未排序）的检索结果。

        Returns:
            list[SearchResult]: 过滤后的结果，至少 top_k 个。
        """
        if len(results) <= self._top_k:
            return results

        sorted_results = sorted(results, key=lambda r: r.score or 0, reverse=True)
        scores = [r.score or 0 for r in sorted_results]

        # 找所有相邻位置的最大断层
        max_gap = 0.0
        gap_idx = 0
        for i in range(len(scores) - 1):
            gap = scores[i] - scores[i + 1]
            if gap > max_gap:
                max_gap = gap
                gap_idx = i + 1

        # 断层显著 且 截断后仍有 >= top_k 个结果 → 在断层处截断
        if max_gap >= self._min_gap and gap_idx >= self._top_k - 1:
            filtered = sorted_results[:gap_idx]
            logger.info(
                "动态阈值: 断层=%.4f at idx=%d, 截断 %d→%d",
                max_gap, gap_idx, len(sorted_results), len(filtered),
            )
            return filtered

        return sorted_results[: self._top_k]

    def _rewrite_query(self, question: str, chat_history: list[dict]) -> str:
        """用 LLM 改写查询，补全省略信息，使其更适合向量检索。

        Args:
            question: 用户原始问题。
            chat_history: 对话历史。

        Returns:
            str: 改写后的查询文本。
        """
        # 只取最近 3 轮对话作为改写上下文
        recent = chat_history[-6:] if len(chat_history) > 6 else chat_history
        history_lines = []
        for turn in recent:
            role = "用户" if turn.get("role") == "user" else "助手"
            history_lines.append(f"{role}: {turn.get('content', '')}")
        history_text = "\n".join(history_lines) if history_lines else "（无对话历史）"

        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(
                content=QUERY_REWRITE_TEMPLATE.format(
                    history=history_text, question=question
                )
            ),
        ]

        try:
            response = self._llm.invoke(messages)
            rewritten = response.content.strip()
            # 如果改写结果为空或太短，回退到原问题
            if not rewritten or len(rewritten) < 2:
                return question
            return rewritten
        except Exception as e:
            logger.warning("查询改写失败，使用原问题: %s", e)
            return question

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

    def _build_direct_messages(
        self,
        question: str,
        chat_history: list[dict] | None,
    ) -> list:
        """知识库无结果时，组装直接询问大模型的消息列表（不带 RAG 上下文）。

        Args:
            question: 用户问题。
            chat_history: 可选的对话历史。

        Returns:
            list: LangChain 消息列表。
        """
        messages = [SystemMessage(content=DIRECT_SYSTEM_PROMPT)]

        if chat_history:
            for turn in chat_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(SystemMessage(content=content))

        messages.append(HumanMessage(content=DIRECT_USER_TEMPLATE.format(question=question)))
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
