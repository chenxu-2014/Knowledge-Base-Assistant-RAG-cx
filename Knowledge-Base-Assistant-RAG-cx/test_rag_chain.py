"""
RAG Chain 测试脚本。
使用 mock embedding + mock LLM，不依赖外部 API。
"""

import hashlib
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test")

TEST_DIR = Path(__file__).parent / "test_chroma_db"


# ================================================================
# Mock 组件
# ================================================================

class MockEmbedding(Embeddings):
    def __init__(self, dim: int = 64):
        self.dim = dim

    def _text_to_vector(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [h[i % len(h)] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._text_to_vector(text)


class MockLLM(BaseChatModel):
    """Mock LLM，返回固定的回答"""

    fixed_answer: str = "这是 Mock LLM 的回答。"

    def _generate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
        msg = AIMessage(content=self.fixed_answer)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "mock"


class MockStreamLLM(BaseChatModel):
    """Mock 流式 LLM，逐字返回"""

    fixed_answer: str = "流式回答测试。"

    def _generate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
        msg = AIMessage(content=self.fixed_answer)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def stream(self, messages: list[BaseMessage], **kwargs: Any):
        for char in self.fixed_answer:
            yield AIMessage(content=char)

    @property
    def _llm_type(self) -> str:
        return "mock_stream"


# ================================================================
# 测试辅助
# ================================================================

def make_docs(texts: list[str], source: str) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": source, "file_type": ".txt"})
        for t in texts
    ]


def setup() -> tuple:
    """创建 mock 组件 + RAGChain"""
    from app.core.vectorstore import VectorStoreManager
    from app.core.chain import RAGChainFactory

    vectorstore = VectorStoreManager(
        embedding=MockEmbedding(),
        persist_dir=str(TEST_DIR),
        collection_name="test_rag",
    )

    # 写入测试数据
    docs = make_docs(
        [
            "公司成立于2020年，专注于人工智能技术研发。",
            "员工手册第一章：工作时间为每周一至周五，上午9点到下午6点。",
            "考勤制度：迟到超过30分钟视为旷工半天。",
            "年假制度：满1年可享受5天年假，满3年可享受10天。",
        ],
        source="员工手册.docx",
    )
    vectorstore.upsert_documents("员工手册.docx", docs)

    policy_docs = make_docs(
        [
            "报销制度：差旅费需提前申请，标准为每日500元。",
            "IT设备：入职时发放笔记本电脑，离职时归还。",
        ],
        source="行政制度.txt",
    )
    vectorstore.upsert_documents("行政制度.txt", policy_docs)

    return vectorstore


# ================================================================
# 测试用例
# ================================================================

def test_invoke(vectorstore):
    """测试完整 RAG 调用"""
    from app.core.chain import RAGChainFactory

    logger.info("=" * 50)
    logger.info("测试: invoke")

    llm = MockLLM(fixed_answer="根据员工手册，公司成立于2020年。")
    chain = RAGChainFactory.create(llm=llm, vectorstore=vectorstore, top_k=3)

    result = chain.invoke("公司什么时候成立的？")

    assert result.answer == "根据员工手册，公司成立于2020年。"
    assert len(result.sources) > 0
    assert any(s.source == "员工手册.docx" for s in result.sources)

    logger.info("回答: %s", result.answer)
    logger.info("引用来源: %s", [(s.source, s.score) for s in result.sources])


def test_invoke_with_history(vectorstore):
    """测试带对话历史的调用"""
    from app.core.chain import RAGChainFactory

    logger.info("=" * 50)
    logger.info("测试: invoke with chat_history")

    llm = MockLLM(fixed_answer="年假为5天。")
    chain = RAGChainFactory.create(llm=llm, vectorstore=vectorstore, top_k=3)

    history = [
        {"role": "user", "content": "公司的考勤制度是什么？"},
        {"role": "assistant", "content": "工作时间为周一至周五9:00-18:00。"},
    ]
    result = chain.invoke("那年假呢？", chat_history=history)

    assert result.answer == "年假为5天。"
    assert len(result.sources) > 0
    logger.info("多轮调用成功: %s", result.answer)


def test_invoke_no_result(vectorstore):
    """测试检索无结果的兜底（threshold 过滤所有结果）"""
    from app.core.chain import RAGChainFactory

    logger.info("=" * 50)
    logger.info("测试: invoke 无结果兜底")

    llm = MockLLM()
    chain = RAGChainFactory.create(
        llm=llm, vectorstore=vectorstore, top_k=3, score_threshold=999
    )

    result = chain.invoke("任何问题都会被 threshold 过滤掉")

    assert "无法回答" in result.answer
    assert len(result.sources) == 0
    logger.info("兜底回答: %s", result.answer)


def test_stream(vectorstore):
    """测试流式输出"""
    from app.core.chain import RAGChainFactory

    logger.info("=" * 50)
    logger.info("测试: stream")

    llm = MockStreamLLM(fixed_answer="流式测试完成。")
    chain = RAGChainFactory.create(llm=llm, vectorstore=vectorstore, top_k=2)

    tokens = list(chain.stream("公司的报销制度是什么？"))

    full_answer = "".join(tokens)
    assert full_answer == "流式测试完成。"
    assert len(tokens) == len("流式测试完成。")
    logger.info("流式输出: %d 个 token, 内容: %s", len(tokens), full_answer)


def test_stream_no_result(vectorstore):
    """测试流式无结果"""
    from app.core.chain import RAGChainFactory

    logger.info("=" * 50)
    logger.info("测试: stream 无结果")

    llm = MockStreamLLM()
    chain = RAGChainFactory.create(
        llm=llm, vectorstore=vectorstore, top_k=2, score_threshold=999
    )

    tokens = list(chain.stream("任何问题"))

    assert len(tokens) == 1
    assert "无法回答" in tokens[0]
    logger.info("流式兜底: %s", tokens[0])


def test_search_only(vectorstore):
    """测试仅检索"""
    from app.core.chain import RAGChainFactory

    logger.info("=" * 50)
    logger.info("测试: search_only")

    llm = MockLLM()
    chain = RAGChainFactory.create(llm=llm, vectorstore=vectorstore, top_k=2)

    sources = chain.search_only("公司的报销制度")

    assert len(sources) > 0
    for s in sources:
        assert s.source in ["员工手册.docx", "行政制度.txt"]
        assert s.score is not None
    logger.info("检索结果: %s", [(s.source, s.score) for s in sources])


def test_custom_system_prompt(vectorstore):
    """测试自定义系统提示词"""
    from app.core.chain import RAGChain

    logger.info("=" * 50)
    logger.info("测试: 自定义 system_prompt")

    custom_prompt = "你是财务助手，只回答财务相关问题。"
    llm = MockLLM(fixed_answer="财务回答。")
    chain = RAGChain(
        llm=llm, vectorstore=vectorstore, top_k=2, system_prompt=custom_prompt
    )

    result = chain.invoke("报销标准是多少？")
    assert result.answer == "财务回答。"
    logger.info("自定义 prompt 成功: %s", result.answer)


def test_build_context():
    """测试上下文格式化"""
    from app.core.chain.rag_chain import RAGChain
    from app.core.vectorstore.models import SearchResult

    logger.info("=" * 50)
    logger.info("测试: _build_context")

    results = [
        SearchResult(content="内容A", metadata={"source": "文件A.txt"}, score=0.9),
        SearchResult(content="内容B", metadata={"source": "文件B.txt"}, score=0.8),
    ]
    context = RAGChain._build_context(results)

    assert "[1] 来源：文件A.txt" in context
    assert "内容A" in context
    assert "[2] 来源：文件B.txt" in context
    assert "内容B" in context
    logger.info("上下文格式化正确:\n%s", context)


def test_prompt_anti_hallucination():
    """验证 prompt 中包含抗幻觉规则"""
    from app.core.chain.prompts import DEFAULT_SYSTEM_PROMPT, USER_TEMPLATE, NO_RESULT_MESSAGE

    logger.info("=" * 50)
    logger.info("测试: prompt 抗幻觉规则")

    # 系统提示词应包含关键约束
    assert "禁止编造" in DEFAULT_SYSTEM_PROMPT
    assert "无法回答" in DEFAULT_SYSTEM_PROMPT
    assert "相关性判断" in DEFAULT_SYSTEM_PROMPT
    assert "忠于原文" in DEFAULT_SYSTEM_PROMPT

    # 用户模板应包含引导无答案的语句
    assert "没有相关信息" in USER_TEMPLATE
    assert "【参考内容】" in USER_TEMPLATE
    assert "【用户问题】" in USER_TEMPLATE

    # 兜底消息应明确"无信息"
    assert "无法回答" in NO_RESULT_MESSAGE

    logger.info("抗幻觉规则验证通过")


def test_reranker_integration(vectorstore):
    """测试 reranker 集成（mock reranker）"""
    from app.core.chain import RAGChain
    from app.core.reranker.base import BaseReranker
    from app.core.vectorstore.models import SearchResult

    logger.info("=" * 50)
    logger.info("测试: reranker 集成")

    class MockReranker(BaseReranker):
        """反转分数的 mock reranker，用于验证 rerank 被调用"""

        def rerank(self, query, results, top_k):
            # 按内容长度重排序（越短越相关，模拟）
            sorted_results = sorted(results, key=lambda r: len(r.content))
            return [
                SearchResult(content=r.content, metadata=r.metadata, score=0.99)
                for r in sorted_results[:top_k]
            ]

    llm = MockLLM(fixed_answer="rerank 回答。")
    chain = RAGChain(
        llm=llm,
        vectorstore=vectorstore,
        top_k=2,
        retrieve_k=5,
        reranker=MockReranker(),
    )

    result = chain.invoke("公司的报销制度")

    assert result.answer == "rerank 回答。"
    assert len(result.sources) == 2
    # MockReranker 把 score 设为 0.99
    assert all(s.score == 0.99 for s in result.sources)
    # 应该是最短的两条（MockReranker 按长度排序）
    lengths = [len(s.content) for s in result.sources]
    assert lengths == sorted(lengths)
    logger.info("reranker 集成通过: sources=%s", [len(s.content) for s in result.sources])


def test_retrieve_k(vectorstore):
    """测试 retrieve_k 多召回"""
    from app.core.chain import RAGChain
    from app.core.reranker.base import BaseReranker

    logger.info("=" * 50)
    logger.info("测试: retrieve_k 多召回 + rerank 截断")

    call_log = {}

    class TrackingReranker(BaseReranker):
        def rerank(self, query, results, top_k):
            call_log["input_count"] = len(results)
            call_log["top_k"] = top_k
            return results[:top_k]

    llm = MockLLM(fixed_answer="回答")
    chain = RAGChain(
        llm=llm,
        vectorstore=vectorstore,
        top_k=2,
        retrieve_k=5,
        reranker=TrackingReranker(),
    )

    chain.invoke("考勤制度")

    assert call_log["input_count"] <= 5  # 最多 retrieve_k 个候选
    assert call_log["top_k"] == 2
    logger.info("retrieve_k 验证通过: 输入%d, 输出%d", call_log["input_count"], call_log["top_k"])


# ================================================================
# 主流程
# ================================================================

def cleanup():
    if TEST_DIR.exists():
        try:
            shutil.rmtree(TEST_DIR)
            logger.info("清理测试目录: %s", TEST_DIR)
        except PermissionError:
            logger.warning("清理失败（文件被占用），跳过")


def main():
    vectorstore = setup()

    try:
        test_invoke(vectorstore)
        test_invoke_with_history(vectorstore)
        test_invoke_no_result(vectorstore)
        test_stream(vectorstore)
        test_stream_no_result(vectorstore)
        test_search_only(vectorstore)
        test_custom_system_prompt(vectorstore)
        test_build_context()
        test_prompt_anti_hallucination()
        test_reranker_integration(vectorstore)
        test_retrieve_k(vectorstore)

        logger.info("=" * 50)
        logger.info("全部测试通过！")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
