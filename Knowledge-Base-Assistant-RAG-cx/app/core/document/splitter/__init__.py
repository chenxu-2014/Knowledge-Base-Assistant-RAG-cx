"""
分块器工厂模块 —— 按策略名称创建对应的分块器实例。

支持的分块策略：
    "recursive" — RecursiveSplitter: 按分隔符层级递归切分，保留语义边界（默认推荐）
    "smart"     — SmartSplitter: 智能分块，优先级 标题→段落→句子→字符
    "fixed"     — FixedLengthSplitter: 固定长度硬切，不考虑语义
    "token"     — TokenSplitter: 按 token 数分块（预留）

调用关系：
    DocumentPipeline.__init__()
      └─> DocumentSplitterFactory.create(strategy="recursive", chunk_size=512, chunk_overlap=50)
           └─> RecursiveSplitter(chunk_size=512, chunk_overlap=50)
"""
from app.core.document.splitter.base import BaseDocumentSplitter
from app.core.document.splitter.fixed import FixedLengthSplitter
from app.core.document.splitter.recursive import RecursiveSplitter
from app.core.document.splitter.smart import SmartSplitter
from app.core.document.splitter.token import TokenSplitter


class DocumentSplitterFactory:
    """按策略名称创建对应分块器实例。

    _strategy_map 维护策略名 → 分块器类的映射。

    使用方式：
        splitter = DocumentSplitterFactory.create("recursive", chunk_size=512, chunk_overlap=50)
        chunks = splitter.split(documents)
    """

    _strategy_map: dict[str, type[BaseDocumentSplitter]] = {
        "fixed": FixedLengthSplitter,
        "recursive": RecursiveSplitter,
        "smart": SmartSplitter,
        "token": TokenSplitter,
    }

    @classmethod
    def create(cls, strategy: str = "recursive", **kwargs) -> BaseDocumentSplitter:
        """根据策略名创建分块器实例。

        Args:
            strategy: 策略名称，可选 "recursive" / "smart" / "fixed" / "token"。
            **kwargs: 传递给分块器构造函数的参数（如 chunk_size, chunk_overlap）。

        Returns:
            BaseDocumentSplitter: 对应策略的分块器实例。

        Raises:
            ValueError: 策略名不在白名单中。
        """
        splitter_cls = cls._strategy_map.get(strategy)
        if splitter_cls is None:
            available = sorted(cls._strategy_map.keys())
            raise ValueError(
                f"不支持的分块策略: '{strategy}'，可选: {available}"
            )
        return splitter_cls(**kwargs)

    @classmethod
    def available_strategies(cls) -> list[str]:
        """返回所有可用的分块策略名称。"""
        return sorted(cls._strategy_map.keys())


__all__ = [
    "BaseDocumentSplitter",
    "FixedLengthSplitter",
    "RecursiveSplitter",
    "SmartSplitter",
    "TokenSplitter",
    "DocumentSplitterFactory",
]
