from app.core.document.splitter.base import BaseDocumentSplitter
from app.core.document.splitter.fixed import FixedLengthSplitter
from app.core.document.splitter.recursive import RecursiveSplitter
from app.core.document.splitter.smart import SmartSplitter
from app.core.document.splitter.token import TokenSplitter


class DocumentSplitterFactory:
    """按策略名称创建对应分块器实例"""

    _strategy_map: dict[str, type[BaseDocumentSplitter]] = {
        "fixed": FixedLengthSplitter,
        "recursive": RecursiveSplitter,
        "smart": SmartSplitter,
        "token": TokenSplitter,
    }

    @classmethod
    def create(cls, strategy: str = "recursive", **kwargs) -> BaseDocumentSplitter:
        splitter_cls = cls._strategy_map.get(strategy)
        if splitter_cls is None:
            available = sorted(cls._strategy_map.keys())
            raise ValueError(
                f"不支持的分块策略: '{strategy}'，可选: {available}"
            )
        return splitter_cls(**kwargs)

    @classmethod
    def available_strategies(cls) -> list[str]:
        return sorted(cls._strategy_map.keys())


__all__ = [
    "BaseDocumentSplitter",
    "FixedLengthSplitter",
    "RecursiveSplitter",
    "SmartSplitter",
    "TokenSplitter",
    "DocumentSplitterFactory",
]
