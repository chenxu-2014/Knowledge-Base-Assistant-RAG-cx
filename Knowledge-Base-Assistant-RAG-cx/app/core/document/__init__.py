"""
文档处理管线 —— RAG 离线阶段的统一入口。

职责：将文档加载 (DocumentLoaderFactory) 和文本拆分 (DocumentSplitterFactory)
      串联为一个 process() 方法，外部代码无需关心内部细节。

调用关系：
    document.py 中 upload_document()
      └─> pipeline.process(file_path)
           ├─ DocumentLoaderFactory.create(path)    # 第一步：按格式加载
           │    └─> loader.load(path) → list[Document]
           └─ DocumentSplitterFactory.create()       # 第二步：文本分块
                └─> splitter.split(documents) → list[Document] (chunks)
"""
import logging
from pathlib import Path

from langchain_core.documents import Document

from app.core.document.loader import DocumentLoaderFactory
from app.core.document.splitter import DocumentSplitterFactory, BaseDocumentSplitter

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """文档处理管线：加载 → 分块。

    对外唯一调用点，外部不接触 Loader / Splitter 内部细节。

    Attributes:
        _loader_factory: 文档加载器工厂，按扩展名创建 Loader。
        _splitter: 文本分块器实例，按策略切分文档。
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        splitter_strategy: str = "recursive",
        loader_factory: type[DocumentLoaderFactory] = DocumentLoaderFactory,
    ):
        """初始化管线。

        Args:
            chunk_size: 分块大小（字符数），默认 500。
            chunk_overlap: 分块重叠（字符数），默认 100，保证上下文连续性。
            splitter_strategy: 分块策略，可选 "recursive" / "smart" / "fixed" / "token"。
            loader_factory: 加载器工厂，默认 DocumentLoaderFactory。
        """
        self._loader_factory = loader_factory
        # 根据策略创建对应的分块器，例如 RecursiveSplitter(chunk_size=500, chunk_overlap=100)
        self._splitter: BaseDocumentSplitter = DocumentSplitterFactory.create(
            strategy=splitter_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process(self, file_path: str | Path) -> list[Document]:
        """处理单个文件：加载 → 分块 → 返回可直接入库的 chunks。

        流程：
            1. DocumentLoaderFactory.create() — 按扩展名选 Loader
            2. loader.load() — 加载文件为 Document 列表
            3. splitter.split() — 按语义切分为 chunk
            4. 为每个 chunk 注入 _chunk_index 索引

        Args:
            file_path: 待处理的文件路径。

        Returns:
            list[Document]: 分块后的 Document 列表，可直接传入 vectorstore.upsert_documents()。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件格式不支持。
        """
        path = Path(file_path)
        logger.info("开始处理文件: %s", path.name)

        try:
            # 第一步：按扩展名选 Loader，加载文件
            loader = self._loader_factory.create(path)
            documents = loader.load(path)  # 返回 list[Document]
        except (FileNotFoundError, ValueError) as e:
            logger.error("文件加载失败: %s, 错误: %s", path.name, e)
            raise
        except Exception as e:
            logger.error("文件加载异常: %s, 错误: %s", path.name, e)
            raise

        # 第二步：文本分块
        chunks = self._splitter.split(documents)
        # 为每个 chunk 打上索引，写入 metadata，便于后续调试和版本管理
        for i, chunk in enumerate(chunks):
            chunk.metadata["_chunk_index"] = i

        logger.info("文件处理完成: %s → %d 个块", path.name, len(chunks))
        return chunks

    def process_batch(self, file_paths: list[str | Path]) -> list[Document]:
        """批量处理多个文件，合并所有 chunks。

        某个文件处理失败时跳过并记录日志，不影响其他文件。

        Args:
            file_paths: 待处理的文件路径列表。

        Returns:
            list[Document]: 所有文件的 chunks 合并后的列表。
        """
        if not file_paths:
            logger.warning("批量处理输入为空")
            return []

        all_chunks: list[Document] = []
        failed: list[str] = []

        for fp in file_paths:
            try:
                chunks = self.process(fp)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error("批量处理中跳过文件: %s, 错误: %s", fp, e)
                failed.append(str(fp))

        if failed:
            logger.warning(
                "批量处理完成: %d/%d 成功, 失败: %s",
                len(file_paths) - len(failed),
                len(file_paths),
                failed,
            )
        else:
            logger.info(
                "批量处理完成: %d 个文件, 共 %d 个块",
                len(file_paths),
                len(all_chunks),
            )

        return all_chunks

    @staticmethod
    def supported_extensions() -> list[str]:
        """返回支持的文件扩展名列表。"""
        return DocumentLoaderFactory.supported_extensions()

    @staticmethod
    def available_strategies() -> list[str]:
        """返回可用的分块策略列表。"""
        return DocumentSplitterFactory.available_strategies()
