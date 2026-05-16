import logging
from pathlib import Path

from langchain_core.documents import Document

from app.core.document.loader import DocumentLoaderFactory, BaseDocumentLoader
from app.core.document.splitter import DocumentSplitter

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    文档处理管线：加载 → 分块。
    对外唯一调用点，外部不接触 Loader / Splitter 内部细节。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        loader_factory: type[DocumentLoaderFactory] = DocumentLoaderFactory,
    ):
        self._factory = loader_factory
        self._splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process(self, file_path: str | Path) -> list[Document]:
        """
        处理单个文件：加载 → 分块 → 返回可直接入库的 chunks。
        每个 chunk 的 metadata 中包含 source（原始文件名）。
        """
        path = Path(file_path)
        logger.info("开始处理文件: %s", path.name)

        try:
            loader = self._factory.create(path)
            documents = loader.load(path)
        except (FileNotFoundError, ValueError) as e:
            logger.error("文件加载失败: %s, 错误: %s", path.name, e)
            raise
        except Exception as e:
            logger.error("文件加载异常: %s, 错误: %s", path.name, e)
            raise

        chunks = self._splitter.split(documents)

        logger.info("文件处理完成: %s → %d 个块", path.name, len(chunks))
        return chunks

    def process_batch(self, file_paths: list[str | Path]) -> list[Document]:
        """批量处理多个文件，合并所有 chunks"""
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
        """返回支持的文件扩展名列表"""
        return DocumentLoaderFactory.supported_extensions()
