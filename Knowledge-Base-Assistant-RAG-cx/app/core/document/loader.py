import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """文档加载器抽象基类"""

    SUPPORTED_EXTENSIONS: set[str] = set()

    def _validate(self, file_path: Path) -> None:
        """校验文件是否存在且扩展名匹配"""
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"不是文件: {file_path}")
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {file_path.suffix}，"
                f"期望: {self.SUPPORTED_EXTENSIONS}"
            )

    @staticmethod
    def _build_metadata(file_path: Path) -> dict:
        """统一生成文档 metadata"""
        stat = file_path.stat()
        return {
            "source": file_path.name,
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "source_path": str(file_path.resolve()),
        }

    @abstractmethod
    def load(self, file_path: str | Path) -> list[Document]:
        """加载文档，返回 Document 列表"""
        ...


class PdfLoader(BaseDocumentLoader):
    """PDF 文档加载器，逐页加载"""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        self._validate(path)
        logger.info("加载 PDF 文件: %s", path.name)

        try:
            loader = PyPDFLoader(str(path))
            documents = loader.load()
        except Exception as e:
            logger.error("PDF 加载失败: %s, 错误: %s", path.name, e)
            raise

        metadata = self._build_metadata(path)
        for doc in documents:
            doc.metadata.update(metadata)

        logger.info("PDF 加载完成: %s, 共 %d 页", path.name, len(documents))
        return documents


class DocxLoader(BaseDocumentLoader):
    """Word 文档加载器"""

    SUPPORTED_EXTENSIONS = {".docx"}

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        self._validate(path)
        logger.info("加载 DOCX 文件: %s", path.name)

        try:
            loader = Docx2txtLoader(str(path))
            documents = loader.load()
        except Exception as e:
            logger.error("DOCX 加载失败: %s, 错误: %s", path.name, e)
            raise

        metadata = self._build_metadata(path)
        for doc in documents:
            doc.metadata.update(metadata)

        logger.info("DOCX 加载完成: %s", path.name)
        return documents


class MarkdownLoader(BaseDocumentLoader):
    """Markdown 文档加载器"""

    SUPPORTED_EXTENSIONS = {".md", ".markdown"}

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        self._validate(path)
        logger.info("加载 Markdown 文件: %s", path.name)

        try:
            loader = TextLoader(str(path), encoding="utf-8")
            documents = loader.load()
        except Exception as e:
            logger.error("Markdown 加载失败: %s, 错误: %s", path.name, e)
            raise

        metadata = self._build_metadata(path)
        for doc in documents:
            doc.metadata.update(metadata)

        logger.info("Markdown 加载完成: %s", path.name)
        return documents


class DocumentLoaderFactory:
    """根据文件扩展名分发到对应 Loader"""

    _extension_map: dict[str, type[BaseDocumentLoader]] = {
        ".pdf": PdfLoader,
        ".docx": DocxLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".txt": MarkdownLoader,
    }

    @classmethod
    def create(cls, file_path: str | Path) -> BaseDocumentLoader:
        suffix = Path(file_path).suffix.lower()
        loader_cls = cls._extension_map.get(suffix)
        if loader_cls is None:
            supported = sorted(set(cls._extension_map.keys()))
            raise ValueError(
                f"不支持的文件格式: '{suffix}'，支持的格式: {supported}"
            )
        return loader_cls()

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return sorted(set(cls._extension_map.keys()))
