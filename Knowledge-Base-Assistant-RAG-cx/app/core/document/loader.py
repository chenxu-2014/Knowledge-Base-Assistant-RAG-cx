"""
文档加载模块 —— RAG 离线阶段第一步。

职责：将不同格式的文件（PDF / DOCX / Markdown / TXT）解析为统一的
      langchain Document 对象列表，每个 Document 携带标准化 metadata。

设计模式：工厂模式 (DocumentLoaderFactory) + 策略模式 (各 Loader 实现)

调用关系：
    DocumentPipeline.process()
      └─> DocumentLoaderFactory.create(path)   # 按扩展名分发
           └─> PdfLoader / DocxLoader / MarkdownLoader
                └─> loader.load(path)           # 返回 list[Document]
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """文档加载器抽象基类。

    所有格式特定的 Loader（PdfLoader、DocxLoader 等）继承此类，
    复用文件校验 (_validate) 和 metadata 构建 (_build_metadata) 逻辑。
    子类只需实现 load() 方法。
    """

    SUPPORTED_EXTENSIONS: set[str] = set()

    def _validate(self, file_path: Path) -> None:
        """校验文件是否存在、是否为普通文件、扩展名是否在白名单内。

        Args:
            file_path: 待加载的文件路径。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 不是文件或扩展名不支持。
        """
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
        """统一生成文档 metadata，注入到每个 Document 对象中。

        metadata 字段说明：
            source      — 文件名（后续检索时作为引用来源标识）
            filename    — 文件名（冗余，方便某些场景使用）
            file_type   — 扩展名，如 ".pdf"
            created_at  — 文件创建时间 ISO 格式
            source_path — 文件的绝对路径

        这些 metadata 会在后续分块时自动继承到每个 chunk，
        最终写入 Chroma 向量库，用于检索结果的来源追溯。
        """
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
        """加载文档，返回 Document 列表。

        PDF 每页一个 Document，DOCX/Markdown 整篇一个 Document。
        每个 Document 的 metadata 已通过 _build_metadata 填充。

        Args:
            file_path: 文件路径。

        Returns:
            list[Document]: 加载后的文档对象列表。
        """
        ...


class PdfLoader(BaseDocumentLoader):
    """PDF 文档加载器，逐页加载。

    使用 langchain_community 的 PyPDFLoader 解析 PDF，
    每页生成一个 Document，保留页码等信息。
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def load(self, file_path: str | Path) -> list[Document]:
        """加载 PDF 文件，每页一个 Document。

        Returns:
            list[Document]: 每页一个 Document 对象，metadata 包含文件信息。
        """
        path = Path(file_path)
        self._validate(path)  # 校验文件存在且扩展名为 .pdf
        logger.info("加载 PDF 文件: %s", path.name)

        try:
            loader = PyPDFLoader(str(path))
            documents = loader.load()  # 逐页解析，每页返回一个 Document
        except Exception as e:
            logger.error("PDF 加载失败: %s, 错误: %s", path.name, e)
            raise

        # 将统一 metadata（source、file_type 等）注入到每个页面的 Document
        metadata = self._build_metadata(path)
        for doc in documents:
            doc.metadata.update(metadata)

        logger.info("PDF 加载完成: %s, 共 %d 页", path.name, len(documents))
        return documents


class DocxLoader(BaseDocumentLoader):
    """Word 文档加载器。

    使用 langchain_community 的 Docx2txtLoader 解析 .docx 文件，
    整篇文档生成一个 Document。
    """

    SUPPORTED_EXTENSIONS = {".docx"}

    def load(self, file_path: str | Path) -> list[Document]:
        """加载 DOCX 文件，整篇一个 Document。"""
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
    """Markdown / TXT 文档加载器。

    使用 langchain_community 的 TextLoader 按 UTF-8 编码读取文本。
    .md、.markdown、.txt 均复用此 Loader。
    """

    SUPPORTED_EXTENSIONS = {".txt",".md", ".markdown"}

    def load(self, file_path: str | Path) -> list[Document]:
        """加载 Markdown/TXT 文件，整篇一个 Document。"""
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
    """文档加载器工厂 —— 根据文件扩展名自动分发到对应 Loader。

    _extension_map 维护扩展名 → Loader 类的映射关系。
    .txt 复用 MarkdownLoader（纯文本读取逻辑相同）。

    使用方式：
        loader = DocumentLoaderFactory.create("report.pdf")  # 返回 PdfLoader 实例
        docs = loader.load("report.pdf")
    """

    _extension_map: dict[str, type[BaseDocumentLoader]] = {
        ".pdf": PdfLoader,
        ".docx": DocxLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".txt": MarkdownLoader,  # 纯文本复用 MarkdownLoader
    }

    @classmethod
    def create(cls, file_path: str | Path) -> BaseDocumentLoader:
        """根据文件扩展名创建对应 Loader 实例。

        Args:
            file_path: 文件路径，仅用于提取扩展名。

        Returns:
            BaseDocumentLoader: 对应格式的 Loader 实例。

        Raises:
            ValueError: 扩展名不在白名单中。
        """
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
        """返回所有支持的扩展名列表。"""
        return sorted(set(cls._extension_map.keys()))
