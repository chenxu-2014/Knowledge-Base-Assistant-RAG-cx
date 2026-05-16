from pathlib import Path

from langchain_core.documents import Document


class DocumentLoader:
    """文档加载器，支持多种格式"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def load(self, file_path: str | Path) -> list[Document]:
        """根据文件扩展名选择合适的 Loader 加载文档"""
        raise NotImplementedError

    def validate(self, file_path: str | Path) -> bool:
        """校验文件格式是否支持"""
        raise NotImplementedError


class DocumentSplitter:
    """文档分块器"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        """将文档切分为固定大小的块"""
        raise NotImplementedError
