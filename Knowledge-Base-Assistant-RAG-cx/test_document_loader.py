"""
文档加载器测试脚本。

功能：
1. 自动生成测试用 PDF / DOCX / MD 样本文件
2. 逐一验证 DocumentPipeline 能否正确解析
3. 验证分块结果、元数据完整性
4. 验证异常处理（不支持格式、文件不存在）
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test")

TEST_DIR = Path(__file__).parent / "test_samples"


# ============================================================
# 1. 生成测试样本文件
# ============================================================

def create_test_markdown() -> Path:
    """生成测试用 Markdown 文件"""
    path = TEST_DIR / "test_sample.md"
    path.write_text(
        "# 企业知识库助手\n\n"
        "## 项目简介\n"
        "本项目基于 RAG 架构，实现企业内部知识库的智能问答。\n\n"
        "## 技术栈\n"
        "- FastAPI\n"
        "- LangChain\n"
        "- Chroma\n\n"
        "## 使用说明\n"
        "请将文档上传至系统，系统会自动解析并建立索引。\n"
        "用户可以通过 API 接口进行问答查询。\n",
        encoding="utf-8",
    )
    return path


def create_test_docx() -> Path:
    """生成测试用 DOCX 文件"""
    from docx import Document

    path = TEST_DIR / "test_sample.docx"
    doc = Document()
    doc.add_heading("员工手册", level=1)
    doc.add_paragraph("第一章 公司简介")
    doc.add_paragraph("本公司成立于2020年，专注于人工智能技术的研发与应用。")
    doc.add_paragraph("第二章 考勤制度")
    doc.add_paragraph("工作时间为每周一至周五，上午9:00至下午18:00。")
    doc.add_paragraph("迟到超过30分钟视为旷工半天。")
    doc.add_heading("第三章 假期制度", level=1)
    doc.add_paragraph("年假根据工龄计算，满1年可享受5天年假。")
    doc.add_paragraph("病假需提供医院证明，每年累计不超过15天。")
    doc.save(str(path))
    return path


def create_test_pdf() -> Path:
    """生成测试用 PDF 文件"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = TEST_DIR / "test_sample.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, height - 100, "Product Manual")

    c.setFont("Helvetica", 12)
    lines = [
        "Chapter 1: Introduction",
        "This product is an enterprise knowledge base assistant.",
        "It uses RAG (Retrieval-Augmented Generation) technology.",
        "",
        "Chapter 2: Features",
        "- Document upload and parsing (PDF, DOCX, Markdown)",
        "- Semantic search powered by Chroma vector database",
        "- Multi-model support (DeepSeek, Xiaomi MiMo, local models)",
        "",
        "Chapter 3: Getting Started",
        "1. Upload your documents via the API",
        "2. Wait for the system to process and index them",
        "3. Start asking questions through the chat interface",
    ]
    for i, line in enumerate(lines):
        c.drawString(100, height - 140 - i * 24, line)

    c.save()
    return path


def setup_test_files() -> dict[str, Path]:
    """生成所有测试样本，返回 {格式: 路径} 映射"""
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    files = {}
    files["md"] = create_test_markdown()
    logger.info("已生成 Markdown 测试文件: %s", files["md"])

    files["docx"] = create_test_docx()
    logger.info("已生成 DOCX 测试文件: %s", files["docx"])

    files["pdf"] = create_test_pdf()
    logger.info("已生成 PDF 测试文件: %s", files["pdf"])

    return files


# ============================================================
# 2. 测试用例
# ============================================================

REQUIRED_METADATA_KEYS = {"source", "filename", "file_type", "created_at", "source_path"}


def _assert_metadata(chunks: list, file_path: Path, expected_type: str):
    """验证所有 chunk 的 metadata 完整性"""
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        assert chunk.page_content, f"第 {i} 个 chunk 内容为空"

        missing = REQUIRED_METADATA_KEYS - set(meta.keys())
        assert not missing, f"第 {i} 个 chunk 缺少 metadata 字段: {missing}"

        assert meta["source"] == file_path.name, (
            f"第 {i} 个 chunk source 不正确: {meta['source']}"
        )
        assert meta["filename"] == file_path.name
        assert meta["file_type"] == expected_type
        assert meta["source_path"] == str(file_path.resolve())
        assert meta["created_at"], "created_at 为空"


def test_markdown(pipeline, file_path: Path):
    """测试 Markdown 解析"""
    logger.info("=" * 50)
    logger.info("测试 Markdown 解析: %s", file_path.name)

    chunks = pipeline.process(file_path)
    assert len(chunks) > 0, "Markdown 解析结果为空"
    _assert_metadata(chunks, file_path, ".md")

    logger.info("Markdown 测试通过: %d 个 chunk", len(chunks))
    _print_chunks_preview(chunks)


def test_docx(pipeline, file_path: Path):
    """测试 DOCX 解析"""
    logger.info("=" * 50)
    logger.info("测试 DOCX 解析: %s", file_path.name)

    chunks = pipeline.process(file_path)
    assert len(chunks) > 0, "DOCX 解析结果为空"
    _assert_metadata(chunks, file_path, ".docx")

    logger.info("DOCX 测试通过: %d 个 chunk", len(chunks))
    _print_chunks_preview(chunks)


def test_pdf(pipeline, file_path: Path):
    """测试 PDF 解析"""
    logger.info("=" * 50)
    logger.info("测试 PDF 解析: %s", file_path.name)

    chunks = pipeline.process(file_path)
    assert len(chunks) > 0, "PDF 解析结果为空"
    _assert_metadata(chunks, file_path, ".pdf")

    logger.info("PDF 测试通过: %d 个 chunk", len(chunks))
    _print_chunks_preview(chunks)


def test_unsupported_format(pipeline):
    """测试不支持的文件格式"""
    logger.info("=" * 50)
    logger.info("测试不支持的文件格式: .xyz")

    fake_file = TEST_DIR / "test.xyz"
    fake_file.write_text("hello")

    try:
        pipeline.process(fake_file)
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        logger.info("异常处理正确: %s", e)


def test_file_not_found(pipeline):
    """测试文件不存在"""
    logger.info("=" * 50)
    logger.info("测试文件不存在")

    try:
        pipeline.process(TEST_DIR / "nonexistent.pdf")
        assert False, "应该抛出 FileNotFoundError"
    except FileNotFoundError as e:
        logger.info("异常处理正确: %s", e)


def test_batch_process(pipeline, files: dict[str, Path]):
    """测试批量处理"""
    logger.info("=" * 50)
    logger.info("测试批量处理")

    chunks = pipeline.process_batch(list(files.values()))
    assert len(chunks) > 0, "批量处理结果为空"

    sources = {c.metadata.get("source") for c in chunks}
    assert len(sources) == len(files), f"期望 {len(files)} 个文件的 chunk，实际来源: {sources}"

    logger.info("批量处理测试通过: %d 个 chunk，来源文件 %s", len(chunks), sources)


# ============================================================
# 3. 工具函数
# ============================================================

def _print_chunks_preview(chunks, max_preview: int = 3):
    """打印 chunk 预览"""
    for i, chunk in enumerate(chunks[:max_preview]):
        preview = chunk.page_content[:80].replace("\n", " ")
        logger.info("  chunk[%d]: %s... | meta: %s", i, preview, chunk.metadata)
    if len(chunks) > max_preview:
        logger.info("  ... 共 %d 个 chunk", len(chunks))


def cleanup():
    """清理测试生成的文件"""
    import shutil
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        logger.info("已清理测试目录: %s", TEST_DIR)


# ============================================================
# 4. 主流程
# ============================================================

def main():
    from app.core.document import DocumentPipeline

    pipeline = DocumentPipeline(chunk_size=200, chunk_overlap=20)

    logger.info("支持的文件格式: %s", pipeline.supported_extensions())

    try:
        # 生成测试文件
        files = setup_test_files()

        # 逐格式测试
        test_markdown(pipeline, files["md"])
        test_docx(pipeline, files["docx"])
        test_pdf(pipeline, files["pdf"])

        # 异常测试
        test_unsupported_format(pipeline)
        test_file_not_found(pipeline)

        # 批量测试
        test_batch_process(pipeline, files)

        logger.info("=" * 50)
        logger.info("全部测试通过！")

    finally:
        cleanup()


if __name__ == "__main__":
    main()
