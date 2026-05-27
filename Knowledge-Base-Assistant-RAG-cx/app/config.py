"""
统一配置管理 —— 所有配置项从 .env 文件注入。

职责：
    使用 pydantic-settings 的 BaseSettings 类自动从 .env 文件读取配置，
    无需手动 load_dotenv，类型安全，支持默认值。

配置分类：
    - LLM 配置: llm_provider, deepseek/xiaomi/local/ollama 的 api_key/base_url/model
    - Embedding 配置: embedding_provider, 各供应商的 model 名
    - Chroma 配置: 持久化目录、集合名称
    - 分块配置: chunk_size, chunk_overlap
    - 检索配置: search_top_k

使用方式：
    from app.config import settings
    print(settings.llm_provider)       # "deepseek"
    print(settings.chunk_size)         # 512
"""
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

# 以 config.py 自身位置推算项目根目录，无论从哪里运行都能找到 .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """统一配置管理，所有配置从 .env 注入，硬编码默认值已移除。

    pydantic-settings 会自动：
        1. 从 .env 文件读取环境变量
        2. 按字段名匹配（大小写不敏感）
        3. 类型转换（如 str → int）
    """

    # ─── LLM 配置 ───
    llm_provider: str = "deepseek"  # LLM 供应商选择: deepseek | xiaomi | ollama

    # DeepSeek 配置
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # 小米 MiMo 配置
    xiaomi_api_key: str = ""
    xiaomi_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    xiaomi_model: str = "mimo-v2-pro"

    # 本地模型预留
    local_model_path: str = ""
    local_model_name: str = ""

    # Ollama 配置
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"

    # ─── Embedding 配置 ───
    embedding_provider: str = "deepseek"  # Embedding 供应商: deepseek | xiaomi | local | ollama
    embedding_deepseek_model: str = "deepseek-embedding"
    embedding_xiaomi_model: str = "text-embedding-001"
    embedding_local_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_ollama_model: str = "nomic-embed-text:v1.5"

    # ─── Chroma 配置 ───
    chroma_persist_dir: str = "data/chroma_db"  # 向量库持久化目录
    chroma_collection_name: str = "knowledge_base"  # 集合名称（类似数据库表名）

    # ─── 文档分块配置 ───
    chunk_size: int = 512    # 每个 chunk 的最大字符数
    chunk_overlap: int = 50  # 相邻 chunk 的重叠字符数

    # ─── 检索配置 ───
    search_top_k: int = 4  # 检索返回的最相关结果数量
    search_score_threshold: float = 0.6  # 余弦相似度阈值，低于此分数视为无关，回退到大模型直接回答

    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8"}


# 全局单例，所有模块统一引用此实例
settings = Settings()
