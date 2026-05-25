from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

# 以 config.py 自身位置推算项目根目录，无论从哪里运行都能找到 .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """统一配置管理，所有配置从 .env 注入，硬编码默认值已移除"""

    # LLM 配置
    llm_provider: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    xiaomi_api_key: str = ""
    xiaomi_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    xiaomi_model: str = "mimo-v2-pro"

    # 本地模型预留
    local_model_path: str = ""
    local_model_name: str = ""

    # Ollama 配置
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"

    # Embedding 配置
    embedding_provider: str = "deepseek"
    embedding_deepseek_model: str = "deepseek-embedding"
    embedding_xiaomi_model: str = "text-embedding-001"
    embedding_local_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_ollama_model: str = "nomic-embed-text:v1.5"

    # Chroma 配置
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # 文档分块配置
    chunk_size: int = 512
    chunk_overlap: int = 50

    # 检索配置
    search_top_k: int = 4

    model_config = {"env_file": str(_ENV_PATH), "env_file_encoding": "utf-8"}


settings = Settings()
