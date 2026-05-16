from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """统一配置管理，通过环境变量或 .env 文件注入"""

    # LLM 配置
    llm_provider: str = Field(default="deepseek", description="大模型供应商: deepseek | xiaomi | local")
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    xiaomi_api_key: str = ""
    xiaomi_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    xiaomi_model: str = "mimo-v2-pro"

    # 本地模型预留
    local_model_path: str = ""
    local_model_name: str = ""

    # Embedding 配置
    embedding_provider: str = Field(default="deepseek", description="Embedding 供应商: deepseek | xiaomi | local")
    embedding_deepseek_model: str = "deepseek-embedding"
    embedding_xiaomi_model: str = "text-embedding-001"
    embedding_local_model: str = "BAAI/bge-small-zh-v1.5"

    # Chroma 配置
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # 文档分块配置
    chunk_size: int = 512
    chunk_overlap: int = 50

    # 检索配置
    search_top_k: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
