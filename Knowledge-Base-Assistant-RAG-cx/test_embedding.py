"""
Embedding Factory 测试脚本。
验证根据环境变量自动切换 provider。
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test")


def test_factory_available_providers():
    """测试可用 provider 列表"""
    from app.core.embedding import EmbeddingFactory

    logger.info("=" * 50)
    available = EmbeddingFactory.available_providers()
    logger.info("可用 Embedding 供应商: %s", available)
    assert "deepseek" in available
    assert "xiaomi" in available
    logger.info("可用供应商测试通过")


def test_factory_create_deepseek():
    """测试工厂创建 DeepSeek embedding"""
    from app.core.embedding import EmbeddingFactory
    from langchain_openai import OpenAIEmbeddings

    logger.info("=" * 50)
    emb = EmbeddingFactory.create(
        "deepseek",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-embedding",
    )
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.model == "deepseek-embedding"
    logger.info("DeepSeek embedding 创建成功: model=%s", emb.model)


def test_factory_create_xiaomi():
    """测试工厂创建 Xiaomi embedding"""
    from app.core.embedding import EmbeddingFactory
    from langchain_openai import OpenAIEmbeddings

    logger.info("=" * 50)
    emb = EmbeddingFactory.create(
        "xiaomi",
        api_key="test-key",
        base_url="https://token-plan-cn.xiaomimimo.com/v1",
        model="text-embedding-001",
    )
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.model == "text-embedding-001"
    logger.info("Xiaomi embedding 创建成功: model=%s", emb.model)


def test_factory_unsupported_provider():
    """测试不支持的 provider"""
    from app.core.embedding import EmbeddingFactory

    logger.info("=" * 50)
    try:
        EmbeddingFactory.create("nonexistent", api_key="x")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        logger.info("异常处理正确: %s", e)


def test_from_settings_deepseek():
    """测试从配置创建 DeepSeek embedding"""
    from app.core.embedding import EmbeddingFactory
    from app.config import Settings

    logger.info("=" * 50)

    settings = Settings(
        embedding_provider="deepseek",
        deepseek_api_key="test-ds-key",
        deepseek_base_url="https://api.deepseek.com",
        embedding_deepseek_model="deepseek-embedding",
    )

    emb = EmbeddingFactory.from_settings(settings)
    assert emb.model == "deepseek-embedding"
    logger.info("from_settings(deepseek) 成功: model=%s", emb.model)


def test_from_settings_xiaomi():
    """测试从配置创建 Xiaomi embedding"""
    from app.core.embedding import EmbeddingFactory
    from app.config import Settings

    logger.info("=" * 50)

    settings = Settings(
        embedding_provider="xiaomi",
        xiaomi_api_key="test-xm-key",
        xiaomi_base_url="https://token-plan-cn.xiaomimimo.com/v1",
        embedding_xiaomi_model="text-embedding-001",
    )

    emb = EmbeddingFactory.from_settings(settings)
    assert emb.model == "text-embedding-001"
    logger.info("from_settings(xiaomi) 成功: model=%s", emb.model)


def test_from_settings_env_switch():
    """测试通过环境变量自动切换 provider"""
    from importlib import reload
    import app.config as config_mod

    logger.info("=" * 50)

    # 模拟 .env: EMBEDDING_PROVIDER=xiaomi
    os.environ["EMBEDDING_PROVIDER"] = "xiaomi"
    os.environ["XIAOMI_API_KEY"] = "env-xm-key"
    os.environ["XIAOMI_BASE_URL"] = "https://token-plan-cn.xiaomimimo.com/v1"
    os.environ["EMBEDDING_XIAOMI_MODEL"] = "text-embedding-001"

    try:
        reload(config_mod)
        settings = config_mod.Settings()

        from app.core.embedding import EmbeddingFactory
        emb = EmbeddingFactory.from_settings(settings)
        assert emb.model == "text-embedding-001"
        logger.info("环境变量切换成功: provider=%s, model=%s", settings.embedding_provider, emb.model)
    finally:
        # 清理环境变量
        for key in ["EMBEDDING_PROVIDER", "XIAOMI_API_KEY", "XIAOMI_BASE_URL", "EMBEDDING_XIAOMI_MODEL"]:
            os.environ.pop(key, None)
        reload(config_mod)


def test_custom_provider():
    """测试注册自定义 provider"""
    from app.core.embedding import EmbeddingFactory
    from langchain_openai import OpenAIEmbeddings

    logger.info("=" * 50)

    def create_custom_embedding(api_key: str, **kwargs):
        return OpenAIEmbeddings(api_key=api_key, base_url="https://custom.com", model="custom-emb")

    EmbeddingFactory.register("custom", create_custom_embedding)
    emb = EmbeddingFactory.create("custom", api_key="custom-key")
    assert emb.model == "custom-emb"
    assert "custom" in EmbeddingFactory.available_providers()
    logger.info("自定义 provider 注册成功: model=%s", emb.model)


def main():
    try:
        test_factory_available_providers()
        test_factory_create_deepseek()
        test_factory_create_xiaomi()
        test_factory_unsupported_provider()
        test_from_settings_deepseek()
        test_from_settings_xiaomi()
        test_from_settings_env_switch()
        test_custom_provider()

        logger.info("=" * 50)
        logger.info("全部测试通过！")
    except Exception as e:
        logger.error("测试失败: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
