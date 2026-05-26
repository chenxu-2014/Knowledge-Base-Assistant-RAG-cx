"""
LLM 供应商抽象层 —— 大模型多供应商支持。

设计模式：工厂模式 (LLMFactory) + 策略模式 (各 Provider 实现)

职责：定义统一的 LLM 供应商接口，通过 LLMFactory 根据配置创建对应供应商实例。

支持的供应商：
    "deepseek" — DeepSeek API（OpenAI 兼容协议）
    "xiaomi"   — 小米 MiMo API（OpenAI 兼容协议）
    "local"    — 本地大模型（预留）

注意：实际的 RAG 链中（main.py）并未使用此模块的工厂，而是直接用 ChatOpenAI / ChatOllama，
      因为 DeepSeek 和 MiMo 都兼容 OpenAI 协议。此模块保留用于需要按供应商差异
      配置不同参数的场景（如禁用 MiMo 的 thinking 模式）。

调用关系：
    llm = LLMFactory.create("deepseek", api_key="xxx", base_url="...", model="...")
    provider = llm.get_chat_model()
"""
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI


class BaseLLMProvider(ABC):
    """大模型供应商抽象基类。

    定义两个接口：
        get_chat_model()  — 返回普通对话的模型实例
        get_agent_model() — 返回 Agent 场景的模型实例（禁用 thinking 等特殊模式）
    """

    @abstractmethod
    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        """返回用于普通对话的模型实例。"""
        ...

    @abstractmethod
    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        """返回用于 Agent 多轮工具调用的模型实例（禁用 thinking 等）。"""
        ...


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API 供应商。

    DeepSeek API 兼容 OpenAI 协议，直接使用 ChatOpenAI 类。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError

    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError


class XiaomiProvider(BaseLLMProvider):
    """小米 MiMo API 供应商。

    MiMo 默认开启 thinking 模式，Agent 场景需
    extra_body={'enable_thinking': False} 禁用。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError

    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError


class LocalLLMProvider(BaseLLMProvider):
    """本地大模型供应商（预留）。

    未来可支持通过 vLLM / llama.cpp 等本地推理框架加载模型。
    """

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name

    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError

    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError


class LLMFactory:
    """LLM 供应商工厂 —— 根据配置创建对应的 LLM 供应商实例。

    _providers 维护供应商名 → 供应商类的映射。

    使用方式：
        provider = LLMFactory.create("deepseek", api_key="xxx", base_url="...", model="...")
        llm = provider.get_chat_model()

    扩展方式：
        LLMFactory.register("custom", CustomProvider)  # 注册自定义供应商
    """

    _providers: dict[str, type[BaseLLMProvider]] = {
        "deepseek": DeepSeekProvider,
        "xiaomi": XiaomiProvider,
        "local": LocalLLMProvider,
    }

    @classmethod
    def create(cls, provider_name: str, **kwargs) -> BaseLLMProvider:
        """根据供应商名创建实例。

        Args:
            provider_name: 供应商名称。
            **kwargs: 传递给供应商构造函数的参数。

        Returns:
            BaseLLMProvider: 供应商实例。

        Raises:
            ValueError: 供应商不存在。
        """
        if provider_name not in cls._providers:
            raise ValueError(f"不支持的供应商: {provider_name}，可选: {list(cls._providers.keys())}")
        return cls._providers[provider_name](**kwargs)

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLMProvider]):
        """注册自定义供应商。

        Args:
            name: 供应商名称。
            provider_class: 供应商类（需继承 BaseLLMProvider）。
        """
        cls._providers[name] = provider_class
