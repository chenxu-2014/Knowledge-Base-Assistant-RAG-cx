from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI


class BaseLLMProvider(ABC):
    """大模型供应商抽象基类"""

    @abstractmethod
    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        """返回用于普通对话的模型实例"""
        ...

    @abstractmethod
    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        """返回用于 Agent 多轮工具调用的模型实例（禁用 thinking 等）"""
        ...


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API 供应商"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError

    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError


class XiaomiProvider(BaseLLMProvider):
    """小米 MiMo API 供应商"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError

    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError


class LocalLLMProvider(BaseLLMProvider):
    """本地大模型供应商（预留）"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name

    def get_chat_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError

    def get_agent_model(self, **kwargs) -> ChatOpenAI:
        raise NotImplementedError


class LLMFactory:
    """根据配置创建对应的 LLM 供应商实例"""

    _providers: dict[str, type[BaseLLMProvider]] = {
        "deepseek": DeepSeekProvider,
        "xiaomi": XiaomiProvider,
        "local": LocalLLMProvider,
    }

    @classmethod
    def create(cls, provider_name: str, **kwargs) -> BaseLLMProvider:
        if provider_name not in cls._providers:
            raise ValueError(f"不支持的供应商: {provider_name}，可选: {list(cls._providers.keys())}")
        return cls._providers[provider_name](**kwargs)

    @classmethod
    def register(cls, name: str, provider_class: type[BaseLLMProvider]):
        """注册自定义供应商"""
        cls._providers[name] = provider_class
