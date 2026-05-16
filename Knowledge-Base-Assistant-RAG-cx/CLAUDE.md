# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

基于 RAG（检索增强生成）架构的企业知识库问答系统。

## 技术栈

- **Web 框架**: FastAPI
- **LLM 编排**: LangChain（langchain-openai、langchain-chroma、langchain-community）
- **向量数据库**: Chroma（本地持久化，目录 `data/chroma_db/`）
- **大模型**: 通过 `LLMFactory` 工厂模式支持多供应商（DeepSeek / Xiaomi MiMo / 本地模型预留）

## 项目结构

```
app/
├── main.py           # FastAPI 入口，挂载路由
├── config.py         # pydantic-settings 统一配置，从 .env 注入
├── api/routes/       # API 路由层（chat.py、document.py）
├── core/
│   ├── llm.py        # LLM 供应商抽象 + 工厂模式（LLMFactory）
│   ├── vectorstore.py # Chroma 向量库管理
│   ├── document.py   # 文档加载与分块
│   └── chain.py      # RAG 链编排
├── schemas/          # Pydantic 请求/响应模型
└── utils/            # 工具函数
data/
├── documents/        # 上传文档存放
└── chroma_db/        # Chroma 持久化
tests/                # 测试
```

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务
uvicorn app.main:app --reload

# 运行测试
pytest tests/

# 查看 API 文档（启动后）
# http://localhost:8000/docs
```

## 核心设计

- **LLM 多供应商**: `app/core/llm.py` 中 `LLMFactory.create(provider_name)` 按配置创建供应商实例，`BaseLLMProvider` 定义 `get_chat_model`（普通对话）和 `get_agent_model`（Agent 调用，禁用 thinking 模式）两个接口。
- **配置管理**: `app/config.py` 使用 pydantic `BaseSettings`，所有配置项通过 `.env` 注入，新增配置只需加字段。
- **向量库**: `VectorStoreManager` 封装 Chroma 的增删查操作，对外暴露 `add_documents` / `similarity_search` / `delete_document`。
- **MiMo 模型**: 默认开启 thinking 模式，Agent 场景需 `extra_body={'enable_thinking': False}` 禁用。
