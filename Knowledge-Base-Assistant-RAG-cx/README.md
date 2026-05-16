# 企业知识库助手（RAG）

基于 RAG（检索增强生成）架构的企业知识库问答系统。

## 技术栈

- **Web 框架**: FastAPI
- **LLM 编排**: LangChain
- **向量数据库**: Chroma（本地持久化）
- **大模型**: DeepSeek / Xiaomi MiMo（通过 API），预留本地模型接入

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 3. 启动服务
uvicorn app.main:app --reload

# 访问 API 文档
# http://localhost:8000/docs
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 知识库问答 |
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 文档列表 |
| DELETE | `/api/documents/{doc_id}` | 删除文档 |
| GET | `/health` | 健康检查 |

## 后续计划

- 知识库 RAG 系统整体搭建，实现从文档解析、向量化、召回到生成回答的完整链路，支持 PDF、Word、Markdown 等多格式知识源接入。
- 基于 LangChain 构建检索增强生成流程，对文档进行切分、Embedding 向量化存储，并通过语义相似度检索实现上下文精准召回。
- 使用 Chroma 搭建本地向量数据库，完成知识分片索引管理，支持增量更新、批量重建及历史版本回溯。
- 设计 Prompt 模板和上下文拼接机制，优化检索结果排序，降低模型幻觉问题，提高答案准确率及知识命中率。
- 使用 FastAPI 提供统一 REST 接口，支持前端问答页面、内部 IM Bot 和 API 服务调用。
- 基于日志埋点记录用户提问、检索内容、模型响应耗时，建立问答链路监控，辅助优化召回质量与推理性能。
- 使用 Docker 完成本地容器化部署，支持快速环境复制及内网落地。