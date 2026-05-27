# 企业知识库助手（RAG）

基于 RAG（检索增强生成）架构的企业知识库问答系统，支持多格式文档上传、混合检索、对话记忆和流式输出。

## 技术栈

- **Web 框架**: FastAPI
- **LLM 编排**: LangChain
- **向量数据库**: Chroma（本地持久化）/ Milvus（混合检索）
- **大模型**: DeepSeek / Xiaomi MiMo / Ollama
- **Embedding**: 本地 BAAI/bge-small-zh-v1.5（中文优化）
- **Reranker**: BAAI/bge-reranker-v2-m3（CrossEncoder 精排）

## 项目结构

```
app/
├── main.py                     # FastAPI 入口，组件初始化
├── config.py                   # pydantic-settings 统一配置（.env 注入）
├── api/routes/
│   ├── chat.py                 # POST /api/chat — 知识库问答（SSE 流式）
│   └── document.py             # 文档上传/列表/删除
├── core/
│   ├── chain/
│   │   ├── __init__.py         # RAGChainFactory
│   │   ├── rag_chain.py        # RAG 推理链（检索→改写→rerank→LLM）
│   │   └── prompts.py          # Prompt 模板（抗幻觉 + 查询改写）
│   ├── document/
│   │   ├── __init__.py         # DocumentPipeline（加载→分块）
│   │   ├── loader.py           # 文档加载器（PDF/DOCX/MD/TXT）
│   │   └── splitter/           # 分块策略（recursive/smart/fixed/token）
│   ├── embedding/              # Embedding 供应商抽象
│   ├── reranker/               # CrossEncoder 重排序器
│   ├── vectorstore/
│   │   ├── __init__.py         # VectorStoreFactory（Chroma/Milvus 路由）
│   │   ├── manager.py          # Chroma 向量库管理
│   │   ├── milvus_manager.py   # Milvus 向量库管理（混合检索）
│   │   └── models.py           # 数据模型（SearchResult 等）
│   └── llm.py                  # LLM 供应商抽象 + 工厂
├── schemas/                    # Pydantic 请求/响应模型
static/
└── index.html                  # 前端问答页面
data/
├── documents/                  # 上传文档存放
└── chroma_db/                  # Chroma 持久化
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 3. 启动服务
uvicorn app.main:app --reload

# 访问页面: http://localhost:8000
# API 文档: http://localhost:8000/docs
```

## 核心特性

### 混合检索（Milvus）

支持 Dense 向量 + Sparse BM25 混合检索，通过 WeightedRanker 融合两路结果。切换方式：

```env
VECTORSTORE_PROVIDER=milvus
MILVUS_URI=http://localhost:19530
```

### 查询改写

有对话历史时，自动用 LLM 补全省略信息（如"它还有什么方法？"→"HashMap还有什么方法？"），提升向量检索命中率。

### 动态阈值

不依赖固定相似度阈值，通过分数断层检测自动判断相关性边界，至少保留 top_k 个结果。

### Reranker 精排

向量检索粗召回后，用 CrossEncoder（BAAI/bge-reranker-v2-m3）对结果重排序，过滤噪声。

### 对话记忆

前端维护最近 10 轮对话历史，支持多轮上下文问答。

### 知识库兜底

知识库无结果时自动调用大模型直接回答，并标注"当前知识库中未找到相关信息"。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 知识库问答（支持 SSE 流式） |
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 文档列表 |
| DELETE | `/api/documents/{doc_id}` | 删除文档 |
| GET | `/health` | 健康检查 |

## RAG 流程

```
用户问题
  │
  ├─ 查询改写（有对话历史时）→ 补全省略信息
  │
  ├─ 向量检索 → retrieve_k 个候选
  │    ├─ Dense 向量语义匹配
  │    └─ Sparse BM25 关键词匹配（Milvus 混合检索时）
  │
  ├─ 动态阈值过滤 → 分数断层检测 + top_k 保底
  │
  ├─ Reranker 精排（可选）→ CrossEncoder 重排序
  │
  ├─ 无结果 → 调用大模型直接回答
  │
  └─ 有结果 → 拼接上下文 → 调用 LLM → 返回回答 + 引用来源
```

## 关键配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `llm_provider` | `deepseek` | LLM 供应商：deepseek / xiaomi / ollama |
| `embedding_provider` | `local` | Embedding 供应商：local / deepseek / xiaomi / ollama |
| `vectorstore_provider` | `chroma` | 向量库：chroma / milvus |
| `chunk_size` | `512` | 分块大小（字符数） |
| `chunk_overlap` | `50` | 分块重叠（字符数） |
| `search_top_k` | `4` | 检索返回数量 |
| `reranker_backend` | `cross_encoder` | 重排序器：none / cross_encoder |
| `query_rewrite` | `true` | 是否启用查询改写 |
| `dynamic_threshold` | `true` | 是否启用动态阈值 |

## 更新记录

### 2026-05-28

- **向量数据库**: 支持 Milvus 混合检索（Dense + BM25），通过 `VECTORSTORE_PROVIDER` 切换
- **Reranker**: 启用 CrossEncoder 精排（BAAI/bge-reranker-v2-m3），提升检索精度
- **查询改写**: 有对话历史时自动用 LLM 补全省略信息，提升向量检索命中率
- **动态阈值**: 检索阈值改为分数断层检测 + top_k 保底，不再依赖固定阈值

### 2026-05-27

- **模型切换**: LLM 切换至 DeepSeek v4-flash，Embedding 切换至本地 BAAI/bge-small-zh-v1.5
- **对话记忆**: 前端维护聊天历史，支持多轮上下文问答
- **知识库兜底**: 无结果时调用大模型直接回答
- **分块优化**: MD 文件使用 SmartSplitter（按标题/段落语义分块）

### 2026-05-26

- 文档加载支持 TXT 格式
- 前端改为 SSE 流式接收
- 添加 RAG 流程文档
