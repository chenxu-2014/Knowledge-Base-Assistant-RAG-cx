# RAG 整体流程文档

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        离线数据处理阶段                              │
│                                                                     │
│  文档加载          文本拆分          文本嵌入          存入向量数据库   │
│  ┌──────────┐    ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Document │───>│ BaseDocument │─>│ Embedding    │─>│ VectorStore│  │
│  │ Loader   │    │ Splitter     │  │ Factory      │  │ Manager   │  │
│  │ Factory  │    │ Factory      │  │              │  │           │  │
│  └──────────┘    └──────────────┘  └──────────────┘  └───────────┘  │
│       │                │                                    │       │
│       v                v                                    v       │
│  PdfLoader       RecursiveSplitter    create_deepseek_     Chroma   │
│  DocxLoader      SmartSplitter        embedding()          (本地    │
│  MarkdownLoader  FixedLengthSplitter  create_xiaomi_       持久化)  │
│  (loader.py)     (splitter/)          embedding()                  │
│                                     (embedding/)                   │
│                                                                     │
│  统一入口: DocumentPipeline.process(file_path)                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        在线问答阶段 (Chat)                           │
│                                                                     │
│  用户问题      向量检索       拼接上下文      调用LLM      生成回答    │
│  ┌────────┐  ┌───────────┐  ┌──────────┐  ┌─────────┐  ┌────────┐ │
│  │ Chat   │─>│ VectorStore│─>│ RAGChain │─>│ ChatOpen│─>│ RAGRes-│ │
│  │ Request│  │ Manager   │  │          │  │ AI/Chat │  │ ult    │ │
│  │        │  │ .similarity│  │ _build_  │  │ Ollama  │  │        │ │
│  └────────┘  │ _search() │  │ context()│  │         │  └────────┘ │
│              └───────────┘  │ _build_  │  └─────────┘             │
│                             │ messages()│                          │
│                             └──────────┘                           │
│                                    │                               │
│                             ┌──────┴──────┐                       │
│                             │ Prompts 模板 │                       │
│                             │ (抗幻觉设计) │                       │
│                             └─────────────┘                       │
│                                                                     │
│  可选增强: BaseReranker (CrossEncoderReranker)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 一、离线数据阶段

### 1.1 文档加载

**目标**: 将不同格式的文档（PDF、DOCX、Markdown、TXT）加载为统一的 `Document` 对象。

| 类名 | 文件 | 职责 |
|------|------|------|
| `DocumentLoaderFactory` | `app/core/document/loader.py` | 根据文件扩展名自动分发到对应 Loader |
| `BaseDocumentLoader` | `app/core/document/loader.py` | 加载器抽象基类，定义校验和 metadata 构建 |
| `PdfLoader` | `app/core/document/loader.py` | PDF 逐页加载（依赖 `PyPDFLoader`） |
| `DocxLoader` | `app/core/document/loader.py` | Word 文档加载（依赖 `Docx2txtLoader`） |
| `MarkdownLoader` | `app/core/document/loader.py` | Markdown / TXT 加载（依赖 `TextLoader`） |

**调用链**:
```
DocumentPipeline.process(file_path)
  └─> DocumentLoaderFactory.create(path)        # 按扩展名选 Loader
       └─> PdfLoader / DocxLoader / MarkdownLoader
            └─> loader.load(path)                # 返回 list[Document]
```

### 1.2 文本拆分

**目标**: 将长文档按语义边界切分为小块（chunk），便于后续嵌入和检索。

| 类名 | 文件 | 职责 |
|------|------|------|
| `DocumentSplitterFactory` | `app/core/document/splitter/__init__.py` | 按策略名称创建对应分块器 |
| `BaseDocumentSplitter` | `app/core/document/splitter/base.py` | 分块器抽象基类，参数校验 |
| `RecursiveSplitter` | `app/core/document/splitter/recursive.py` | 递归分块，按分隔符层级切分，保留语义边界（默认策略） |
| `SmartSplitter` | `app/core/document/splitter/smart.py` | 智能分块，优先级: 标题 → 段落 → 句子 → 字符 |
| `FixedLengthSplitter` | `app/core/document/splitter/fixed.py` | 固定长度分块，按字符位置硬切 |
| `TokenSplitter` | `app/core/document/splitter/token.py` | 按 Token 数分块（预留） |

**调用链**:
```
DocumentPipeline.process(file_path)
  └─> self._splitter.split(documents)            # 返回 list[Document] (chunks)
       └─> RecursiveSplitter.split()
            └─> RecursiveCharacterTextSplitter.split_documents()
```

### 1.3 文本嵌入

**目标**: 将文本块转换为高维向量（embedding），支持多供应商。

| 类名 / 函数 | 文件 | 职责 |
|-------------|------|------|
| `EmbeddingFactory` | `app/core/embedding/__init__.py` | 工厂类，按供应商创建 Embedding 实例 |
| `create_deepseek_embedding()` | `app/core/embedding/api.py` | 创建 DeepSeek Embedding（OpenAI 兼容协议） |
| `create_xiaomi_embedding()` | `app/core/embedding/api.py` | 创建小米 MiMo Embedding |
| `create_local_embedding()` | `app/core/embedding/local.py` | 创建本地 HuggingFace Embedding（bge-small-zh） |
| `create_ollama_embedding()` | `app/core/embedding/ollama.py` | 创建 Ollama 本地模型 Embedding |

**调用链**:
```
main._init_components()
  └─> EmbeddingFactory.from_settings(settings)    # 根据 embedding_provider 配置选择
       └─> create_deepseek_embedding() / create_xiaomi_embedding() / ...
            └─> 返回 OpenAIEmbeddings / HuggingFaceEmbeddings / OllamaEmbeddings
```

### 1.4 存入向量数据库

**目标**: 将嵌入后的文档块存入 Chroma 向量数据库，支持版本管理和回滚。

| 类名 | 文件 | 职责 |
|------|------|------|
| `VectorStoreFactory` | `app/core/vectorstore/__init__.py` | 从配置创建 `VectorStoreManager` |
| `VectorStoreManager` | `app/core/vectorstore/manager.py` | Chroma 向量库核心管理器 |
| `SearchResult` | `app/core/vectorstore/models.py` | 语义检索结果数据模型 |
| `SourceInfo` | `app/core/vectorstore/models.py` | 来源文件信息 |
| `VersionInfo` | `app/core/vectorstore/models.py` | 版本记录信息 |

**`VectorStoreManager` 核心方法**:

| 方法 | 职责 |
|------|------|
| `add_documents()` | 写入文档块，生成确定性 ID |
| `similarity_search()` | 语义检索，只返回当前版本 |
| `upsert_documents()` | 增量更新（旧版本归档 + 新版本写入） |
| `rollback()` | 回滚到指定版本 |
| `delete_by_source()` | 删除某文件所有版本 |
| `rebuild()` | 全量重建 |

**调用链**:
```
document.py 中 upload_document()
  └─> vectorstore.upsert_documents(filename, chunks)
       └─> VectorStoreManager.upsert_documents()
            ├─ _get_current_version(source)    # 查询当前版本号
            ├─ 旧版本标记 _is_current=False    # 归档
            └─ add_documents(new_chunks)       # 写入新版本
                 └─ Chroma.add_documents()     # 底层 Chroma 写入
```

### 离线阶段统一入口

| 类名 | 文件 | 职责 |
|------|------|------|
| `DocumentPipeline` | `app/core/document/__init__.py` | 统一处理管线：加载 → 分块 |

```python
# 使用方式
pipeline = DocumentPipeline(chunk_size=512, chunk_overlap=50)
chunks = pipeline.process("data/documents/example.pdf")  # 加载+分块
vectorstore.upsert_documents("example.pdf", chunks)       # 存入向量库
```

---

## 二、在线问答阶段 (Chat)

### 2.1 完整调用链

```
POST /api/chat
  └─> chat_endpoint()                            # app/api/routes/chat.py
       └─> chain.invoke(question, chat_history)   # app/core/chain/rag_chain.py
            │
            ├─ _retrieve(question)                # 向量检索
            │    ├─ vectorstore.similarity_search()  # 召回候选
            │    ├─ score_threshold 过滤             # 阈值过滤
            │    └─ reranker.rerank() (可选)          # 重排序
            │
            ├─ _build_context(results)            # 拼接上下文
            │    └─ CONTEXT_ITEM 模板格式化
            │
            ├─ _build_messages()                  # 组装消息
            │    ├─ SystemMessage (系统提示词)
            │    ├─ HumanMessage (对话历史, 可选)
            │    └─ HumanMessage (USER_TEMPLATE: context + question)
            │
            └─ llm.invoke(messages)               # 调用大模型
                 └─ 返回 RAGResult(answer, sources)
```

### 2.2 各步骤对应类名

| 步骤 | 类名 / 函数 | 文件 | 职责 |
|------|------------|------|------|
| 用户问题 | `ChatRequest` | `app/schemas/chat.py` | 请求模型（question + chat_history） |
| 向量检索 | `VectorStoreManager.similarity_search()` | `app/core/vectorstore/manager.py` | 语义相似度检索 |
| 阈值过滤 | `RAGChain._retrieve()` | `app/core/chain/rag_chain.py` | 按 score 过滤低质量结果 |
| 重排序 | `BaseReranker` / `CrossEncoderReranker` | `app/core/reranker/` | Cross-Encoder 精排 |
| 拼接上下文 | `RAGChain._build_context()` | `app/core/chain/rag_chain.py` | 将检索结果格式化为文本 |
| Prompt 模板 | `DEFAULT_SYSTEM_PROMPT` / `USER_TEMPLATE` | `app/core/chain/prompts.py` | 抗幻觉提示词设计 |
| 组装消息 | `RAGChain._build_messages()` | `app/core/chain/rag_chain.py` | 组装 LangChain 消息列表 |
| 调用 LLM | `ChatOpenAI` / `ChatOllama` | `main.py` 中初始化 | 大模型调用 |
| 生成回答 | `RAGResult` / `ChatResponse` | `rag_chain.py` / `schemas/chat.py` | 响应数据模型 |

### 2.3 Reranker（可选增强）

| 类名 | 文件 | 职责 |
|------|------|------|
| `BaseReranker` | `app/core/reranker/base.py` | 重排序器抽象基类 |
| `CrossEncoderReranker` | `app/core/reranker/cross_encoder.py` | 基于 Cross-Encoder 的精排（BAAI/bge-reranker-v2-m3） |

---

## 三、配置管理

| 类名 | 文件 | 职责 |
|------|------|------|
| `Settings` | `app/config.py` | pydantic-settings 统一配置，从 `.env` 注入 |

**关键配置项**:

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `llm_provider` | `deepseek` | LLM 供应商选择 |
| `embedding_provider` | `deepseek` | Embedding 供应商选择 |
| `chunk_size` | `512` | 分块大小 |
| `chunk_overlap` | `50` | 分块重叠 |
| `search_top_k` | `4` | 检索返回数量 |
| `chroma_persist_dir` | `data/chroma_db` | Chroma 持久化目录 |
| `chroma_collection_name` | `knowledge_base` | Chroma 集合名 |

---

## 四、应用启动流程

```
main.py
  └─> _init_components()                    # 初始化所有 RAG 组件
       ├─ EmbeddingFactory.from_settings()  # 创建 Embedding
       ├─ VectorStoreFactory.from_settings()# 创建向量库管理器
       ├─ DocumentPipeline()                # 创建文档处理管线
       ├─ ChatOpenAI / ChatOllama()         # 创建 LLM 实例
       └─ RAGChainFactory.create()          # 创建 RAG 链
  └─> lifespan() → app.state.rag = {...}    # 挂载到 FastAPI app.state
  └─> app.include_router(chat.router)       # 注册聊天路由
  └─> app.include_router(document.router)   # 注册文档管理路由
```

---

## 五、API 接口

| 接口 | 路由 | 入口函数 | 调用的 RAG 组件 |
|------|------|---------|----------------|
| 知识库问答 | `POST /api/chat` | `chat_endpoint()` | `RAGChain.invoke()` |
| 上传文档 | `POST /api/documents/upload` | `upload_document()` | `DocumentPipeline.process()` + `VectorStoreManager.upsert_documents()` |
| 列出文档 | `GET /api/documents/` | `list_documents()` | `VectorStoreManager.get_sources()` |
| 删除文档 | `DELETE /api/documents/{filename}` | `delete_document()` | `VectorStoreManager.delete_by_source()` |
