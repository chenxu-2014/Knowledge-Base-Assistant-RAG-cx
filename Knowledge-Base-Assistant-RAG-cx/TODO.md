1. Embedding 模型（影响最大）                                                            
                                                          
  当前用的 Ollama nomic-embed-text:v1.5 是通用英文模型，中文语义理解能力弱，是命中率低的主要原因。
  - 换用中文专用 embedding：bge-m3、m3e-large、或 DeepSeek 的 deepseek-embedding                                                                                                                                                                     
  - API embedding 通常比本地小模型效果好很多                           
                                                                                                                                                                                                                                                     
  2. 分块策略                                                   
                                                                                                                                                                                                                                                     
  当前是固定 512 字符切分，可能把完整的语义段落截断。                              
  - app/core/document/splitter/ 下已经有 smart.py、recursive.py 等多种策略，可以尝试换用
  - 按语义分块（段落/标题）比按字符数分块效果好

  3. 混合检索

  当前只有向量检索，缺少关键词匹配。
  - 加入 BM25 关键词检索 + 向量检索的混合方案
  - 对于专业术语、缩写等精确匹配场景，关键词检索更有效

  4. Reranker 重排序

  代码里已经有 reranker 模块（app/core/reranker/），但没有启用。
  - 开启 CrossEncoder reranker，对粗召回结果精排
  - 能有效过滤向量检索中的噪声结果

  5. 查询改写

  用户原始问题可能表述不精确。
  - 在检索前用 LLM 改写/扩展用户问题（如加入同义词、补全上下文）
  - 或者将一个问题拆成多个子问题分别检索

  6. 阈值调优

  当前 search_score_threshold=0.7 是硬编码的，不同 embedding 模型的分数分布差异很大。
  - 换 embedding 后需要重新测试确定阈值
  - 可以改为动态阈值或相对排序策略

  7. 文档处理

  当前支持 PDF/DOCX/MD/TXT，但对表格、图片中的文字没有处理。
  - 表格内容丢失会直接导致相关问答失败

  ---
  建议优先级：换 embedding 模型 → 开启 reranker → 调整分块策略，这三个改完命中率会有明显提升。要从哪个开始？