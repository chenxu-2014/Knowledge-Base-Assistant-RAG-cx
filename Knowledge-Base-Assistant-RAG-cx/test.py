from langchain_openai import ChatOpenAI
import os
import dotenv

# 加载 .env 文件中的环境变量（API_KEY、BASE_URL 等）
dotenv.load_dotenv()
os.environ['XIAOMI_API_KEY'] = os.getenv('XIAOMI_API_KEY')
os.environ['XIAOMI_BASE_URL'] = os.getenv('XIAOMI_BASE_URL')

# ============ 创建基础对话模型 ============
# 用于普通对话，不涉及Agent多轮工具调用
chat_model = ChatOpenAI(
    model_name='mimo-v2-pro',
    base_url=os.environ['XIAOMI_BASE_URL'],
    api_key=os.getenv('XIAOMI_API_KEY'),
    temperature=0.7,
    max_tokens=2000,
)

# 创建大模型实例

# 直接提供问题，并调用llm
response = chat_model.invoke("你是什么大模型？")
print(response)
# ============ 创建 Agent 专用模型 ============
# 重要：小米 MiMo 模型默认开启 thinking 模式，
# Agent 内部多轮调用时需要禁用该模式，否则会报错：
# 'The reasoning_content in the thinking mode must be passed back to the API.'
agent_model = ChatOpenAI(
    model_name='mimo-v2-pro',
    base_url=os.environ['XIAOMI_BASE_URL'],
    api_key=os.getenv('XIAOMI_API_KEY'),
    temperature=0.7,
    max_tokens=2000,
    extra_body={'enable_thinking': False},  # 禁用thinking模式
)
print('='*50)
response = agent_model.invoke("你是什么大模型？")
print(response)
print('模型初始化完成！')