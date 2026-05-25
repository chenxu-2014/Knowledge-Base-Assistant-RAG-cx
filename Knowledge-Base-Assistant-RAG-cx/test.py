#导入 dotenv 库的 load_dotenv 函数，用于加载环境变量文件（.env）中的配置
import dotenv
from langchain_openai import ChatOpenAI
import os

dotenv.load_dotenv()  #加载当前目录下的 .env 文件

os.environ['OPENAI_API_KEY'] = os.getenv("DEEPSEEK_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("DEEPSEEK_BASE_URL")

# 创建大模型实例
llm = ChatOpenAI(model="deepseek-v4-pro")  # 默认使用 gpt-3.5-turbo

# 直接提供问题，并调用llm
response = llm.invoke("什么是大模型？")
print("response"*10)
print(response)
print("response"*10)