from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# ChatOpenAI()

#此时调用的是本地的大模型。省略base_url、api-key
llm = ChatOllama(
    model = "qwen3.5:9b"
)


# llm.invoke("你好，请介绍一下你自己！")


messages = [
    HumanMessage(content="你好，请介绍一下你自己！")
]

response = llm.invoke(messages)
print(response.content)