from gpt_llm import openai_chatgpt_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신의 이름은 챗채래챗챗챗봇 입니다. 항상 한글로 답변해주세요."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | openai_chatgpt_llm.generate_llm()
