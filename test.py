import openai
from dotenv import load_dotenv

load_dotenv()


print(openai.__version__)


import os
from langchain_openai import ChatOpenAI


# 객체 생성
llm = ChatOpenAI(
    temperature=0.1, # 창의성
    max_tokens=2048, # 최대 토큰 수
    model="gpt-3.5-turbo", # 모델명
    api_key=os.environ['OPEN_API_KEY'],
)

question = "lang chain을 공부하려고 하는데, lang chain은 무엇입니까??"

print(f"[답변]: {llm.invoke(question)}")