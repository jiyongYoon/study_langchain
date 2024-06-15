from langserve import RemoteRunnable

# ngrok remote 주소 설정
ngrok_url = 'https://62a6-220-75-173-230.ngrok-free.app'

# chat
chat = ngrok_url + '/prompt'
chain = RemoteRunnable(chat)

for token in chain.stream({"topic": "딥러닝에 대해서 알려줘"}):
    print(token, end="")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LLM
llm_url = ngrok_url + '/llm'
llm = RemoteRunnable(llm_url)

prompt = ChatPromptTemplate.from_template(
    "다음 내용을 SNS 게시글 형식으로 변환해주세요:\n{input}"
)

message: str = "deep-learning is too difficult"

chain = prompt | llm
invoke = chain.invoke({"input": message})
print(invoke)
