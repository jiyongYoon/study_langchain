from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("대한민국의 수도는 어디야?")
print(prompt)
"""
input_variables=[] template='대한민국의 수도는 어디야?'
"""

prompt = PromptTemplate.from_template("{country}의 수도는 어디야?")
print(prompt)
"""
input_variables=['country'] template='{country}의 수도는 어디야?'
"""

from gpt_llm import openai_chatgpt_llm

model = openai_chatgpt_llm.generate_llm()

#################

chain = prompt | model
# chain.invoke()
""" input 값이 빠졌음!
TypeError: RunnableSequence.invoke() missing 1 required positional argument: 'input'
"""
answer = chain.invoke({"country": "멕시코"})
print(answer)
"""
content='멕시코의 수도는 멕시코시티입니다.' response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 21, 'total_tokens': 41}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-cf09fb59-f176-4b2f-ba9a-db23042777fb-0' usage_metadata={'input_tokens': 21, 'output_tokens': 20, 'total_tokens': 41}
"""

########## Runnable에 대해 알아보자 ###########

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
########## RunnablePassthrough ###########

# chain.invoke("멕시코") # 에러
# 이렇게 변수를 주고자 할 때, dictionary 값으로 주지 않고 바로 할당할 수 있도록 해주는 문법
# 필요한 변수들을 여러 형태로 세팅할 수 있다.
passthrough_chain = {"country": RunnablePassthrough()} | prompt | model
answer = passthrough_chain.invoke("멕시코")
print(answer)
"""
content='멕시코의 수도는 멕시코시티(Mexico City)입니다.' response_metadata={'token_usage': {'completion_tokens': 25, 'prompt_tokens': 21, 'total_tokens': 46}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-d59ef9aa-108e-4e16-a6d1-8eb26d76fdb6-0' usage_metadata={'input_tokens': 21, 'output_tokens': 25, 'total_tokens': 46}
"""
# 2개라면
# chain = {"document": retriever, "question": RunnablePassthrough()} | prompt | model
# 이런 식으로 가능

########## RunnableParallel ###########

# # 병렬 처리를 가능한게 해줌

prompt2 = PromptTemplate.from_template("{country}의 인구는 몇명이야?")

chain1 = {"country": RunnablePassthrough()} | prompt | model
chain2 = {"country": RunnablePassthrough()} | prompt2 | model

map_chain = RunnableParallel(a=chain1, b=chain2)
answer = map_chain.invoke("대한민국")
print(answer)
"""
{
    'a': AIMessage(content='대한민국의 수도는 서울이다.', response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 22, 'total_tokens': 38}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-e0f47726-b162-4cb7-9c2b-a30bbbdb0d67-0', usage_metadata={'input_tokens': 22, 'output_tokens': 16, 'total_tokens': 38}), 
    'b': AIMessage(content='2021년 7월 기준 대한민국의 인구는 약 5천 130만 명입니다.', response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 23, 'total_tokens': 57}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-02aeb56b-7499-4999-9b36-7eaf3b3ef7c4-0', usage_metadata={'input_tokens': 23, 'output_tokens': 34, 'total_tokens': 57})
}
"""

########## RunnableLambda ###########
# # chain의 결과값을 내가 만든 함수에 전달함

def combine_text(text):
    return text['a'].content + ' ' + text['b'].content


combine_prompt = PromptTemplate.from_template("다음의 내용을 자연스럽게 교정하고, 내용이 지루하지 않도록 이모티콘을 적절하게 추가해줘:\n{info} ")
print(combine_prompt)
"""
input_variables=['info'] template='다음의 내용을 자연스럽게 교정하고, 내용이 지루하지 않도록 이모티콘을 적절하게 추가해줘:\n{info} '
"""

# map_chain의 결과값이  combine_text의 파라미터로 들어가고,
# 해당 결과값이 "info"라는 key값을 가진 dictionary의 value로 들어간 후,
# combine_prompt와 model chain을 타게 된다
final_chain = (
        map_chain
        | {"info": RunnableLambda(combine_text)}
        | combine_prompt
        | model
)

answer = final_chain.invoke("대한민국")
print(answer)
"""
content='대한민국의 수도는 서울이며, 2021년 7월 기준 대한민국의 인구는 약 5천 130만 명이에요! 🇰🇷🏙️' response_metadata={'token_usage': {'completion_tokens': 64, 'prompt_tokens': 110, 'total_tokens': 174}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-ce3c55b6-dcb4-4dad-904c-4554745a54ff-0' usage_metadata={'input_tokens': 110, 'output_tokens': 64, 'total_tokens': 174}
"""
