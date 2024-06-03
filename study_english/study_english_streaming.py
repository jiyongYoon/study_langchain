from llm import openai_chatgpt_llm

model = openai_chatgpt_llm.generate_llm_streaming_ver()

answer = model.predict("AI 엔지니어링에 대해서 학습하고 싶은데, 어떤 순서로 학습을 하면 좋을지 알려줘")
print(answer)