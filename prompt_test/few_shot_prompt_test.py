### Few-shot 프롬프트 예제 제공
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


examples = [
  {
    "question": "아이유로 삼행시 만들어줘",
    "answer":
    """
    아: 아이유는
    이: 이런 강의를 들을 이
    유: 유가 없다.
    """
  },

  {
    "question": "김민수로 삼행시 만들어줘",
    "answer":
    """
    김: 김치는 맛있다
    민: 민달팽이도 좋아하는 김치!
    수: 수억을 줘도 김치는 내꺼!
    """
  }
]

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

print(example_prompt.format(**examples[0]))

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"]
)

print(prompt.format(input="호날두로 삼행시 만들어줘"))


from gpt_llm import openai_chatgpt_llm


model = openai_chatgpt_llm.generate_llm_streaming_ver()
print(model.invoke(
        input="호날두로 삼행시 만들어줘",
    ).content,
    end=""
)


chain = prompt | model
print(chain.invoke(
        input="호날두로 삼행시 만들어줘",
    ).content,
    end=""
)