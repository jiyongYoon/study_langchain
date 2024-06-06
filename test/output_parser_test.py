from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from llm import openai_chatgpt_llm
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

### 출력값을 내가 원하는 형태로 고정 가능
output_parser = CommaSeparatedListOutputParser()
comma_format_instructions = output_parser.get_format_instructions()
print(comma_format_instructions)
"""
Your response should be a list of comma separated values, eg: `foo, bar, baz` or `foo,bar,baz`
"""

prompt = PromptTemplate(
    template="{주제} 5개를 추천해줘.\n{format_instructions}",
    input_variables=["주제"],
    partial_variables={"format_instructions": comma_format_instructions}
)

# model = openai_chatgpt_llm.generate_llm()
model = OpenAI(
    temperature=0,
    api_key=os.environ['OPEN_API_KEY'],
)

_input = prompt.format(주제="자동차")
output = model(_input)

print(output)
print(output_parser.parse(output))