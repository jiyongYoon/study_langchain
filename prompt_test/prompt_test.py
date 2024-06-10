### 일반 PromptTemplate은 아래와 같이 생성해서 사용 가능
from langchain.prompts.prompt import PromptTemplate


template = """
너는 요리사야. 내가 가진 재료들을 가지고 만들 수 있는 요리를 추천하고, 그 요리의 레시피를 제시해줘.
내가 가진 재료는 아래와 같아.

<재료>
{재료}

"""

prompt_template = PromptTemplate(
    input_variables=['재료'],
    template=template
)

print(prompt_template.format_prompt(재료 = '토마토, 계란, 식빵'))

### chatGPT 프롬프트는 아래와 같이 생성해서 사용 가능
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from gpt_llm import openai_chatgpt_llm


chatgpt = openai_chatgpt_llm.generate_llm_streaming_ver()

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "{재료}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

message = chatgpt(chat_prompt.format_prompt(재료="토마토, 계란, 식빵").to_messages())
print(message.content)