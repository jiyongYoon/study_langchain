import json


def print_schema(schema):
    print(json.dumps(schema, indent=4))


input_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "age"],
}

print_schema(input_schema)

###########################################
from llm import openai_chatgpt_llm
from langchain.prompts import ChatPromptTemplate


model = openai_chatgpt_llm.generate_llm()
prompt = ChatPromptTemplate.from_template("{topic}에 대하여 3문장으로 설명해줘.")

chain = prompt | model

###########################################
# schema
print(chain.input_schema.schema())
"""
{'title': 'PromptInput', 'type': 'object', 'properties': {'topic': {'title': 'Topic', 'type': 'string'}}}
"""
print(model.input_schema.schema())
"""
{'title': 'ChatOpenAIInput', 'anyOf': [{'type': 'string'}, {'$ref': '#/definitions/StringPromptValue'}, {'$ref': '#/definitions/ChatPromptValueConcrete'}, {'type': 'array', 'items': {'anyOf': [{'$ref': '#/definitions/AIMessage'}, {'$ref': '#/definitions/HumanMessage'}, {'$ref': '#/definitions/ChatMessage'}, {'$ref': '#/definitions/SystemMessage'}, {'$ref': '#/definitions/FunctionMessage'}, {'$ref': '#/definitions/ToolMessage'}]}}], 'definitions': {'StringPromptValue': {'title': 'StringPromptValue', 'description': 'String prompt value.', 'type': 'object', 'properties': {'text': {'title': 'Text', 'type': 'string'}, 'type': {'title': 'Type', 'default': 'StringPromptValue', 'enum': ['StringPromptValue'], 'type': 'string'}}, 'required': ['text']}, 'ToolCall': {'title': 'ToolCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'args': {'title': 'Args', 'type': 'object'}, 'id': {'title': 'Id', 'type': 'string'}}, 'required': ['name', 'args', 'id']}, 'InvalidToolCall': {'title': 'InvalidToolCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'args': {'title': 'Args', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'error': {'title': 'Error', 'type': 'string'}}, 'required': ['name', 'args', 'id', 'error']}, 'UsageMetadata': {'title': 'UsageMetadata', 'type': 'object', 'properties': {'input_tokens': {'title': 'Input Tokens', 'type': 'integer'}, 'output_tokens': {'title': 'Output Tokens', 'type': 'integer'}, 'total_tokens': {'title': 'Total Tokens', 'type': 'integer'}}, 'required': ['input_tokens', 'output_tokens', 'total_tokens']}, 'AIMessage': {'title': 'AIMessage', 'description': 'Message from an AI.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'ai', 'enum': ['ai'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'example': {'title': 'Example', 'default': False, 'type': 'boolean'}, 'tool_calls': {'title': 'Tool Calls', 'default': [], 'type': 'array', 'items': {'$ref': '#/definitions/ToolCall'}}, 'invalid_tool_calls': {'title': 'Invalid Tool Calls', 'default': [], 'type': 'array', 'items': {'$ref': '#/definitions/InvalidToolCall'}}, 'usage_metadata': {'$ref': '#/definitions/UsageMetadata'}}, 'required': ['content']}, 'HumanMessage': {'title': 'HumanMessage', 'description': 'Message from a human.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'human', 'enum': ['human'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'example': {'title': 'Example', 'default': False, 'type': 'boolean'}}, 'required': ['content']}, 'ChatMessage': {'title': 'ChatMessage', 'description': 'Message that can be assigned an arbitrary speaker (i.e. role).', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'chat', 'enum': ['chat'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'role': {'title': 'Role', 'type': 'string'}}, 'required': ['content', 'role']}, 'SystemMessage': {'title': 'SystemMessage', 'description': 'Message for priming AI behavior, usually passed in as the first of a sequence\nof input messages.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'system', 'enum': ['system'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}}, 'required': ['content']}, 'FunctionMessage': {'title': 'FunctionMessage', 'description': 'Message for passing the result of executing a function back to a model.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'function', 'enum': ['function'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}}, 'required': ['content', 'name']}, 'ToolMessage': {'title': 'ToolMessage', 'description': 'Message for passing the result of executing a tool back to a model.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'tool', 'enum': ['tool'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'}}, 'required': ['content', 'tool_call_id']}, 'ChatPromptValueConcrete': {'title': 'ChatPromptValueConcrete', 'description': 'Chat prompt value which explicitly lists out the message types it accepts.\nFor use in external schemas.', 'type': 'object', 'properties': {'messages': {'title': 'Messages', 'type': 'array', 'items': {'anyOf': [{'$ref': '#/definitions/AIMessage'}, {'$ref': '#/definitions/HumanMessage'}, {'$ref': '#/definitions/ChatMessage'}, {'$ref': '#/definitions/SystemMessage'}, {'$ref': '#/definitions/FunctionMessage'}, {'$ref': '#/definitions/ToolMessage'}]}}, 'type': {'title': 'Type', 'default': 'ChatPromptValueConcrete', 'enum': ['ChatPromptValueConcrete'], 'type': 'string'}}, 'required': ['messages']}}}
"""
print(chain.output_schema.schema())
"""
{'title': 'ChatOpenAIOutput', 'anyOf': [{'$ref': '#/definitions/AIMessage'}, {'$ref': '#/definitions/HumanMessage'}, {'$ref': '#/definitions/ChatMessage'}, {'$ref': '#/definitions/SystemMessage'}, {'$ref': '#/definitions/FunctionMessage'}, {'$ref': '#/definitions/ToolMessage'}], 'definitions': {'ToolCall': {'title': 'ToolCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'args': {'title': 'Args', 'type': 'object'}, 'id': {'title': 'Id', 'type': 'string'}}, 'required': ['name', 'args', 'id']}, 'InvalidToolCall': {'title': 'InvalidToolCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, 'args': {'title': 'Args', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'error': {'title': 'Error', 'type': 'string'}}, 'required': ['name', 'args', 'id', 'error']}, 'UsageMetadata': {'title': 'UsageMetadata', 'type': 'object', 'properties': {'input_tokens': {'title': 'Input Tokens', 'type': 'integer'}, 'output_tokens': {'title': 'Output Tokens', 'type': 'integer'}, 'total_tokens': {'title': 'Total Tokens', 'type': 'integer'}}, 'required': ['input_tokens', 'output_tokens', 'total_tokens']}, 'AIMessage': {'title': 'AIMessage', 'description': 'Message from an AI.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'ai', 'enum': ['ai'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'example': {'title': 'Example', 'default': False, 'type': 'boolean'}, 'tool_calls': {'title': 'Tool Calls', 'default': [], 'type': 'array', 'items': {'$ref': '#/definitions/ToolCall'}}, 'invalid_tool_calls': {'title': 'Invalid Tool Calls', 'default': [], 'type': 'array', 'items': {'$ref': '#/definitions/InvalidToolCall'}}, 'usage_metadata': {'$ref': '#/definitions/UsageMetadata'}}, 'required': ['content']}, 'HumanMessage': {'title': 'HumanMessage', 'description': 'Message from a human.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'human', 'enum': ['human'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'example': {'title': 'Example', 'default': False, 'type': 'boolean'}}, 'required': ['content']}, 'ChatMessage': {'title': 'ChatMessage', 'description': 'Message that can be assigned an arbitrary speaker (i.e. role).', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'chat', 'enum': ['chat'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'role': {'title': 'Role', 'type': 'string'}}, 'required': ['content', 'role']}, 'SystemMessage': {'title': 'SystemMessage', 'description': 'Message for priming AI behavior, usually passed in as the first of a sequence\nof input messages.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'system', 'enum': ['system'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}}, 'required': ['content']}, 'FunctionMessage': {'title': 'FunctionMessage', 'description': 'Message for passing the result of executing a function back to a model.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'function', 'enum': ['function'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}}, 'required': ['content', 'name']}, 'ToolMessage': {'title': 'ToolMessage', 'description': 'Message for passing the result of executing a tool back to a model.', 'type': 'object', 'properties': {'content': {'title': 'Content', 'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'object'}]}}]}, 'additional_kwargs': {'title': 'Additional Kwargs', 'type': 'object'}, 'response_metadata': {'title': 'Response Metadata', 'type': 'object'}, 'type': {'title': 'Type', 'default': 'tool', 'enum': ['tool'], 'type': 'string'}, 'name': {'title': 'Name', 'type': 'string'}, 'id': {'title': 'Id', 'type': 'string'}, 'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'}}, 'required': ['content', 'tool_call_id']}}}
"""

###########################################
# Stream: 실시간 출력
for s in chain.stream({"topic": "멀티모달"}):
    # 스트림에서 받은 데이터의 내용을 출력, 줄바꿈 없이 이어서 출력하며, 버퍼를 즉시 비움
    print(s.content, end="", flush=True)

###########################################
# invoke: 처리 수행
invoke_response = chain.invoke({"topic": "Samsung"})
print(invoke_response.content)

###########################################
# batch: 단위 실행 - 일괄 처리

# chain.batch([{"topic": "반도체"}, {"topic": "달러 가치"}])

batch_response = chain.batch(
    [
        {"topic": "파이썬"},
        {"topic": "자바"},
        {"topic": "코틀린"},
        {"topic": "러스트"},
    ],
    # 배치 작업에서 동시에 처리할 수 있는 최대 작업 수
    config={"max_concurrency": 2}
)
print(batch_response)
"""
[
AIMessage(content='파이썬은 인터프리터 방식의 고수준 프로그래밍 언어로, 문법이 간결하고 읽기 쉽다. 다양한 운영 체제에서 사용할 수 있으며, 다양한 라이브러리와 모듈을 제공하여 다양한 응용 프로그램을 개발할 수 있다. 파이썬은 데이터 분석, 인공지능, 웹 개발 등 다양한 분야에서 널리 사용되고 있다.', response_metadata={'token_usage': {'completion_tokens': 145, 'prompt_tokens': 27, 'total_tokens': 172}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-43732b9b-09e7-40fe-b53f-43242eefd27e-0', usage_metadata={'input_tokens': 27, 'output_tokens': 145, 'total_tokens': 172}), 
AIMessage(content='자바는 객체지향 프로그래밍 언어로, 다양한 플랫폼에서 사용되는 범용 프로그래밍 언어이다. 자바는 가상머신을 통해 다양한 운영체제에서 동작할 수 있으며, 안정성과 이식성이 뛰어나다. 자바는 다양한 라이브러리와 프레임워크를 제공하여 개발자들이 효율적으로 소프트웨어를 개발할 수 있도록 도와준다.', response_metadata={'token_usage': {'completion_tokens': 163, 'prompt_tokens': 24, 'total_tokens': 187}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-5335c665-1987-40a4-95fd-80285f2632c2-0', usage_metadata={'input_tokens': 24, 'output_tokens': 163, 'total_tokens': 187}), 
AIMessage(content='코틀린은 JetBrains에서 개발한 JVM 언어로, 자바와의 상호운용성이 뛰어나다. 간결하고 실용적인 문법을 제공하여 개발자들이 생산성을 높일 수 있으며, 안드로이드 앱 개발에도 널리 사용된다. 함수형 프로그래밍과 객체지향 프로그래밍을 모두 지원하여 다양한 프로그래밍 스타일을 적용할 수 있다.', response_metadata={'token_usage': {'completion_tokens': 142, 'prompt_tokens': 28, 'total_tokens': 170}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-f70e2861-9be8-45d5-be53-5e6cb4368706-0', usage_metadata={'input_tokens': 28, 'output_tokens': 142, 'total_tokens': 170}), 
AIMessage(content='러스트는 안전하고 병렬 처리가 용이한 시스템 프로그래밍 언어이다. 메모리 안정성을 보장하며 높은 성능을 제공한다. 생산성과 안정성을 동시에 갖춘 언어로 많은 개발자들에게 인기가 있다.', response_metadata={'token_usage': {'completion_tokens': 93, 'prompt_tokens': 23, 'total_tokens': 116}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-123d9b18-fb32-44a7-b8fc-0575187fd8fe-0', usage_metadata={'input_tokens': 23, 'output_tokens': 93, 'total_tokens': 116})
]
"""

###########################################
# async: 비동기
## async stream
async for s in chain.astream({"topic": "Naver"}):
    print(s.content, end="\n", flush=True)

## async invoke
await chain.ainvoke({"topic": "NVDA"})

## async batch
await chain.abatch(
    [{"topic": "YouTube"}, {"topic": "Instagram"}, {"topic": "Facebook"}]
)

## async stream 중간단계 디버깅
