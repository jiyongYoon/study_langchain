from gpt_llm import openai_chatgpt_llm
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def ask_to_motivator(human_message):
    llm = openai_chatgpt_llm.generate_llm_streaming_ver()

    messages = [
            # llm에 역할을 부여하는 SystemMessage
            SystemMessage(
                content="너는 동기부여를 잘 하는 10년차 개발자야. 질문에 대해 한글로 답변을 하면서 강한 동기부여를 줘."
            ),
            # 인간이 전달하는 HumanMessage
            HumanMessage(
                content=human_message
            ),
    ]

    # 답변으로 돌아오는것이 AIMessage
    return llm(messages)


question = "백엔드 개발자로 오래 일하고 싶은데, 어떤 역량을 기르면 좋을까? 주변 사람들이 다 나보다 잘하는 것 같아서 걱정돼"
answer = ask_to_motivator(question)


print(answer)
"""
content='너무 걱정하지 마세요! 
백엔드 개발자로서 오래 일하기 위해 중요한 역량은 여러 가지가 있지만, 
그 중에서도 가장 중요한 것은 지속적인 학습과 성장입니다. 
기술은 빠르게 변하고 발전하기 때문에 끊임없이 새로운 기술과 도구에 대해 배우고 적용하는 능력이 필요합니다.
\n\n또한, 문제 해결 능력과 창의성도 중요한 역량입니다. 
백엔드 개발은 복잡한 시스템을 다루기 때문에 문제가 발생했을 때 빠르게 해결할 수 있는 능력이 필요하며, 새로운 아이디어를 내어 기존 시스템을 개선하는 능력도 중요합니다.
\n\n마지막으로, 협업과 커뮤니케이션 능력도 중요합니다. 백엔드 개발은 다른 팀원들과의 협업이 필수적이기 때문에 원활한 커뮤니케이션과 팀원들과의 조화로운 협업이 가능해야 합니다.
\n\n자신을 믿고 끊임없는 노력과 열정으로 계속해서 성장해 나가면, 주변 사람들보다 나은 백엔드 개발자가 될 수 있을 거예요. 자신을 믿고 도전해보세요! 
함께 성장하는 모습이 멋진 개발자가 되는 비결입니다.화이팅!' 
response_metadata={'token_usage': {'completion_tokens': 461, 'prompt_tokens': 130, 'total_tokens': 591}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-37350f9e-0ea1-4b52-8a32-9cdcea368ad8-0' usage_metadata={'input_tokens': 130, 'output_tokens': 461, 'total_tokens': 591}
"""