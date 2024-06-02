from langchain.prompts import PromptTemplate


def generate_prompt_template():
    template = """
    당신은 영어를 가르치는 10년차 영어 선생님입니다. 상황에 대한 일반적인 회화내용을 총 10번 이내의 대화 내용으로 구성해서 [FORMAT]에 영어 회화를 작성해 주세요.
    
    상황:
    {question}
    
    FORMAT:
    - 영어 회화:
    - 한글 해석:
    """

    return PromptTemplate.from_template(template)


