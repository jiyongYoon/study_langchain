from llm import openai_chatgpt_llm, output_parser
import study_english_prompt_template


prompt = study_english_prompt_template.generate_prompt_template()
model = openai_chatgpt_llm.generate_llm()
output_parser = output_parser.output_parser()

chain = prompt | model | output_parser


# print(chain.invoke({"question": "저는 런던 시내에서 숙소를 구하고 싶어요"}))
# print(chain.invoke({"question": "저는 식당에 가서 음식을 주문하고 싶어요"}))
print(chain.invoke({"question": "저는 식당에 가서 음식을 주문하고 싶어요"}))
