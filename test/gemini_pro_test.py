from gemini_llm import gemini_llm


llm = gemini_llm.generate_gemini_pro_llm()
# result = llm.invoke("엔비디아에 대해 보고서를 작성해줘")
# print(result.content)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from custom_util import read_pdf

clean_text = read_pdf.read_pdf_to_clean_text('nvidia.pdf')


from langchain.schema import Document

pages = [Document(page_content=clean_text)]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

texts = text_splitter.split_documents(pages)


from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

docsearch = Chroma.from_documents(texts, hf)

retriever = docsearch.as_retriever(
    search_type="mmr", # 최대한 다양한 답변을 위한 다양한 Chunk들를 불러오기
    search_kwargs={'k': 3, 'fetch_k': 10})
documents = retriever.get_relevant_documents("엔비디아의 2024년 영업이익률의 평균을 알려줘")

print(documents)
"""
[
Document(page_content='2024 2 22\nTech팀 엔비디아\nNVDA US\n문준호\nSenior Analyst\nFY 4Q24 review  EASY\njoonhomoonsamsungcom\n 실적가이던스 서프라이즈 지속\n 신제품 전망도 긍정적 차기 아키텍처 B100은 출시 전부터 공급 제약을 언급\n 최근까지도 높아진 컨센서스의 추가 상향이 기대되는 한편 valuation 매력도 부각\nWHATS THE STORY\nFY 4Q24 review 4분기에도 어닝 서프라이즈 기록 매출액은 전년 동기 대비 265 전분기\n AT A GLANCE 대비 22 증가하며 컨센서스를 8 상회 Data Center 매출액은 전년 동기 대비 324 전\n현재주가 67472 USD 분기 대비 25 성장 수출 규제로 중국 비중이 2025에서 한 자릿수 중반까지 급감했음\n블룸버그 평균목표주가 73448 USD 에도 전 지역산업에서 수요가 강력했다고 설명 특히 회사는 작년 연간 Data Center 매출'), 
Document(page_content='자료 NVIDIA\nNVIDIA 데이터센터 로드맵\n2021 2023 2024 2025\nGH200NVL GB200NVL GX200NVL Arm Training Inference\nGH200NVL\nGH200 GB200 GX200 Arm Inference\nAmpere Hopper Blackwell\nA100 H100 H200 B100 X100 X86 Training  Inference\nL4OS B40 X40 X86 Enterprise  Inference\n자료 NVIDIA\n12개월 forward PE 추이 SOX 대비 valuation premium 추이\n배 배 \n70 70 180\n160\n60\n60 2 SD 140\n50 120\n50 1 SD 40 100\n80\n40 평균 30 60\n40\n20\n30 1 SD 20\n10 0\n20 2 SD 1902 2002 2102 2202 2302 2402\n프리미엄우측 NVIDIA 좌측\n10\n1902 2002 2102 2202 2302 2402 SOX 좌측'), 
Document(page_content='순부채 5434 8254 3843 9521 1265\n현금흐름표 재무비율 및 주당지표\n1월 31일 기준 백만달러 2019 2020 2021 2022 2023 1월 31일 기준 2019 2020 2021 2022 2023\n영업활동에서의 현금흐름 3743 4761 5822 9108 5641 증감률 \n당기순이익 4141 2796 4332 9752 4368 매출액 206 68 527 614 02\n현금유출입이없는 비용 및 수익 455 949 1258 1737 2188 영업이익 185 252 592 1216 579\n유무형자산 감가상각비 262 381 1098 1174 1544 순이익 359 325 549 1251 552\n기타 193 568 160 563 644 수정 EPS 350 128 727 776 248\n영업활동 자산부채 변동 1115 635 866 3555 2459 주당지표')
]
"""


from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap

template = """Answer the question as based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)



gemini = gemini_llm.generate_gemini_pro_llm()

chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x['question']),
    "question": lambda x: x['question']
}) | prompt | gemini

print(chain.invoke({'question': 'nvidia의 강점에 대해서 설명해줘'}).content)