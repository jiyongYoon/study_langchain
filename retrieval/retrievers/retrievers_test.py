import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


### PDF
import os
import pdfplumber

# project_directory = 'D:\\dev_yoon\\py\\study_langchain'
project_directory = 'F:\\dev\\python\\study\\langchain'
pdf_filename = 'generate-aipdf.pdf'
pdf_filepath = os.path.join(project_directory, pdf_filename)

pdf_content = ""
try:
    with pdfplumber.open(pdf_filepath) as pdf:
        for page in pdf.pages:
            pdf_content += page.extract_text() + "\n"
    print("PDF 파일을 성공적으로 로드하고 텍스트를 추출했습니다.")
except FileNotFoundError:
    print(f"PDF 파일 '{pdf_filename}'을 찾을 수 없습니다.")
except Exception as e:
    print(f"PDF 파일을 읽는 중 오류 발생: {e}")


### 특수문자 제거
import re


def remove_special_characters(text):
    # 특수 문자 제거 (정규 표현식 사용)
    return re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)


cleaned_text = remove_special_characters(pdf_content)


### Document 형태로 변환
from langchain.schema import Document


pages = [Document(page_content=cleaned_text)]


from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=tiktoken_len
)
texts = text_splitter.split_documents(pages)

from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

docsearch = Chroma.from_documents(texts, hf)


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os


load_dotenv()


openai = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0,
    api_key=os.environ['OPEN_API_KEY'],
)

qa = RetrievalQA.from_chain_type(
    llm=openai,
    chain_type="stuff",
    retriever=docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 3, 'fetch_k': 10}),
    return_source_documents=True # 출처를 확인하기 위한 옵션
)

query = "생성형 AI에 대해서 우리나라의 역량을 설명해줘"
result = qa(query)
print(result)
"""
우리나라는 생성형 AI 분야에서 기술적 역량이 미국과 중국에 비해 큰 격차가 있는 상황입니다. 최근 5년간 생성형 AI 분야에서 발표된 논문을 분석한 결과, 우리나라는 전체 논문 수에서 5위를 차지하고 있습니다. 중국이 1위를 차지하고 미국이 2위를 차지하며, 우리나라는 2682건의 논문을 발표했습니다. 이는 우리나라가 생성형 AI 분야에서 활발한 연구를 진행하고 있음을 시사합니다.
{
    'query': '생성형 AI에 대해서 우리나라의 역량을 설명해줘', 
    'result': '우리나라는 생성형 AI 분야에서 기술적 역량이 미국과 중국에 비해 큰 격차가 있는 상황입니다. 최근 5년간 생성형 AI 분야에서 발표된 논문을 분석한 결과, 우리나라는 전체 논문 수에서 5위를 차지하고 있습니다. 중국이 1위를 차지하고 미국이 2위를 차지하며, 우리나라는 2682건의 논문을 발표했습니다. 이는 우리나라가 생성형 AI 분야에서 활발한 연구를 진행하고 있음을 시사합니다.', 
    'source_documents': [
        Document(page_content='생태계 조성을 위한 기반 마련 필요\n AI 기술 초격차를 위한 생성형 AI 기술개발 우수인력 확보 및 양성 강화\n 생성형 AI 관련 데이터 구축개방 컴퓨팅 파워 및 자원 제공 윤리 및 신뢰성 확보 등 산업생태계\n기반조성\n AI 관련 원천기술을 고도화하기 위한 산학연 협력 및 딥테크 창업기업 지원을 강화할 필요\n 생성형 AI에 대한 진흥을 강화하는 한편 장단기적으로 미래 위험 요인에 대응하기 위한 정책적\n조치를 검토할 필요\n AI 튜터 도입 딥페이크 등 잘못된 정보에 따른 미디어보호 등을 통해 AI의 전반적 활용을 강화\n 새로운 기능 도입에 따른 출시 전 테스트 강화 국가 안보 위험성 사전 검토 등을 통해 생성형\nAI의 안전성을 국가 차원에서 보장\n참고문헌\n글로벌 과학기술정책정보 서비스ST GPS 235호 이미지 인식 등 한층 고도화된 GPT4 업계\n도입 경쟁 활발 2023314\nKISTEP 사과플러스 제10호 생성형 AI로 인한 사회변화와 대응방향 2023331'), 
        Document(page_content='그림 1 텍스트 생성 및 대화형 AI 지형도\n 출처  데이브레이크인사이츠 AI타임스23320\n2 AI타임스23320\n 2 \n 글로벌시장조사기관인 CB insights가 발표232한 생성형 AI 관련 글로벌 스타트업 250개 중\n한국기업은 3개에 불과 미국기업이 다수 차지3\n 미국 126개 인도영국 14개 이스라엘 12개 캐나다 10개 프랑스 6개 중국호주일본 5개 네덜란드스페인\n4개 한국 3개로 13위를 차지\n 시각미디어 분야의 딥브레인AI 디오비스튜디오 클레온 등\n 기술적 역량 우리나라의 생성형 AI 관련 기술적 역량은 압도적 선두국가인 미국 중국과의 격차가\n크게 벌어진 상황\n 분석대상  최근 5년간20182022년 생성형 AI분야의 Web of Science Core Collection 중 book을 제외한 논문\n54899건\n 분석기관  클래리베이트233\n 최근 5년간 생성형 AI 분야에서 발표된 논문을 분석한 결과 우리나라의 전체 논문 수는 총\n2682건으로 전체 5위를 차지\n 중국 19318건1위 미국 11624건2위 인도 4058건3위 영국 3484건4위'), 
        Document(page_content='KISTEP 브리프\n생성형 AI 관련 주요 이슈 및\n정책적 시사점\n23413 과학기술정책센터 고윤미 심정민\n1 검토배경\n 세계 빅테크 기업에서 초거대 생성형 AI 서비스인 ChatGPT가 출시되면서 최근 전 세계적인 화두로 등장\n ChatGPT는 OpenAI에서 개발한 GPT35 아키텍처를 기반으로 한 대화형 인공지능임\n 22년 11월 35버전을 발표하였으며 단 4개월만에 GPT 40 공개233로 전 세계의 이목을 집중\n GPTGenerative Pretrained Transformer3206는 대규모 텍스트 데이터 셋을 사용하여 머신러닝을\n통해 문맥의 의미를 이해한 사전학습 후 특정한 태스크에 대해 파인튜닝을 수행하는 자연어 이해 및 생성능력을\n갖춘 인공지능 모델로 기존 버전에 비해 큰 주목을 받음\n ChatGPT로 촉발된 언어모델 기반 서비스의 가능성이 가시화됨에 따라 빅테크 기업들은 대규모\n언어모델1과 생성형 AI 챗봇 서비스 출시 계획을 발표\n 국내 대기업에서도 한국어 기반의 초거대 언어모델 개발을 적극 추진하고 있으며 다수의 스타트업\n들에서도 다양한 AI 서비스를 제공 중')
    ]
}
"""