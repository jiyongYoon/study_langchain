# Build a sample vectorDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load blog post
loader = WebBaseLoader("https://n.news.naver.com/mnews/article/003/0012317114?sid=105")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma.from_documents(documents=splits, embedding=ko_embedding)


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

question = "삼성전자 갤럭시 S24는 어떨 예정이야?"
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.environ['OPEN_API_KEY']
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm
)

# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
""" MultiQueryRetriever가 사용자 질문으로 재생성한 질문
INFO:langchain.retrievers.multi_query:Generated queries: [
    '1. 삼성 갤럭시 S24 출시 일정은 언제인가요?', 
    '2. 삼성 갤럭시 S24의 기능과 스펙은 어떻게 되나요?', 
    '3. 삼성 갤럭시 S24의 가격대는 어느 정도일까요?'
]
"""

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print(len(unique_docs))
print(unique_docs)
"""
[
    Document(
        page_content="[서울=뉴시스] 삼성전자가 17일 오전 10시(현지시간, 한국 시간 18일 오전 3시) 미국 캘리포니아주 산호세(새너제이)에서 '삼성 갤럭시 언팩 2024'를 열고 갤럭시 S24를 공개한다. 사진은 포르투갈에서 유출된 갤럭시 S24 시리즈 포스터 추정 이미지 (사진=theonecid 엑스 캡처)  *재판매 및 DB 금지[서울=뉴시스]윤정민 기자 = 인공지능(AI) 서비스가 대거 탑재될 삼성전자 플래그십 스마트폰 '갤럭시 S24'가 18일 베일을 벗는다. 갤럭시 S23이 전작 대비 카메라, 디자인 등 대폭 개선됐다면, 이번 신작은", 
        metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"}), 
    Document(
        page_content="[서울=뉴시스] 삼성전자가 17일 오전 10시(현지시간, 한국 시간 18일 오전 3시) 미국 캘리포니아주 산호세(새너제이)에서 열 '삼성 갤럭시 언팩 2024'의 주제는 '모바일 AI의 새로운 시대 개막'이다. 앞서 삼성전자가 AI를 스마트폰 차기작 특징으로 예고했던 만큼 어떤 AI 기능이 실릴지 관심이 쏠린다.삼성전자가 공식적으로 밝힌 AI 서비스는 실시간 통화 통역이다. 이미 SK텔레콤 '에이닷' 등 통화 통역을 지원하는 앱이 있다. 하지만 자체 AI가 탑재될 갤럭시 S24는 별도 앱을 설치하지 않아도 통역 통화를 이용할 수", 
        metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"}), 
    Document(
        page_content="[서울=뉴시스] 8일 업계에 따르면 삼성전자는 미국 삼성닷컴 새 스마트폰 사전예약 알림 창구에 '갤럭시 AI와 함께하는 줌이 온다(Zoom with Galaxy AI is coming)'을 주제로 한 영상을 게재했다. (영상=미국 삼성닷컴 캡처) *재판매 및 DB 금지 *재판매 및 DB 금지AI를 통해 사진 속 일부 물체 크기를 더 키울 수 있는 기능도 나올 가능성이 있다. 삼성전자는 미국 삼성닷컴 새 스마트폰 사전 예약 알림 창구에 '갤럭시 AI와 함께하는 줌이 온다'를 주제로 한 영상을 게재했다. 나인투파이브구글 등 해외 IT", 
        metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"}), 
    Document(
        page_content='닫기\n\n\n \n\n글자 크기 변경하기\n\n\n\n가1단계\n작게\n\n\n가2단계\n보통\n\n\n가3단계\n크게\n\n\n가4단계\n아주크게\n\n\n가5단계\n최대크게\n\n\n\n\n\n\nSNS 보내기\n\n\n\n인쇄하기\n\n\n\n\n\n\n\n\n삼성전자, 17일(美 현지시간) 언팩서 갤럭시 S24 시리즈 공개국내 출고가 일반·플러스 동결, 울트라 약 10만원 인상 전망실시간 통화 통역, 이미지 자동 편집 등 AI 기능 탑재 예상',
        metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"}), 
    Document(
        page_content='[서울=뉴시스] 삼성전자 갤럭시 S24 시리즈 예상 색상 유출본 (사진=인터넷 커뮤니티 캡처) *재판매 및 DB 금지외관의 경우 갤럭시 S24 울트라 모델이 전작과 달리 티타늄 소재를 썼다는 데 눈에 띈다. 티타늄은 알루미늄보다 무게가 무겁지만 내구성이 강하다.스마트폰 두뇌 역할을 하는 모바일 애플리케이션 프로세서(AP)에는 퀄컴 스냅드래곤8 3세대와 삼성 엑시노스 2400이 적용될 것으로 보인다. 국내 시장에 출시될 제품의 경우 퀄컴 칩은 울트라 모델에만 탑재되고 일반과 플러스 모델에는 엑시노스 칩이 실릴 전망이다.작업 처리에', 
        metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"}), 
    Document(
        page_content='한층 진화된 AI 서비스로 새로운 스마트폰 기준을 제시할 것으로 전망된다.업계 모두가 스마트폰 시장 선두 주자인 삼성전자의 신작을 궁금해하는 만큼 공신력 있는 IT 팁스터(정보유출자)들이 여러 정보를 유출했다. 14일 현재까지 삼성전자가 공개한 스마트폰 신작 정보와 업계에 유출된 내용을 종합해 갤럭시 S24 예상 사양을 정리해 봤다."손안에 만능 비서 담았다"…인터넷 연결 없어도 AI 쓸 수 있는 \'온디바이스 AI폰\'', 
        metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"}), 
    Document(page_content='기준 오닉스 블랙, 마블 그레이, 코발트 바이올렛, 앰버 옐로우 등 4가지이며 울트라는 티타늄 블랙, 티타늄 그레이, 티타늄 바이올렛, 티타늄 옐로우 등 4가지가 될 것으로 예상된다. 이 밖에 삼성닷컴 판매 한정으로 티타늄 그린, 티타늄 블루, 티타늄 오렌지도 추가될 전망이다.국내 출고가 일반·플러스 동결, 울트라 약 10만원 인상될 듯…19일 사전예약소비자 입장에서 제일 궁금한 건 가격과 사전 예약 혜택이다.일반형, 플러스 모델(256GB 기준) 가격은 전작과 같을 것으로 전망된다. 256GB 용량 기준 일반형과 플러스 모델', metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"}), Document(page_content='예상 출고가는 각각 115만5000원, 135만3000원으로 잠정 책정된 것으로 알려졌다. 512GB의 경우 전작 대비 2만2000원 비싸진 129만8000원, 149만6000원이 될 전망이다.울트라 모델은 전작보다 비싸질 것으로 알려졌다. 256GB는 9만9000원 오른 169만8400원, 512GB는 12만1000원 오른 184만1400원으로 예상된다.사전 예약 기간은 19일부터 25일까지 7일간 진행될 것으로 예상되며 사전 예약 혜택은 더블 스토리지, 갤럭시 워치 할인, 갤럭시 버즈 FE 할인 등이 거론되고 있다.더블', metadata={'language': 'ko', 'source': 'https://n.news.naver.com/mnews/article/003/0012317114?sid=105', 'title': "언팩 D-4, 세계 최초 AI폰 '갤S24' 이렇게 나온다"})]
"""
