from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore # 저장소에 Key - Value 형태로 Parent - Child로 묶기 위한 저장공간
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()
project_directory = os.environ['PROJECT_DIRECTORY']

pdf_filepath = os.path.join(project_directory, 'generate-aipdf.pdf')

loaders = [PyPDFLoader(pdf_filepath)]

docs = []
for loader in loaders:
    docs.extend(loader.load_and_split()) # docs 리스트에 extend 형태로 파라미터 값을 추가함
    # loader.load_and_split()을 하게 되면 기본적으로 pdf 페이지별로 쪼개지게 됨

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=ko_embedding
)
# The storage layer for the parent documents
inmemory_store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=inmemory_store, # ParentDocumentRetriever가 Parent - Child 관계를 InmemoryStore에 저장하게 한다.
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)
print(len(list(inmemory_store.yield_keys())))
"""
7
"""

question = '각 나라별 ai관련 기업의 개수'

sub_docs = vectorstore.similarity_search(question) # child_document가 나오게 됨

print("글 길이: {}\n\n".format(len(sub_docs[0].page_content)))
print(sub_docs[0].page_content)
"""
글 길이: 477

- 3-❍글로벌시장조사기관인 CB insights가 발표(’23.2.)한 생성형 AI 관련 글로벌 스타트업 250개 중
한국기업은 3개*에 불과, 미국기업이 다수 차지3)
※ 미국 126개, 인도/영국 14개, 이스라엘 12개, 캐나다 10개, 프랑스 6개, 중국/호주/일본 5개, 네덜란드/스페인
4개, 한국 3개로 13위를 차지
* 시각미디어 분야의 딥브레인AI, 디오비스튜디오, 클레온 등
□(기술적 역량) 우리나라의 생성형 AI 관련 기술적 역량은 압도적 선두국가인 미국, 중국과의 격차가
크게 벌어진 상황
▶ 분석대상 : 최근 5년간(2018~2022년) 생성형 AI분야의 Web of Science Core Collection 중 book을 제외한 논문 
54,899건
▶ 분석기관 : 클래리베이트(’23.3.)
❍최근 5년간 생성형 AI 분야에서 발표된 논문을 분석한 결과, 우리나라의 전체 논문 수는 총 
2,682건으로 전체 5위를 차지
"""

retrieved_docs = retriever.get_relevant_documents(question) # child_document의 parent_document가 나오게 됨

print("글 길이: {}\n\n".format(len(retrieved_docs[0].page_content)))
print(retrieved_docs[0].page_content)
"""
글 길이: 1010

---------------- 이 부분이 자식 Chunk 부분 -------------
- 3-❍글로벌시장조사기관인 CB insights가 발표(’23.2.)한 생성형 AI 관련 글로벌 스타트업 250개 중
한국기업은 3개*에 불과, 미국기업이 다수 차지3)
※ 미국 126개, 인도/영국 14개, 이스라엘 12개, 캐나다 10개, 프랑스 6개, 중국/호주/일본 5개, 네덜란드/스페인
4개, 한국 3개로 13위를 차지
* 시각미디어 분야의 딥브레인AI, 디오비스튜디오, 클레온 등
□(기술적 역량) 우리나라의 생성형 AI 관련 기술적 역량은 압도적 선두국가인 미국, 중국과의 격차가
크게 벌어진 상황
▶ 분석대상 : 최근 5년간(2018~2022년) 생성형 AI분야의 Web of Science Core Collection 중 book을 제외한 논문 
54,899건
▶ 분석기관 : 클래리베이트(’23.3.)
❍최근 5년간 생성형 AI 분야에서 발표된 논문을 분석한 결과, 우리나라의 전체 논문 수는 총 
2,682건으로 전체 5위를 차지
--------------------------------------------------------
※ 중국 19,318건(1위), 미국 11,624건(2위), 인도 4,058건(3위), 영국 3,484건(4위)
❍피인용 상위 1% 논문은 총 70건으로 조사대상국 중 7위 수준4)으로 1위(미국), 2위(중국)와 
격차가 큰 편 
※ 미국 691건(1위), 중국 565건(2위), 영국 144건(3위), 독일 107건(4위), 호주 93건(5위)
❍세부 기술분류별*로 살펴보면, 우리나라는 이미지 생성형 AI 분야의 논문이 가장 많이 게재되었고
다음으로 자연어처리(NLP), 비디오의 순
* 자연어처리(NLP), 음성, 이미지, 비디오, 신약개발 등 5개 세부 기술로 분류
-인용수 대비 상위 1% 논문은 이미지 생성형 AI 분야에 집중, 자연어 처리(NLP) 분야의 경우
전체 논문수에 비해 인용수 대비 상위 1% 논문 비중이 저조한 편
-미국은 자연어처리(NLP) 생성형 AI분야에 강점을 보유하고 있으며, 중국은 이미지 생성형 
AI 분야의 논문 수가 가장 높게 나타남
3)머니투데이(2023.2.22.)
4)클래리베이트 분석결과(’23.3.)
"""

### 본문의 Full_Chunk가 너무 길 때
# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=ko_embedding,
)
# The storage layer for the parent documents
inmemory_store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=inmemory_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter, # 이것도 같이 넣어줌
)

retriever.add_documents(docs)
print(len(list(inmemory_store.yield_keys()))) # parent_document의 키가 총 몇개인지 -> 13 : 13 * 4 정도의 비율로 인메모리에 저장하게 되겠다
"""
13
"""
sub_docs = vectorstore.similarity_search(question)
print(sub_docs[0].page_content)
print(len(sub_docs[0].page_content))
"""
- 3-❍글로벌시장조사기관인 CB insights가 발표(’23.2.)한 생성형 AI 관련 글로벌 스타트업 250개 중
한국기업은 3개*에 불과, 미국기업이 다수 차지3)
※ 미국 126개, 인도/영국 14개, 이스라엘 12개, 캐나다 10개, 프랑스 6개, 중국/호주/일본 5개, 네덜란드/스페인
4개, 한국 3개로 13위를 차지
185
"""
retrieved_docs = retriever.get_relevant_documents(question)
print(retrieved_docs[0].page_content)
print(len(retrieved_docs[0].page_content))
"""
- 3-❍글로벌시장조사기관인 CB insights가 발표(’23.2.)한 생성형 AI 관련 글로벌 스타트업 250개 중
한국기업은 3개*에 불과, 미국기업이 다수 차지3)
※ 미국 126개, 인도/영국 14개, 이스라엘 12개, 캐나다 10개, 프랑스 6개, 중국/호주/일본 5개, 네덜란드/스페인
4개, 한국 3개로 13위를 차지
* 시각미디어 분야의 딥브레인AI, 디오비스튜디오, 클레온 등
□(기술적 역량) 우리나라의 생성형 AI 관련 기술적 역량은 압도적 선두국가인 미국, 중국과의 격차가
크게 벌어진 상황
▶ 분석대상 : 최근 5년간(2018~2022년) 생성형 AI분야의 Web of Science Core Collection 중 book을 제외한 논문 
54,899건
▶ 분석기관 : 클래리베이트(’23.3.)
❍최근 5년간 생성형 AI 분야에서 발표된 논문을 분석한 결과, 우리나라의 전체 논문 수는 총 
2,682건으로 전체 5위를 차지
※ 중국 19,318건(1위), 미국 11,624건(2위), 인도 4,058건(3위), 영국 3,484건(4위)
❍피인용 상위 1% 논문은 총 70건으로 조사대상국 중 7위 수준4)으로 1위(미국), 2위(중국)와 
격차가 큰 편 
※ 미국 691건(1위), 중국 565건(2위), 영국 144건(3위), 독일 107건(4위), 호주 93건(5위)
❍세부 기술분류별*로 살펴보면, 우리나라는 이미지 생성형 AI 분야의 논문이 가장 많이 게재되었고
다음으로 자연어처리(NLP), 비디오의 순
* 자연어처리(NLP), 음성, 이미지, 비디오, 신약개발 등 5개 세부 기술로 분류
799
"""