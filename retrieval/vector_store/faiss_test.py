import tiktoken


tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


### PDF
import os
from langchain_community.document_loaders import PyPDFLoader
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

# split it into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=tiktoken_len
)

docs = text_splitter.split_documents(pages)

from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
ko = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# faiss 저장
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(docs, ko)


query = "생성형 AI 관련한 주요 기업과 해당 기업의 생성형 AI의 모델명을 알려줘"
docs = db.similarity_search(query)
print(docs[0].page_content)


docs_and_scores = db.similarity_search_with_score(query) # 점수가 낮을수록 거리가 가깝다 -> 유사하다
print(docs_and_scores)


db.save_local("faiss_index")

new_db = FAISS.load_local(
    "faiss_index",
    ko,
    allow_dangerous_deserialization=True
)

query = "생성형 AI 관련한 주요 기업과 해당 기업의 생성형 AI의 모델명을 알려줘"
docs = new_db.similarity_search_with_relevance_scores(query, k=3) # 이번에는 유사도 점수가 높을수록 유사함



print("질문: {} \n".format(query))
for i in range(len(docs)):
    print("{0}번째 유사 문서 유사도 \n{1}".format(i+1,round(docs[i][1],2)))
    print("-"*100)
    print(docs[i][0].page_content)
    print("\n")
    print(docs[i][0].metadata)
    print("-"*100)


docs = new_db.max_marginal_relevance_search(
    query,
    k=3,
    fetch_k=10, # 유사도 순으로 상위 fetch_k개의 문서 중 다양성을 유지하여 k개의 답변 추출
    lambda_mult=0.3 # 0 ~ 1 까지 다양성 -> 유사도  중에 가중치
)

print("질문: {} \n".format(query))
for i in range(len(docs)):
    print("{}번째 유사 문서:".format(i+1))
    print("-"*100)
    print(docs[i].page_content)
    print("\n")
    print(docs[i].metadata)
    print("-"*100)
    print("\n\n")
