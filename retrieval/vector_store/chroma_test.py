import tiktoken


tokenizer = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)



### PDF
import os
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber

project_directory = 'D:\\dev_yoon\\py\\study_langchain'
pdf_filename = 'generate-aipdf.pdf'
pdf_filepath = os.path.join(project_directory, pdf_filename)

# pdf_content=''
# try:
#     with open(pdf_filepath, 'rb') as pdf_file:
#         pdf_content = pdf_file.read()
# except FileNotFoundError:
#     print("파일을 찾을 수 없습니다.")

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

# print(pdf_content)

# loader = PyPDFLoader(pdf_filepath)
# pages = loader.load_and_split()
#
#
# print(pages)


### 특수문자 제거
import re


def remove_special_characters(text):
    # 특수 문자 제거 (정규 표현식 사용)
    return re.sub(r'[^A-Za-z0-9가-힣\s]', '', text)


cleaned_text = remove_special_characters(pdf_content)


### Document 형태로 변환
from langchain.schema import Document


pages = [Document(page_content=cleaned_text)]


### text_split
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    length_function=tiktoken_len,
)


split_docs = text_splitter.split_documents(pages)
# split_text = text_splitter.split_text(pdf_content)

# create the open-source embedding function
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


# load it into Chroma
db = Chroma.from_documents(split_docs, hf)
# text_db = Chroma.from_texts(split_text, hf)

# query it
query = "생성형 ai가 발전함에 따라 조심해야 하는 것이 어떤 것들이 있을까?"
find_docs = db.similarity_search(query)
# find_docs = text_db.similarity_search(query)

# print results
print(find_docs[0])
# print(find_docs[0].page_content)
print(find_docs)


# tiktoken_len(find_docs[0])


### 영구적으로 저장하고 싶은 경우 벡터 저장소 로컬 저장
# save to disk
db2 = Chroma.from_documents(
    find_docs,
    hf,
    persist_directory="./chroma_db" # 해당 경로로 저장할 것 -> sqlite로 저장됨
)
docs = db2.similarity_search(query)


# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=hf)
docs = db3.similarity_search(query)
print(docs[0].page_content)


# 유사도 점수만 확인하고 싶을 때
docs = db3.similarity_search_with_relevance_scores(query, k=3)


print("가장 유사한 문서:\n\n {}\n\n".format(docs[0][0].page_content))
print("문서 유사도:\n {}".format(docs[0][1]))