import dotenv
import os

# project_directory = 'D:\dev_yoon\py\study_langchain'
dotenv.load_dotenv()
project_directory = os.environ['PROJECT_DIRECTORY']
filename = 'lorem.txt'
filepath = os.path.join(project_directory, filename)

with open(filepath, encoding='utf-8') as f:
    lorem_txt = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200, # chunk 앞 뒤가 겹치는 부분 -> 맥락 이해를 위해 필요함 (앞 뒤 일정부분 내용을 참고해야 하는 경우가 생기기 때문)
    length_function=len, # chunk 단위 - len은 글자 길이가 됨
)

texts = text_splitter.split_text(lorem_txt)
print(texts[0])
print("-"*100)
print(texts[1])
print("-"*100)
print(texts[2])


char_list = []
for i in range(len(texts)):
    char_list.append(len(texts[i]))
print(char_list)
"""
설정한 chunk size를 넘지 않음
[1000, 987, 996, 955, 957, 931, 943, 930, 993, 958, 987, 981, 991, 999, 903, 974, 951, 994, 958, 993, 972, 992, 950, 903, 981, 485]
"""



from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON))
print(RecursiveCharacterTextSplitter.get_separators_for_language(Language.JAVA))


PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs[0])
"""
page_content='def hello_world():\n    print("Hello, World!")'
"""
print(python_docs[1])
"""
page_content='# Call the function\nhello_world()'
"""

print("-"*100)
print("-"*100)

### 토큰 단위 텍스트 분할기
### 결국 LLM api들은 토큰 단위로 설정이 되다보니, 이렇게 분할하는 것이 가장 일반적이여진다.
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base") # gpt 모델들에 대해서 토크나이징할 때 사용되는 임베딩 모델

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

print(tiktoken_len(texts[1]))



### 새롭게 text_splitter 선언
text_splitter2 = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=tiktoken_len, # tiktoken_len을 사용
)

from langchain.schema import Document

lorem_doc = [Document(page_content=lorem_txt)]

tiktoken_texts = text_splitter2.split_documents(lorem_doc)
print(tiktoken_texts[0])
print("-"*100)
print(tiktoken_texts[1])
print("-"*100)
print(tiktoken_texts[2])

print("-"*100)
print("-"*100)
print(len(tiktoken_texts))

token_list = []
for i in range(len(tiktoken_texts)):
    token_list.append(tiktoken_len(tiktoken_texts[i].page_content))
print(token_list)
