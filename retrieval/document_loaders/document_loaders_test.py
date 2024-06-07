### 웹
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://n.news.naver.com/mnews/article/092/0002307222?sid=105")

data = loader.load()
print(data[0].page_content)
"""
url에서 가져올 수 있는 모든 텍스트를 다 불러옴
"""


### PDF
import os
from langchain_community.document_loaders import PyPDFLoader

project_directory = 'D:\\dev_yoon\\py\\study_langchain'
pdf_filename = 'lorem-pdf.pdf'
pdf_filepath = os.path.join(project_directory, pdf_filename)

pdf_content=''
try:
    with open(pdf_filepath, 'rb') as pdf_file:
        pdf_content = pdf_file.read()
except FileNotFoundError:
    print("파일을 찾을 수 없습니다.")

# loader = PyPDFLoader(pdf_content)
loader = PyPDFLoader(pdf_filepath)
pages = loader.load_and_split()

# 1페이지
print(pages[0])


### DOCX
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("/content/drive/MyDrive/1등_통계+바로쓰기+공모전+수상작.docx")
data = loader.load()


### CSV
from langchain_community.document_loaders.csv_loader import CSVLoader


loader = CSVLoader(file_path='/content/drive/MyDrive/basketball.csv', csv_args={
    'delimiter': ',', # csv 데이터 구분자
    'quotechar': '"',
    'fieldnames': ['ID', 'Name', 'Position', 'Height', 'Weight', 'Sponsorship Earnings', 'Shoe Sponsor', 'Career Stage', 'Age'] # 열 이름
})

data = loader.load()