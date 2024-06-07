### 웹
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://n.news.naver.com/mnews/article/092/0002307222?sid=105")

data = loader.load()
print(data[0].page_content)
"""
url에서 가져올 수 있는 모든 텍스트를 다 불러옴
"""


from google.colab import drive
drive.mount('/content/drive')

### PDF
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/drive/MyDrive/BOK이슈노트제2023-26호_수출입경로를 통한 해외 기후변화 물리적 리스크의 국내 파급영향.pdf")
pages = loader.load_and_split()

# 1페이지
pages[0]


### DOCX
from langchain.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("/content/drive/MyDrive/1등_통계+바로쓰기+공모전+수상작.docx")
data = loader.load()


### CSV
from langchain.document_loaders.csv_loader import CSVLoader


loader = CSVLoader(file_path='/content/drive/MyDrive/basketball.csv', csv_args={
    'delimiter': ',', # csv 데이터 구분자
    'quotechar': '"',
    'fieldnames': ['ID', 'Name', 'Position', 'Height', 'Weight', 'Sponsorship Earnings', 'Shoe Sponsor', 'Career Stage', 'Age'] # 열 이름
})

data = loader.load()