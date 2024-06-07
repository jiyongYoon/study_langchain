# RAG (Retrieval Augmented Generation)

- 검색 증강 생성
- 외부 데이터를 참조하여 LLM이 답변할 수 있도록 해주는 프레임워크

## RAG의 구조
  1. 유저가 질문을 하면 
  2. 외부 데이터 저장소(Vector DB, Featre Store 등)에서 사용자의 질문과 유사 문장을 검색하고
  3. Q/A 시스템이 유사문장을 포함한 질문을 LLM에게 프롬프트로 전달한 후
  4. 답변을 받는 형태로 이루어진다.

## Langchain Retreval

- RAG 대부분의 구성요소를 아우르는 말
- 구성요소 하나하나가 RAG의 품질을 좌우함

### Retrieval의 구성요소
- Document Loaders: 문서를 불러오는 역할
- Text Splitters: 문서 텍스트를 분할
- Vector Embeddings: 임베딩 모델을 거쳐 수치화된 데이터들을 벡터 저장소에 저장
- Retrievers: 사용자의 질문과 벡터 저장소에서 가장 유사한 문장을 추출하여 Langchain의 Chain을 통해 LLM의 답변을 생성

### Document Loaders

- `Page_content`(문서의 내용) + `Metadata`(문서의 위치, 제목, 페이지 넘버 등) 의 두 가지 구성요소로 문서를 불러올 수 있음

### Text Splitters

<img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/a31753a3-4674-4966-a646-3a2f81894a96" width="100%"/>

- 토큰 제한이 있는 LLM이 여러 문장을 참고해 답변할 수 있도록 문서를 분할하는 역할을 함
- 문서 자체를 글 덩어리(Chunk)로 자르고, 이를 Vector store에 저장한다.
  - 글 덩어리(Chunk) 하나 당 Vector 하나가 배치된다.
- 대부분의 경우 RecursiveCharacterTextSplitter를 통해 분할
  - 줄바꿈, 마침표, 쉼표 순으로 재귀적으로 분할하므로 max_token을 지켜서 분할하게 됨