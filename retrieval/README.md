# 1. RAG (Retrieval Augmented Generation)

- 검색 증강 생성
- 외부 데이터를 참조하여 LLM이 답변할 수 있도록 해주는 프레임워크

## RAG의 구조

  <img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/8f9dbdb6-9d82-44af-8ea9-3ff09cb5ac02" width="100%"/>

  1. 유저가 질문을 하면 
  2. 외부 데이터 저장소(Vector DB, Featre Store 등)에서 사용자의 질문과 유사 문장을 검색하고
  3. Q/A 시스템이 유사문장을 포함한 질문을 LLM에게 프롬프트와 함께 전달한 후
  4. 답변을 받는 형태로 이루어진다.


## Langchain Retreval

- RAG 대부분의 구성요소를 아우르는 말
- 구성요소 하나하나가 RAG의 품질을 좌우함

## Retrieval의 구성요소
- Document Loaders: 문서를 불러오는 역할
- Text Splitters: 문서 텍스트를 분할
- Vector Embeddings: 임베딩 모델을 거쳐 수치화된 데이터들을 벡터 저장소에 저장
- Retrievers: 사용자의 질문과 벡터 저장소에서 가장 유사한 문장을 추출하여 Langchain의 Chain을 통해 LLM의 답변을 생성

### 1. Document Loaders

- `Page_content`(문서의 내용) + `Metadata`(문서의 위치, 제목, 페이지 넘버 등) 의 두 가지 구성요소로 문서를 불러올 수 있음
- Web, Pdf, Json, Filesystem, Images, Docs 등 여러가지 Loader들이 있고, 이를 인터페이스화 해놓음

### 2. Text Splitters

<img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/a31753a3-4674-4966-a646-3a2f81894a96" width="100%"/>

- 글 덩어리(Chunk)를 만드는 역할
- 토큰 제한이 있는 LLM이 여러 문장을 참고해 답변할 수 있도록 문서를 분할하는 역할을 함
- 문서 자체를 글 덩어리(Chunk)로 잘라야, 이를 Vector store에 저장하기 좋다.
  - 글 덩어리(Chunk) 하나 당 Vector 하나가 배치된다.
- 대부분의 경우 RecursiveCharacterTextSplitter를 통해 분할
  - 줄바꿈, 마침표, 쉼표 순으로 재귀적으로 분할하므로 max_token을 지켜서 분할하게 됨
- 문서마다 성격이 다르기 때문에 내용의 유사도만큼 쪼갤 수 있으면 그렇게 할 수록 답변 품질이 좋아짐!!

### 3. Vector Embeddings (Text Embeddings)

- 텍스트를 숫자로 변환하여 문장 간의 유사성을 비교할 수 있도록 함
- 임베딩 모델은 분할된 문서를 하나의 Vector Embedding으로 옮기는 역할을 함
- 즉, 답변의 기반이 되는 Documents를 Embedding Vector화 하고, 사용자 질문을 Embedding Vector화 하여 유사성을 비교하여 정답을 찾는 것이 임베딩 모델의 역할이며, 이 모델이 답변의 성능을 크게 좌우하게 된다.
  - 보통 우리의 말은 비정형데이터이다. 어떤 좌표에 나타낼 수 없는데, 이를 임베딩하면 좌표에 나타낼 수 있게 되며, 이를 통해 유사성을 계산할 수 있게 된다.

> 임베딩이란?
> 자연어처리에서 사람이 쓰는 자연어를 기계가 이해할 수 있도록 숫자형태인 vector로 바꾸는 과정 혹은 일련의 전체 과정

- 우리는 보통 임베딩 모델을 대용량의 말뭉치로 훈련이 되어있는 것(`Pre-trained Embedding Model`)을 가져다 쓰게 된다.

<img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/f1e0e8da-eb43-4617-a61c-7e7c160f1367" width="700%"/>

### 4. Vector stores

- 임베딩 된 데이터를 저장하는 저장소
- Pure vector databases
  - 벡터 데이터만 저장 가능
  - ex) Pinecone, qdrant, chroma(오픈소스), Milvus 등... 
- Vector libraries
  - 벡터 유사도를 계산하는 것에 특화된 툴
  - ex) FAISS 등...
- 기본적으로 VectorStore는 벡터를 일시적으로 저장함. 텍스트와 임베딩 함수를 지정하여 from_documents() 함수에 보내면, 지정된 임베딩 함수를 통해 텍스트를 벡터로 변환하고, 이를 임시 db로 생성함.
- 그리고 similarity_search() 함수에 쿼리를 지정해주면 이를 바탕으로 가장 벡터 유사도가 높은 벡터를 찾고 이를 자연어 형태로 출력함.

### 5. Retrievers

- 비정형 쿼리가 주어지면 문서를 반환하는 인터페이스
- 사용자의 질문을 임베딩 모델을 통해 임베딩한 후, 벡터 저장소에 있는 데이터들과 비교한 후 가장 유사성 있는 데이터들을 뽑아서 컨텍스트로 넘기고 사용자의 질문과 함께 프롬프트로 엮어서 LLM에게 넘겨주는 일련의 과정을 진행하게 됨
- Chain
  1. `Stuff documents`

      <img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/2aaafafc-d947-4bb4-95f5-7bd0766539a8" width="100%"/>
     
     - 가장 간단한 타입의 체인이며, 질문은 A이고 Context는 B라는 것을 그대로 LLM에게 넘겨주는 형태
     - LLM에게 던져주는 참고하라는 분할 된 텍스트 청크를 Context에 그대로 주입하는데, 토큰 이슈가 발생할 수 있어 주의를 요함
     
  2. `Map reduce documents chain`
  
      <img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/f2121113-1130-4d7e-b952-1fed7c0ea289" width="100%"/>
  
     - Chunk 하나하나를 요약하여 `Map`을 만든 후(병렬적으로 처리할 수 있음), `Map`을 참고하여 한번 더 요약을 진행하여 결과물을 만들어내는 형태
     - 요약 작업 수행이 필요하므로 결과물을 얻기 위해 다수의 호출이 필요하기 때문에 속도가 느림
     
  3. `Refine documents chain`
  
      <img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/9eb90d62-d0b9-49c1-8d40-f7f004c0fd68" width="100%"/>
  
     - Chunk를 하나하나씩 돌면서 중간 결과물을 계속 다음 답변에 반영하는 형태
     - 좋은 품질의 결과물을 얻을 수 있으나, `Map reduce document chain` 처럼 병렬적으로 처리하지도 않기 때문에 시간이 매우 오래걸림
  4. `Map re-rank documents chain`
  
      <img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/f012b198-7e99-444f-998f-00cabab80128" width="100%"/>
  
     - 사용자의 질문과 Chunk 하나씩을 프롬프트에 넣고 LLM의 답변을 받는데, 이 때 점수까지 함께 받아서 그 중 점수가 가장 높은 답변을 리턴하는 형태
     - Chunk를 순회하면서 누적답변을 생성하여 그중 가장 유사성이 높은 답변을 선별하기 때문에 품질이 뛰어나지만 시간이 오래걸림

### 6. Prompt

- LLM에게 질문할 방식을 설정하는 것
- LangSmith Hub에서 Prompt를 가져다가 쓰면 좋음!!
- [LangSmith Hub Quickstart](https://docs.smith.langchain.com/old/hub/quickstart)

> LangSmith?
> 
> LangChain EcoSystem 구성요소 중 하나로, RAG Chain을 디버깅(추적 및 평가)하도록 도와주는 툴 <br>
> `.env` 파일에 변수를 추가해서 로드하면 바로 사용이 가능하다
> ```
> LANGCHAIN_TRACING_V2=true
> LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
> LANGCHAIN_API_KEY="<your-api-key>"
> ```
> 링크: [랭스미스](https://python.langchain.com/v0.1/docs/langsmith/)

### 7. LLM Answer

- 실제 제공된 데이터를 기반으로 답변을 생성하는 모델
- 답변을 Output Parser를 통해 구조화 할 수 있음

# 2. RAG 고도화

<img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/e87944da-9d72-4dce-93f4-a7c93adb7a89" width="50%"/>

- 사용자 질문의 의도를 파악하여 고품질 답변을 내놓아야 하는데, 사용자 질문과 유사한 정보를 얼마나 잘 뽑아내서 LLM에게 전달해줄 수 있는지가 굉장히 중요하다.

## 프롬프트 엔지니어링을 하지 않아도 잘 답변을 받기 원하는 경우 -> `Multi-query`

- 질문을 잘 알아듣도록 사용자 질문을 여러개의 유사 질문으로 재생성하여 사용함

  <img src="https://github.com/jiyongYoon/study_langchain/assets/98104603/bd6b5a9a-c3a2-4e21-b000-59728f873e4a" width="100%"/>

## 참고 문서에 담기는 문맥의 퀄리티를 높이고 싶은 경우 -> `Parent-Document `

## 시멘틱 검색 말고 쿼리를 필요로 하는 경우 -> `Self-Querying`

- 사용자가 시멘틱 검색, 즉 질문을 조금만 바꾸어도 답변 일관성이 흔들리는 경우가 많음
- 사용자가 어떤 데이터를 필요로 하는 경우, 어떻게 답변 퀄리티를 잘 뽑아내는지 고민이 필요함

## 오래된 자료의 참고를 낮추고 싶은 경우 (ex, 최근 문서를 더 많이 참고하길 원하는 경우) -> `Time-weighted`


---
### 참고 유튜브 채널: 
- [모두의AI 유튜브](https://www.youtube.com/@AI-km1yn)
- [테디노트](https://youtu.be/1scMJH93v0M?si=fVHZFkuk72lHZ1na)