from datetime import datetime, timedelta

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from gpt_llm import openai_chatgpt_llm
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)


# Initialize the vectorstore as empty
embedding_size = 768 # 각 임베딩 모델에 해당하는 embedding_size를 직접 넣어주어야 함
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(ko_embedding, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01, # decay_rate가 클수록 시간값에 대한 영향을 덜 받는다. 낮을수록 최신 데이터에서 검색한다.
    k=1
)

yesterday = datetime.now() - timedelta(days=1)
one_month_ago = datetime.now() - timedelta(days=30)
retriever.add_documents(
    [Document(page_content="영어는 훌륭합니다.", metadata={"last_accessed_at": yesterday})] # 어제 접근한 문서로 세팅
)
retriever.add_documents(
    [Document(page_content="한국어는 훌륭합니다", metadata={"last_accessed_at": one_month_ago})] # 한달 전에 접근한 문서로 세팅
)

# "Hello World" is returned first because it is most salient, and the decay rate is close to 0., meaning it's still recent enough
print(retriever.get_relevant_documents("영어가 좋아요"))

### 결과 부분은 현재 제대로 나오지는 않음. 실제 사용할 때 다시 테스트해보고 사용해야 함.

""" 0.01
[Document(page_content='영어는 훌륭합니다.', metadata={'last_accessed_at': datetime.datetime(2024, 6, 27, 17, 12, 19, 96163), 'created_at': datetime.datetime(2024, 6, 27, 17, 12, 18, 200109), 'buffer_idx': 0})]
"""

""" 0.99
[Document(page_content='한국어는 훌륭합니다', metadata={'last_accessed_at': datetime.datetime(2024, 6, 27, 17, 11, 53, 453239), 'created_at': datetime.datetime(2024, 6, 27, 17, 11, 53, 389622), 'buffer_idx': 1})]
"""