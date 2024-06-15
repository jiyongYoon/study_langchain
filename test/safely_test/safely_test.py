from numpy import dot
from numpy.linalg import norm


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


print("############## Document 스키마 확인 ##############")
from langchain.schema import Document

schema = Document.schema_json()
print(schema)


print("############## 임베딩 모델 ##############")
from gpt_llm import openai_embeddings_model, openai_chatgpt_llm

embedding_model = openai_embeddings_model.generate_embedding_model()


print("############## 단어 임베딩 ##############")
# text = ["산업안전 관리문서", "지게차 운행 일지", "산업안전 경영목표", "경영팀 비용 정산"]
# embed_documents = embedding_model.embed_documents(text)
#
# print("임베딩된 문서(단어) 갯수: %d" % len(embed_documents))
# print("임베딩된 문서(단어) 차원: %d" % len(embed_documents[0]))
# for i in range(len(embed_documents)):
#     print("%d번 문서(단어) 임베딩 내용: %s" % (i + 1, embed_documents[i]))

collection_name = "company_id_1"

print("############## 벡터 저장소 저장 ##############")
from langchain_community.vectorstores import Chroma

# db = Chroma.from_texts(
#     text,
#     embedding_model,
#     persist_directory="./chroma_db",
#     collection_name=collection_name,
# )
# print(db)

print("############## 질문 임베딩 ##############")
question = "안전 경영목표"
query = embedding_model.embed_query(question)
print("임베딩된 쿼리의 차원: %d" % len(query))

# for i in range(len(embed_documents)):
#     print("%d번 문서(단어) 유사점수: %f" % (i + 1, cos_sim(embed_documents[i], query)))

print("############## 벡터 저장소에서 데이터 꺼내기 ##############")
find_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name=collection_name,
)

print("############## 유사도 확인 ##############")
relevance_scores = find_db.similarity_search_with_relevance_scores(question, k=len(find_db))

print("가장 유사한 문서:\n\n {}\n\n".format(relevance_scores[0][0].page_content))
print("문서 유사도:\n {}".format(relevance_scores[0][1]))
print("전체 찾은 문서: ")
print(relevance_scores)


print("############## LLM으로 질문하기 ##############")
chatgpt = openai_chatgpt_llm.generate_llm()
retriever = find_db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': len(find_db), 'fetch_k': len(find_db)}
)
# TODO retriever 타입 학습 필요!

from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=chatgpt,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True, # 출처를 확인하기 위한 옵션
)

query = "검색 대상 중에 '산업안전 경영목표'라는 단어가 있는지 확인해줘"
result1 = qa.invoke({"query": query})
print("Q: " + query)
print(result1)

query2 = "검색 대상 중에 '작업 일지'라는 단어가 있는지 확인해줘"
result2 = qa.invoke({"query": query2})
print("Q: " + query2)
print(result2)

