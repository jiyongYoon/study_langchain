from llm import openai_embeddings_model

embeddings_model = openai_embeddings_model.generate_embedding_model()

embeddings = embeddings_model.embed_documents(
    [
        "안녕하세요",
        "제 이름은 홍길동입니다.",
        "이름이 무엇인가요?",
        "랭체인은 유용합니다.",
        "Hello World!"
    ]
)

print(
    len(embeddings), # 5 -> 문장 갯수 (임베딩 갯수)
    len(embeddings[0]) # 1536 -> 첫 문장에서의 임베딩 갯수 -> "안녕하세요"라는 문장을 임베딩 했더니 1536차원의 한 행으로 표현이 되었다는 뜻 ([.., .., ..] 이런 좌표가 1536개 있다는 소리겠구만)
)
"""5 1536"""


embedded_query_q = embeddings_model.embed_query("이 대화에서 언급된 이름은 무엇입니까?")
embedded_query_a = embeddings_model.embed_query("이 대화에서 언급된 이름은 홍길동입니다.")
print(len(embedded_query_q), len(embedded_query_a))


from numpy import dot
from numpy.linalg import norm
import numpy as np

# 벡터들 간의 유사도를 확인할 때 가장 많이 사용하는 cos_sim 공식 함수 선언
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))


print(cos_sim(embedded_query_q, embedded_query_a)) # 0.9012151416840549
print(cos_sim(embedded_query_q, embeddings[1])) # 0.8499924168996078
print(cos_sim(embedded_query_q, embeddings[3])) # 0.7753944786592758

