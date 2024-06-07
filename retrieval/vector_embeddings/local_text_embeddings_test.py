from numpy import dot
from numpy.linalg import norm


# 벡터들 간의 유사도를 확인할 때 가장 많이 사용하는 cos_sim 공식 함수 선언
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))


### use local embedding model from huggingface
### !pip install sentence_transformers
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings

# model_name = "BAAI/bge-small-en"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True} # 임베딩 백터 정규화 작업 (단위가 같아야지 비교가 되지!)
# hf = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )
#
# embeddings = hf.embed_documents(
#     [
#     "today is monday",
#     "weather is nice today",
#     "what's the problem?",
#     "langhcain in useful",
#     "Hello World!",
#     "my name is morris"
#     ]
# )
#
# BGE_query_q = hf.embed_query("Hello? who is this?")
# BGE_query_a = hf.embed_query("hi this is harrison")
#
# print(cos_sim(BGE_query_q, BGE_query_a))
# print(cos_sim(BGE_query_q, embeddings[1]))
# print(cos_sim(BGE_query_q, embeddings[5]))


### 한국어 사전학습 모델 임베딩

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
ko = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

sentences = [
    "안녕하세요",
    "제 이름은 홍길동입니다.",
    "이름이 무엇인가요?",
    "랭체인은 유용합니다.",
    "홍길동 아버지의 이름은 홍상직입니다."
    ]

ko_embeddings = ko.embed_documents(sentences)

q = "홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?"
a = "홍길동의 아버지는 엄했습니다."
ko_query_q = ko.embed_query(q)
ko_query_a = ko.embed_query(a)

print("질문: {} \n".format(q), "-"*100)
print("{} \t\t 문장 유사도: ".format(a), round(cos_sim(ko_query_q, ko_query_a),2))
print("{}\t\t\t 문장 유사도: ".format(sentences[1]), round(cos_sim(ko_query_q, ko_embeddings[1]),2))
print("{}\t\t\t 문장 유사도: ".format(sentences[3]), round(cos_sim(ko_query_q, ko_embeddings[3]),2))
print("{}\t 문장 유사도: ".format(sentences[4]), round(cos_sim(ko_query_q, ko_embeddings[4]),2))