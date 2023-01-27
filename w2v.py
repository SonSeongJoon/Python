import pandas as pd
import numpy as np
from konlpy.tag import Okt
from numpy import dot
from numpy.linalg import norm
import re
from gensim.models import word2vec

t = Okt()  

# 데이터 전처리
def preprocessing(text):
    text = re.sub('\\\\n', ' ', text)
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]', ' ', text)
    return text

def cosine_similarity(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def get_features(words, model, num_features):
    # 출력 벡터 초기화
    feature_vector = np.zeros((num_features), dtype=np.float32)
    
    num_words = 0
    # 어휘 사전 준비
    index2word_set = set(model.wv.index2word)
    
    for w in words:
        if w in index2word_set:
            num_words = 1
            # 사전에 해당하는 단어에 대해 단어 벡터를 더함
            feature_vector = np.add(feature_vector, model[w])
            
    # 문장의 단어 수만큼 나누어 단어 벡터의 평균값을 문장 벡터로 함
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector





a1= "이 443억원, 코스닥시장에서는 원익피앤이가 375억원을 지급했다. 2021년에는 SK머티리얼즈가 SK와 합병하면서 당시 코스닥시장 주식매수 청구대금의 86.9%인 5713억원을 지급한 바..."
a2 = "지난해 가장 많은 주식매수 청구대금을 지급한 M&A 사유는 합병이었으며, 유가증권시장에서는, 이 443억, 코스닥시장에서는 원익피앤이가 375억원을 지급했다. 다음으로 에이프로젠(222억원), 롯데제과..."
b = "을 지주회사로 하는 지배구조 개편을 단행했습니다. 유통사들이 속속 지배구조 개편에 나서는 데에는 이유가 있을 텐데요. 아무래도 승계..."
c = "앞선 레고랜드 사태 당시의 사례를 살펴보면 충남지역 건설업체인 우석건설을 시작으로 창원 중견기업인, 동원산업 등이 부도 처리된 바 있다. 이 같은 건설업계의 암울한 상황은 체감경기에도 영향을 미치고 있다...."
d = "2023년 1월 식품 상장기업 브랜드평판 30위 순위는 CJ제일제당, 오리온, 농심, 동서, 오뚜기, 삼양식품, 대상, 풀무원, 매일유업, 한일사료, 하림, 남양유업, "
text = []
train_data = [a1,a2,b,c,d]


stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

tokenized_data = []
for sentence in train_data:
    text = preprocessing(sentence)
    tokenized_sentence=t.morphs(text)
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)

num_features = 100    # 워드 백터 특정값 수
min_word_count = 1  # 단어에 대한 최소 빈도 수
num_workers = 4       # 프로세스 개수
context = 10         # 컨텍스트 윈도우 크기
downsampling = 1e-3   # 다운 샘플링 비율

model = word2vec.Word2Vec(tokenized_data,
                        workers=num_workers,
                        size=num_features,
                        min_count=min_word_count,
                        window=context,
                        sample=downsampling)

result = []
for i in tokenized_data:
    result.append(get_features(i, model, num_features))

print(f"1. 문서1-문서5 간 유사도: {cosine_similarity(result[0], result[4])}")
print(f"1. 문서1-문서2 간 유사도: {cosine_similarity(result[0], result[1])}")
print(f"1. 문서2-문서3 간 유사도: {cosine_similarity(result[1], result[2])}")





