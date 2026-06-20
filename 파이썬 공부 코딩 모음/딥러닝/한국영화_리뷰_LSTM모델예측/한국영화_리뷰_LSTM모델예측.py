from tensorflow.keras.models import load_model
best_model = load_model('/home/jin/deeplearning_prj/movie_review_bestmodel.keras') # 앞서 저장한 모델 로드

import numpy as np
import pandas as pd


pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 50)
pd.set_option('display.colheader_justify','center')  # 컬럼 중앙 출력

train_df=pd.read_csv('/home/jin/deeplearning_prj/202060618/train_stopwords_reviews.csv',index_col=0)




train_df.dropna(how='any',inplace=True)
#Tokenizer=> 특정 단어를 특정 수치(정수)로 매핑 치환하는 역할
from tensorflow.keras.preprocessing.text import Tokenizer
#pad_sequences=> 고정길이 정수 벡터를 생성할때 사용
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_size=11775 #==imdb의 num_word의 역할

tokenizer=Tokenizer(word_size)

tokenizer.fit_on_texts(train_df['document'])
#테스트데이터 정확도 성능 평가

# 새로운 리뷰 데이터 예측
from konlpy.tag import Okt
import os
import re
# 한국어 토근화 및 패딩 처리 위해 Okt 클래스 추가
okt = Okt() # KoNLPy 제공 형태소 분석기
# 조사 위주의 한국어 불용어 제거 리스트
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

def new_review_predict(review_string):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]','', review_string) # 한국어와 공백 이외의내용삭제
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어제거
    print(new_sentence) # ['영화', '굿', '잼']
    # [new_sentence] : 불용어 처리된 단어 리스트를 정수 인코딩 sequences 데이터 형성을# 위해 하나로 묶어서([ ]) 변환해 줘야함
    encoded = tokenizer.texts_to_sequences( [new_sentence] ) # 정수 인코딩
    print(encoded) # [[1, 363, 334]] 
    sentence_padding = pad_sequences(encoded, maxlen = 30) # 패딩 적용 동일 길이 Sequences 형성
    print(sentence_padding)
    #[[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 1 363 334]] 
    score = float(best_model.predict(sentence_padding) ) # new_sentence 예측
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100)) 
    else: 
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
        
        
# new_review_predict('이 영화 굿 잼')
# new_review_predict('이렇게 재미없는 영화는 처음')
# new_review_predict('뭐 이런 영화가 다 있어')
# new_review_predict('에잇 돈 날렸네')
# new_review_predict('이 영화 꼭 추천 도장 꽉!')

new_review_predict('배우들의 연기가 자연스럽고 스토리 전개도 탄탄해서 끝까지 몰입하며 봤다.')
new_review_predict('스토리가 너무 뻔해서 중간부터 결말이 예상됐다.')
new_review_predict('영상미가 뛰어나고 음악도 분위기와 잘 어울렸다.')#안맞음
new_review_predict('배우들의 연기는 괜찮았지만 이야기 전개가 산만하고 개연성이 부족했다.')
new_review_predict('캐릭터들의 매력이 살아 있고 유머와 감동의 균형이 좋았다.')#부적ㅇ으로나옴
new_review_predict('화려한 영상에만 집중한 느낌이다.')
