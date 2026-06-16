import numpy as np
import pandas as pd

movie_df=pd.read_csv('/home/jin/deeplearning_prj/20260616/IMDB Dataset.csv')
movie_df=movie_df[:200].copy()
# print(movie_df)

from sklearn.naive_bayes import MultinomialNB#다항분포 나이브 베이즈

#sentiment 컬럼 라벨을 수치 데이터로 변경
import re
movie_df['review']=movie_df['review'].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))
movie_df['sentiment']=movie_df['sentiment'].map({'positive':1,'negative':0})
train_x=movie_df['review']
train_y=movie_df['sentiment']
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

train_cv=cv.fit_transform(train_x)
train_cv_encode=train_cv.toarray()
print(train_cv_encode)
print(train_cv_encode[0])
# print(cv.get_feature_names_out())

from sklearn.naive_bayes import MultinomialNB

mnb=MultinomialNB()
mnb.fit(train_cv_encode,train_y)
print('acc',mnb.score(train_cv_encode,train_y))

print(mnb.predict(train_cv_encode))
newdata=([     "The movie was amazing and the acting was outstanding",
    "I really enjoyed this film and would watch it again",
    "The story was touching and the characters were memorable",
    "An excellent movie with beautiful cinematography",
    "This was one of the best movies I have seen this year",
    "The plot was engaging from beginning to end",
    "The performances of the actors were truly impressive",
    "I loved every moment of this wonderful film",
    "The movie delivered a powerful and inspiring message",
    "A fantastic film with great storytelling and direction"])
newdata=cv.transform(newdata)
newdata_cv=newdata.toarray()


print(mnb.predict(newdata_cv))
# for i in newdata_cv:
#     if mnb.predict([i])==1:
#         print('positive')
#     else:
#         print('negative')
        