import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import re
spamdf=pd.read_csv('/home/jin/deeplearning_prj/20260616/spam.csv')

spamdf=spamdf[:].copy()
print(spamdf)

spamdf['Category']=spamdf['Category'].map({'ham':0,'spam':1})
spamdf['Message']=spamdf['Message'].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))
print(spamdf)

cv=CountVectorizer(binary=True)

train_x=cv.fit_transform(spamdf['Message'])

train_x=train_x.toarray()
print(train_x)
train_y=spamdf['Category']
bnb=BernoulliNB()
train_y=train_y.astype('int')

bnb.fit(train_x,train_y)
print(bnb.score(train_x,train_y))

#새로운 이메일 데이터 하나 추가 예측
newdata=cv.transform([    'congratulations you are the winner claim your free prize now',
    'free cash reward waiting for you call now',
    'urgent your account has been selected for a special offer',
    'are we still meeting for lunch tomorrow',
    'please send me the project report by tonight',
    'thank you for your help yesterday',
    'congratulations claim your cash reward immediately',
    'dont forget our class starts at ten tomorrow'])
newdata=newdata.toarray()
pred=bnb.predict(newdata)
# print(pred)
for i in pred:
    if i==0:
        print('ham')
    else:
        print('spam')