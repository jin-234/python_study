from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
mnist= load_digits()#0-9 손글씨 숫자 데이터셋 (분류용)

print(mnist['data'][:3])#데이터셋 크기 확인하기
print(len(mnist['data']))#1797개의 데이터
print(mnist['target'])
print(len(mnist['target']))

features=mnist['data']#1797데이터의 8*8이미지 크기
labels=mnist['target']

from sklearn.model_selection import cross_validate#교차검증
RFmodel=RandomForestClassifier()#랜덤 포레스트
RF_scores=cross_validate(RFmodel,features,labels,cv=10)#랜덤 포레스트 10_fold 교차 검증
print(RF_scores['test_score'])#랜덤포레스트 앙상블 검증평가 점수

DT_scores=cross_validate(tree.DecisionTreeClassifier(),features,labels,cv=10)
print(DT_scores['test_score'])

import numpy as np
print('random_forest accuracy : ',np.mean(RF_scores['test_score']))
print('decision tree accuracy:',np.mean(DT_scores['test_score']))
#랜덤 포레스트 앙상블이 별도의 하이퍼 파라미터 설정 없는 의사결정 트리보다 월등히 높은 성능 발휘

import pandas as pd
df=pd.DataFrame({'random_forest':RF_scores['test_score'],'decision_tree':DT_scores['test_score']})

print(df)
df.plot()
plt.savefig('./randforest.jpeg')