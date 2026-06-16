from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
mnist=load_digits()#0-9손글씨 숫자 데이터셋
print(mnist['data'][-3:])#데이터 개수 확인하기 (1797개)
print(mnist['target'][-3:])#데이터 1797


features=mnist['data']#1797개의  8*8이미지 데이터셋
labels=mnist['target']

RFmodel=RandomForestClassifier()
RFmodel.fit(features,labels)#학습시 라벨(labels)은 분할 기준(규칙)을 평가(결정)할 기준이됨,즉 정보이득(불순도 감소)량을 계산해 가장 좋은 분할 기준을 선택(결정)하는 역할

print('score:',RFmodel.score(features,labels))

predicted=RFmodel.predict(features[-5:])
print('labels:',labels[-5:])
print('pre:',predicted)

tempdata=[ 0., 0.,12.,16.,16.,16.,14., 0.,
    0., 0., 0., 0., 4.,12.,12., 0.,
    0., 0., 0., 0., 8.,12., 0., 0.,
    0., 0., 0., 8.,12., 0., 0., 0.,
    0., 0., 4.,12., 0., 0., 0., 0.,
    0., 0.,12., 8., 0., 0., 0., 0.,
    0., 8.,12., 0., 0., 0., 0., 0.,
    0., 4., 0., 0., 0., 0., 0., 0.]

import numpy as np
import matplotlib.pyplot as plt
temparr=np.array(tempdata)#reshape 하기위해 numpy array로 변환
print(temparr)
temp_pred=RFmodel.predict([temparr])

print('temp_pred',temp_pred)

plt.imshow(temparr.reshape(8,8))
plt.savefig('./randforest_hand')