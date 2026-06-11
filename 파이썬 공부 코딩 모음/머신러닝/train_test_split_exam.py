#KNeighborsClassifier 분류모델
#기본값 5개의 이웃 데이터를 기준으로
#주위의 다수를 차지하는 데이터를 정답으로 예측

import numpy as np
from sklearn.neighbors import KNeighborsClassifier #knn분류모델
import matplotlib.pyplot as plt#시각화
from sklearn.model_selection import train_test_split#데이터셋 분할


#1.학습시킬 데이터 데이터셋 준비
# 도미데이터 ( 캐글Fish Market 데이터 참조 )
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
# 빙어데이터 ( 캐글Fish Market 데이터 참조 )

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length=bream_length+smelt_length
weight=bream_weight+smelt_weight

# arr=np.column_stack(([3,4,5,],[8,2,5]))
# print(arr)
fish_data=np.column_stack((length,weight))
print(fish_data[:5])
print(len(fish_data))#49개
#np.concatenate는 두 넘파이 배열을 병합해라
# arr=np.ones((2,3))#((2,3))넘파이 배욜을 만들고 각 요소를 모두 1로 채워
# print(arr)
# arr=np.zeros((2,3))#((2,3))넘파이 배욜을 만들고 각 요소를 모두 0로 채워
# print(arr)

# arr=np.ones(35)#길이가 35인 1차원 넘파이 배열을 만들고 각 요소를 모두 1로
# print(arr)
# arr=np.zeros(14)#길이가 35인 1차원 넘파이 배열을 만들고 각 요소를 모두 1로
# print(arr)

fish_target=np.concatenate(([np.ones(35),np.zeros(14)]))
print(fish_target)#49개

print(fish_data[:5])
#원 데이터를 train/test로 분할(기본 shuffle 제공)
train_x,test_x,train_y,test_y=train_test_split(fish_data,fish_target,stratify=fish_target,
                 random_state=42)

# print(train_x[:5])
# print(len(train_x)) #36개 사용
# print(len(train_y))
# print(len(test_x))#13개 사용
# print(len(test_y))

#train/test 분할한 데이터 셋 준비 완료

#모델 준비(머신러닝)
knnmodel=KNeighborsClassifier() #knn 분류 모델 완성
#모델 학습==>train 데이터만 활용
knnmodel.fit(train_x,train_y)
#모델 성능 평가==> test 데이터를 활용해서
print('acc : ',knnmodel.score(test_x,test_y))
#예측
pred = knnmodel.predict([[25,150]])#임의의 한 데이터를 예측할때도 2차원 배열 형태로 전달
print(pred)#0==> smelt , 1==>bream
# print(train_x)
# print(train_x[ : , 0 ])
# print(train_x[ : , 1 ])

#[25,150] 데이터 기준 거리가 가까운 k개(5개)의 데이터 index 반환
distance,index=knnmodel.kneighbors([[25,150]])
print(distance)
print(index)

print(train_x)
#현제 train_x처럼 데이터의 범주가 클 경우 KNN모델의 문제가 발생할 수 있음
#==> train_x 데이터의 범주를 정규화 시켜서 사용해야함

#표준점수 정규화 ==> standard scaler
plt.scatter(train_x[index,0],train_x[index,1],marker='D',c='red')

plt.scatter(train_x[ : , 0 ],train_x[:,1])
plt.scatter(25,150,c='blue',marker='^')
plt.savefig('fish_data.jpeg')
