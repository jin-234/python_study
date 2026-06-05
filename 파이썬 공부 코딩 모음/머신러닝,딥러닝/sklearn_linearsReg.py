import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #선형회귀 모델


#농어길이데이터 ( 캐글Fish Market 데이터참조)

perch_length=np.array([8.4,13.7,15.0,16.2,17.4,18.0,18.7,19.0,19.6,20.0,21.0,
21.0,21.0,21.3,22.0,22.0,22.0,22.0,22.0,22.5,22.5,22.7,
23.0,23.5,24.0,24.0,24.6,25.0,25.6,26.5,27.3,27.5,27.5,
27.5,28.0,28.7,30.0,32.8,34.5,35.0,36.5,36.0,37.0,37.0,
39.0,39.0,39.0,40.0,40.0,40.0,40.0,42.0,43.0,43.0,43.5,
44.0])
# 농어무게데이터 (캐글FishMarket 데이터참조)

perch_weight=np.array([5.9,32.0,40.0,51.5,70.0,100.0,78.0,80.0,85.0,85.0,110.0,
115.0,125.0,130.0,120.0,120.0,130.0,135.0,110.0,130.0,
150.0,145.0,150.0,170.0,225.0,145.0,188.0,180.0,197.0,
218.0,300.0,260.0,265.0,250.0,250.0,300.0,320.0,514.0,
556.0,840.0,685.0,700.0,700.0,690.0,900.0,650.0,820.0,
850.0,900.0,1015.0,820.0,1100.0,1000.0,1100.0,1000.0,
1000.0])

#농어 길이/농어 무게를 train/test 데이터로 분리
train_x,test_x,train_y,test_y=train_test_split(perch_length,perch_weight,random_state=42)

print(train_x.shape,train_y.shape)
#1치원 ==>2차원으로 변경
train_x=train_x.reshape(-1,1)
test_x=test_x.reshape(-1,1)
print(train_x.shape,train_y.shape)

lrmodel=LinearRegression()

lrmodel.fit(train_x,train_y) #학습 완료시 최적의 w(가중치/기울기),b(절편)찾아짐


#lrmodel.coef_:기을기(a,w)
#lrmodel.intercept_ :절편(b)
print(lrmodel.coef_,lrmodel.intercept_)

#모델 성능평가
print(lrmodel.score(test_x,test_y))

#모델 예측

pred=lrmodel.predict([[30]])
print(pred)

print(lrmodel.coef_*30+lrmodel.intercept_)#wx + b

#머신러닝 모델의 회귀선
plt.plot([15,50],[lrmodel.coef_*15+lrmodel.intercept_,lrmodel.coef_*50+lrmodel.intercept_])
plt.scatter(train_x,train_y)
plt.savefig('fish_linear.jpeg')
