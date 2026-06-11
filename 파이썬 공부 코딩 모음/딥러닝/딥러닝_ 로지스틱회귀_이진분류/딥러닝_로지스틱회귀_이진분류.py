import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns',500)

titanicdf=pd.read_csv('/home/jin/deeplearning_prj/20260611/titanic_passengers.csv')
print(titanicdf)
#survived 컬럼 데이터를 타킷으로 활용 (0,1)
#딥러닝에서 타깃 데이터 0과 1 그대로 사용
print(titanicdf.info())

#titanicdf['Survived']=titanicdf['Survived'].map({1:'survival',0:'fail'})
print(titanicdf.head())

#모델 입력 데이터준비
#gender,Age,Pclass 세가지 컬럼 데이터가 생존/ 비생존에 많은 영향을 미침
print(titanicdf['gender'])

#male을 0으로 female을 1로
titanicdf['gender']=titanicdf['gender'].map({'male':0,'female':1})
print(titanicdf)
#age 컬럼에 np.NaN결측치가 존대 ==>결츶치 제거 필요
titanicdf.dropna(subset='Age',inplace=True)
print(titanicdf.head())
print(titanicdf.info())
#age 컬럼에 결츶치를 평균 데이터로 채워서 사용ㅇ

# titanicdf['Age'].fillna(value=titanicdf['Age'].mean(),inplace=True)
# print(titanicdf.head())
# print(titanicdf.info())

print(titanicdf['Pclass'])#1등석과 2등석 데이터만 추출

#ㅍ판다스에 원핫 인코딩으로 변환해주는 매서드=> get_dummies()
#원핫 인코딩은 모든 수치 데이터를 0과 1로만 표현
#1이면 001 2면 010 3이면 100

onehot_pclass=pd.get_dummies(titanicdf['Pclass'],prefix='Class',dtype=int)
print(onehot_pclass)

#axis=1 ==>열축으로 두 dataFrame을 병합해라
titanicdf= pd.concat([titanicdf , onehot_pclass],axis=1)
print(titanicdf)

#Age,gender,Class_1,Class_2 이 4개 컬럼 데이터를 모델 입력 데이터로 사용
#survived 컬럼을 모델 정답 (target 데이터로 사용)
titanicdf_x=titanicdf[['gender','Age','Class_1','Class_2']]
print(titanicdf_x)

print('타깃 데이터 체크 :')
titanicdf_y=titanicdf['Survived']
print(titanicdf_y)
titanicdf.info()
#train/test 분리해서 사용

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=\
    train_test_split(titanicdf_x,titanicdf_y,random_state=47)

print(train_x[:10])

#특성 데이터의 스케일 변환(정규화) ==>표준점수 (각특성 -평균/표준점수)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train_scaled=scaler.fit_transform(train_x)#train 데이터를 정규화 하는 방법을 학습하고 학습이 끝나면 변환 작업을 수행

#변환 작업을 수행
#test 데이터 셋을 transform()만 해서 적응만 해야함
test_scaled=scaler.transform(test_x)
print(train_scaled[:10])


#입력 특성 데이터 4개

#batch_size=16
#epoch=200
#딥러닝 모델 설계는 입룍4, 8,4,1순으로 모델 설계후 fit가지만 진행





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.models import Sequential
logistic_model=Sequential()
logistic_model.add(Dense(units=8,input_dim=4,activation='leaky_relu'))


logistic_model.add(Dense(units=4,activation='leaky_relu'))
logistic_model.add(Dense(units=1,activation='sigmoid'))

# logistic_model.summary()
logistic_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
logistic_model.fit(train_scaled,train_y,batch_size=16,epochs=500,verbose=1)
print(logistic_model.evaluate(test_scaled,test_y)[1])
# for i in range(10):
#     print(f'예측:{logistic_model.predict(test_scaled)[i]}결과:{test_y[i]}')

logistic_model.save('titanic_bestmodel.keras')#모델 전체(네트워크 구조 및 가증치 )저장