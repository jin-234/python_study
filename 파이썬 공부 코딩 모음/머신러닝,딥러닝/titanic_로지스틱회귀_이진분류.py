import numpy as np
import pandas as pd

pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns',500)

titanicdf=pd.read_csv('/home/jin/deeplearning_prj/20260608/titanic_passengers.csv')
print(titanicdf)
#survived 컬럼 데이터를 타킷으로 활용 (0,1)
#머신러닝 sklearn은 타킷이 문자열 이어도 성능평가가 가능
#surviveㅇ 컬럼데이터 변경
#0==>fail 1==>survival
print(titanicdf.info())

titanicdf['Survived']=titanicdf['Survived'].map({1:'survival',0:'fail'})
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

titanicdf_y=titanicdf['Survived']
print(titanicdf_y)
#train/test 분리해서 사용

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=\
    train_test_split(titanicdf_x,titanicdf_y,random_state=42)

print(train_x[:10])

#특성 데이터의 스케일 변환(정규화) ==>표준점수 (각특성 -평균/표준점수)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
train_scaled=scaler.fit_transform(train_x)#train 데이터를 정규화 하는 방법을 학습하고 학습이 끝나면 변환 작업을 수행

#변환 작업을 수행
#test 데이터 셋을 transform()만 해서 적응만 해야함
test_scaled=scaler.transform(test_x)
print(train_scaled[:10])
#모델 생성 및 평가
#로지스틱 회귀(분류)모델 준비
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
#학습
lr_model.fit(train_scaled,train_y)
#모델 성능 평가

print('test acc:',lr_model.score(test_scaled,test_y))
print('train acc:',lr_model.score(train_scaled,train_y))

#가중치(w), 절편(b)
#:coef_,inter_
print(lr_model.coef_,lr_model.intercept_)


