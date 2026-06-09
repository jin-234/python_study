import numpy as np
import pandas as pd

beandf=pd.read_csv('/home/jin/deeplearning_prj/20260609/Dry_Bean.csv')
print(beandf)
beandf.info()
# print(beandf.columns)

#데이터 벨류와 데이터 타켓으로 분할
beandf_target=beandf['Class']
beandf_values=beandf[['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
       'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
       'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
       'ShapeFactor3', 'ShapeFactor4']]

#훈련,테스트 데이터셋 분리
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(beandf_values,beandf_target,random_state=42)

#특성 데이터 스케일 변환
from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

scale_train=scale.fit_transform(train_x)
scale_teset=scale.transform(test_x)

#모델 생성 
from sklearn.linear_model import LogisticRegression

lr_model=LogisticRegression(multi_class='multinomial',max_iter=1000,C=20)

#모델 훈련
lr_model.fit(scale_train,train_y)

#모델 평가
print(lr_model.score(scale_train,train_y))
print(lr_model.score(scale_teset,test_y))

#모델 예측
print(lr_model.predict(scale_teset[:3]))