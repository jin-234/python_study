import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
wine_df=pd.read_csv('/home/jin/deeplearning_prj/20260611/wine_dataset.csv')
#와인 데이터 이진 딥러닝 분류 모델 설계
#학습까지

wine_df['style']=wine_df['style'].map({'red':0,'white':1})
# print(wine_df)
# wine_df.info()
wine_df_value=wine_df.iloc[:,:12]
wine_df_target=wine_df.iloc[:,12]
# print(wine_df_value)
# print(wine_df_target)
train_x,test_x,train_y,test_y=train_test_split(wine_df_value,wine_df_target,random_state=42)

scaler=StandardScaler()
train_scale=scaler.fit_transform(train_x)
test_scale=scaler.transform(test_x)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(units=24,input_dim=12,activation='leaky_relu'))
model.add(Dense(units=12,activation='leaky_relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')
model.fit(train_scale,train_y,batch_size=64,epochs=10,verbose=1)

print(model.evaluate(test_scale,test_y)[1])

num=int(input('몇번째 예시를 예측할건지'))

pred=model.predict(test_scale[num:num+1])
if pred >= 0.5:
    result=1
else:
    result=0
print(f'예측값 : {result}, 실제값 : {test_y.iloc[num]}')
