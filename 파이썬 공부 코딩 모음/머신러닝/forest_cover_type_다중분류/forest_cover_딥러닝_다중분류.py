import numpy as np
import pandas as pd
#파일 다운로드 위치 : https://www.kaggle.com/datasets/mohsinsherazds/forest-covertype

forset_df=pd.read_csv('/home/jin/deeplearning_prj/20260613 개인복습/forest_cover_type.csv')
# print(forset_df.columns)
# forset_df.info()

# ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
#        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
#        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
#        'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Soil_Type1',
#        'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
#        'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
#        'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
#        'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
#        'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
#        'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
#        'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
#        'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
#        'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
#        'Soil_Type40', 'Wilderness_Area2', 'Wilderness_Area3',
#        'Wilderness_Area4', 'Cover_Type'],

#컬럼 15개
forest_values=forset_df[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4', 'Cover_Type']]

#컬럼 40개
forest_target=forset_df[['Soil_Type1',
       'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
       'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40']]

#테스트/트레인 데이터 분리

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(forest_values,forest_target,random_state=42)

#특정데이터 스케일링

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()
train_scale=scale.fit_transform(train_x)
test_scale=scale.transform(test_x)

#모델 생성
from keras.models import Sequential
from keras.layers import Dense

#dropout
from keras.layers import Dropout
model=Sequential()
model.add(Dense(units=512,input_dim=15,activation='leaky_relu'))

model.add(Dense(units=256,activation='leaky_relu'))
model.add(Dense(units=128,activation='leaky_relu'))
Dropout(0.1)
model.add(Dense(units=64,activation='leaky_relu'))

model.add(Dense(units=40,activation='softmax'))

# model.summary()

#callback 생성
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

forest_path='/home/jin/deeplearning_prj/20260613 개인복습/forest_model.keras'
checkpoint=ModelCheckpoint(save_best_only=True,filepath=forest_path,monitor='val_loss',verbose=1)
erlystop=EarlyStopping(monitor='val_loss',restore_best_weights=True,patience=3)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

#모델 학습
model.fit(train_scale,train_y,validation_data=(test_scale,test_y),verbose=1,epochs=1000,batch_size=256,callbacks=[checkpoint,erlystop])

#딥러닝 3층일때accuracy: 0.6467

#unit80짜리 추가후 0.7096
#80밑에 60짜리 추가후0.7140

#batch_size=1000=>batch_size=256 변환후0.7392 
#256/128/64/40변환후0.7686

#유닛128밑에 dropout 0.1추가0.7972
#유닛 256밑에 dropout 0.1추가0.7807 
#취소후 유닛64밑에 dropout 0.1,추가0.7921 
# 유닛 256밑에 dropout 0.1추가0.7688
#dropout 유닛 128제외하고 삭제후 patience=5 로 변경 0.8106
#patience=3으로 변경후 입력 유닛 512추가
